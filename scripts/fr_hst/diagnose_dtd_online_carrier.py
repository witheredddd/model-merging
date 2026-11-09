"""Cross-image transfer, future-step reuse and matched residual absorption."""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from run_dtd_carrier_diagnostics import Operator, metrics
from train_dtd_online_carrier import ARMS, AUDIT_ANCHORS, AUDIT_STEPS, coordinates, response
from train_dtd_rolling_innovation import probe_mode
from train_eight_expert_queue import HEADS, ROOT, build_model, existing, read, splits


MATCH_COSINE = .6
HORIZON = 4


def independent_probes(encoder, old_groups):
    fit, _ = splits('DTD', encoder.val_preprocess, encoder.val_preprocess)
    used = {i for group in old_groups for i in group}
    available = [i for i, index in enumerate(fit.indices) if index not in used]
    order = torch.randperm(len(available), generator=torch.Generator().manual_seed(20261110))[:64]
    samples, ids = [], []
    for group in order.split(32):
        indices = [available[i] for i in group]
        rows = [fit[i] for i in indices]
        samples.append((torch.stack([x for x, _ in rows]).cuda(), torch.tensor([y for _, y in rows], device='cuda')))
        ids.append([fit.indices[i] for i in indices])
    assert len(set(ids[0]) | set(ids[1]) | used) == 160
    return samples, ids


def unit(v):
    return v / v.norm().clamp_min(1e-12)


def explained(actual, target):
    return float(1. - np.linalg.norm(actual-target)**2 / max(np.linalg.norm(target)**2, 1e-20))


def cosine(a, b):
    return float(np.dot(a,b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-20))


def measure(op, probes, frame, bend, v, rows, targets, gradients, before):
    width = v.norm()
    result, fitted = [], []
    for fit in (0, 1):
        audit = 1 - fit
        x, y = probes[audit]
        signed = (frame @ gradients[fit]).double().cpu().numpy()
        _, coef, _, _ = coordinates(rows[fit], targets[fit], signed, bend)
        direction = torch.as_tensor(coef, dtype=frame.dtype, device=frame.device) @ frame
        after = op.evaluate(x, width * direction)
        result.append(dict(fit_group=fit, audit_group=audit,
                           fit_explained=explained(coef@rows[fit],targets[fit]),
                           audit_explained=explained(coef@rows[audit],targets[audit]),
                           carrier=metrics(before[audit],after,y)))
        fitted.append(coef)
    return result, fitted


def unpack(folder, step):
    path = folder / f'carrier_step{step:03d}.pt'
    value = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
    assert value['step'] == step
    return value


def audit_arm(folder, op, probes, short_only=False):
    stored = {s:unpack(folder,s) for s in sorted(AUDIT_STEPS)}
    frames = {s:x['frame'].cuda() for s,x in stored.items()}
    bends = {s:np.array(x['bend']) if x['bend'] is not None else None for s,x in stored.items()}
    general, reuse, matched = [], [], []
    selected = sorted(AUDIT_STEPS - set(AUDIT_ANCHORS)) if short_only else sorted(stored)
    for step in selected:
        current = stored[step]
        op.load(current)
        target = current['proposal'].cuda()
        innovation = current['innovation'].cuda()
        actual_carrier = target - innovation
        direction = unit(target)
        width = target.norm()
        earlier = [s for s in AUDIT_ANCHORS if s < step] if step in AUDIT_ANCHORS else [step-HORIZON]
        with probe_mode(op.model):
            before = [op.evaluate(x) for x,_ in probes]
            gradients = [unit(op.gradient(x,y)) for x,y in probes]
            responses = {s: [np.stack([response(op,x,v) for v in frames[s]]) for x,_ in probes]
                         for s in [step,*earlier]}
            targets = [response(op,x,direction) for x,_ in probes]
            full = [metrics(b,op.evaluate(x,target),y) for b,(x,y) in zip(before,probes)]
            now, _ = measure(op,probes,frames[step],bends[step],target,responses[step],targets,gradients,before)
            if step not in AUDIT_ANCHORS:
                frozen, _ = measure(op,probes,frames[step-HORIZON],bends[step-HORIZON],target,
                                    responses[step-HORIZON],targets,gradients,before)
                reuse.append(dict(source_step=step-HORIZON,target_step=step,full_proposal=full,
                                  frozen=frozen,current=now))
                print(f'{folder.name} future-step reuse at {step}',flush=True)
                if not short_only:
                    continue
            else:
                actual = []
                for g,(x,y) in enumerate(probes):
                    current_response = response(op,x,actual_carrier/width)
                    actual.append(dict(group=g, explained=explained(current_response,targets[g]),
                                       carrier=metrics(before[g],op.evaluate(x,actual_carrier),y),
                                       full_proposal=full[g]))
                general.append(dict(step=step, recorded_carrier=actual, refitted=now))
            for old in earlier:
                # Match the old recorded innovation against the current proposal
                # component that the old frozen decoder still cannot explain.
                stale, coefs = measure(op,probes,frames[old],bends[old],target,responses[old],targets,gradients,before)
                prior = unit(stored[old]['innovation'].cuda()) * width
                prior_response = [response(op,x,prior/width) for x,_ in probes]
                frozen_prior, _ = measure(op,probes,frames[old],bends[old],prior,
                                          responses[old],prior_response,gradients,before)
                adapted_prior, _ = measure(op,probes,frames[step],bends[step],prior,
                                           responses[step],prior_response,gradients,before)
                pairs = []
                for fit in (0,1):
                    audit = 1-fit
                    x,y = probes[audit]
                    similarity = [cosine(prior_response[g],targets[g]-coefs[fit]@responses[old][g]) for g in (fit,audit)]
                    past = metrics(before[audit],op.evaluate(x,prior),y)
                    pairs.append(dict(fit_group=fit,audit_group=audit,match_fit_cosine=similarity[0],
                                      match_audit_cosine=similarity[1],both_match=min(similarity)>=MATCH_COSINE,
                                      prior_innovation_gain=past, frozen_on_prior=frozen_prior[fit],
                                      current_on_prior=adapted_prior[fit], frozen_on_current=stale[fit],
                                      current_on_current=now[fit], full_current=full[audit]))
                matched.append(dict(earlier_step=old,later_step=step,groups=pairs))
        print(f'{folder.name} cross-image and absorption at {step}',flush=True)
    return dict(generalization=general,time_reuse=reuse,residual_absorption=matched)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--short-absorption', action='store_true', help='Also measure absorption over each four-step pair')
    args = parser.parse_args()
    verification = read(args.replay / 'complete.json')
    assert verification['recorded_steps'] == sorted(AUDIT_STEPS)
    checks = ('exact_visual_replay','exact_frame_replay','exact_backlog_replay','exact_bend_replay')
    assert len(verification['arms']) == 2 and all(all(row[key] for key in checks) for row in verification['arms'])
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder, model, _, flat = build_model(existing,existing.ClassificationHead.load(str(HEADS/'DTD.pt')).cuda())
    del flat
    visual = encoder.model.visual
    model.requires_grad_(False).eval()
    visual.requires_grad_(True)
    op = Operator(model,visual)
    old_groups = read(args.replay/ARMS[0]/'complete.json')['provenance']['probe_groups']
    probes, ids = independent_probes(encoder,old_groups)
    args.output.mkdir(parents=True,exist_ok=True)
    data = dict(match_cosine=MATCH_COSINE,reuse_horizon=HORIZON,probe_indices=ids,
                training_probe_indices=old_groups,verification=verification,arms={})
    for arm in ARMS:
        tag = '_short' if args.short_absorption else ''
        path = args.output / f'{arm}{tag}.json'
        if not path.exists():
            result = audit_arm(args.replay/arm,op,probes,args.short_absorption)
            existing.write(path,result)
        data['arms'][arm] = read(path)
        gc.collect()
        torch.cuda.empty_cache()
    output_file = 'short_absorption.json' if args.short_absorption else 'results.json'
    existing.write(args.output/output_file,data)


if __name__ == '__main__':
    main()
