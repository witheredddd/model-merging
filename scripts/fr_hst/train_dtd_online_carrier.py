"""Learn a full-visual functional carrier and feed deferred updates back into it."""

import argparse
import fcntl
import gc
import math
from statistics import median
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from run_dtd_carrier_diagnostics import Operator
from train_dtd_rolling_innovation import (
    EPOCHS, INTERVAL, LEARNING_RATE, ShadowAdamW, loss_at, probe_mode, probe_sets, test_all,
)
from train_eight_expert_queue import BASE, BATCH_SIZE, HEADS, ROOT, TASKS, build_model, cosine_lr, existing, loader, read, save, splits


ARMS = ('online_linear', 'online_nonlinear')
SEED = 42
WARMUP_STEPS = 75
TASK_WEIGHT = .05
RIDGE = .01
FRAME_RATE = .04
BEND_RATE = .035
DELAY_MIN = .02
DELAY_MAX = .125
AUDIT_ANCHORS = (76, 100, 125, 150, 175, 200, 216)
AUDIT_STEPS = frozenset(AUDIT_ANCHORS) | {s + 4 for s in AUDIT_ANCHORS[:-1]}


def orthonormal(rows):
    values, vectors = torch.linalg.eigh((rows @ rows.T).double())
    if float(values.min()) <= 1e-10:
        raise ValueError('Online carrier has insufficient parameter rank')
    inverse_root = (vectors * values.rsqrt()) @ vectors.T
    return inverse_root.float() @ rows


def initialize(history):
    rows = torch.stack(history)
    rows = rows / rows.norm(dim=1, keepdim=True)
    _, vectors = torch.linalg.eigh((rows @ rows.T).double())
    return orthonormal(vectors[:, -3:].T.flip(0).float() @ rows)


def features(z):
    a, b = z
    return np.array([a * a, a * b, b * b])


def decoder(z, bend):
    if bend is None:
        return z, np.eye(3)
    a, b = z
    value = np.r_[z, bend @ features(z)]
    derivative = np.array([[1., 0.], [0., 1.],
                           [2 * bend[0] * a + bend[1] * b, bend[1] * a + 2 * bend[2] * b]])
    return value, derivative


def coordinates(responses, target, signed_gradient, bend):
    """Bounded local Gauss–Newton in two or three latent coordinates."""
    dim = 3 if bend is None else 2
    denominator = max(float(target @ target), 1e-12)
    gram = responses @ responses.T / denominator
    linear = -responses @ target / denominator + TASK_WEIGHT * signed_gradient
    z = np.zeros(dim)
    eye = np.eye(dim)

    def objective(point):
        value, _ = decoder(point, bend)
        return .5 * value @ gram @ value + linear @ value + .5 * RIDGE * (point @ point)

    for _ in range(6):
        value, jac = decoder(z, bend)
        gradient = jac.T @ (gram @ value + linear) + RIDGE * z
        direction = np.linalg.solve(jac.T @ gram @ jac + RIDGE * eye, gradient)
        before = objective(z)
        for rate in (1., .5, .25, .125):
            proposal = z - rate * direction
            proposal /= max(1., np.linalg.norm(proposal) / 1.5)
            if objective(proposal) < before:
                z = proposal
                break
    value, _ = decoder(z, bend)
    error = (value @ responses - target) / denominator
    bend_next = None
    if bend is not None:
        gradient = (gram @ value + linear)[-1] * features(z)
        bend_next = np.clip(bend - BEND_RATE * gradient / max(1., np.linalg.norm(gradient)), -.5, .5)
    return z, value, error, bend_next


def response(op, x, direction):
    logits, jv = op.logit_jvp(x, direction)
    p = logits.double().softmax(-1)
    return (p.sqrt() * (jv - (p * jv).sum(-1, keepdim=True))).flatten().cpu().numpy() / math.sqrt(len(x))


def functional_pullback(op, x, error):
    p = op.model(x).double().log_softmax(-1)
    embedding = 2 * p.mul(.5).exp() / math.sqrt(len(x))
    residual = torch.as_tensor(error, dtype=embedding.dtype, device=embedding.device).view_as(embedding)
    grads = torch.autograd.grad(embedding, op.params, residual)
    return torch.cat([g.detach().flatten() for g in grads])


def learn(op, x, frame, bend, proposal, batch_gradient, backlog, boundary):
    scale = proposal.norm()
    normalized = proposal / scale
    descent = batch_gradient / batch_gradient.norm().clamp_min(1e-12)
    with probe_mode(op.model):
        responses = np.stack([response(op, x, direction) for direction in frame])
        target = response(op, x, normalized)
        signed = (frame @ descent).double().cpu().numpy()
        z, coeff, error, next_bend = coordinates(responses, target, signed, bend)
        pullback = functional_pullback(op, x, error) + TASK_WEIGHT * descent
        replay_delta = None
        if boundary and bool(backlog.norm() > 0.):
            without_backlog = target - response(op, x, backlog / scale)
            _, no_feedback, _, _ = coordinates(responses, without_backlog, signed, bend)
            replay_delta = float(np.linalg.norm(coeff - no_feedback))
    value = torch.as_tensor(coeff, dtype=frame.dtype, device=frame.device)
    carrier = (value @ frame) * scale
    innovation = proposal - carrier
    # Functional-gradient update in the full visual tangent, including components
    # orthogonal to all current axes, followed by an orthonormal retraction.
    normal = pullback - (frame @ pullback) @ frame
    direction = normal / pullback.norm().clamp_min(1e-12)
    next_frame = orthonormal(frame - FRAME_RATE * (value / value.norm().clamp_min(1e-12))[:, None] * direction)
    capture = 1. - float(np.linalg.norm(value.double().cpu().numpy() @ responses - target) ** 2
                         / max(np.linalg.norm(target) ** 2, 1e-12))
    c_gain = (-descent @ carrier).clamp_min(0)
    r_gain = (-descent @ innovation).clamp_min(0)
    delay = DELAY_MIN + (DELAY_MAX - DELAY_MIN) * float(c_gain / (c_gain + r_gain + 1e-12))
    diagnostics = dict(latent=z.tolist(), decoder_coefficients=coeff.tolist(), function_explained=capture,
                       nonlinear_relative_norm=float(abs(coeff[-1])) if bend is not None else 0.,
                       bend_coefficients=next_bend.tolist() if next_bend is not None else None,
                       frame_overlap=float((frame @ next_frame.T).square().sum() / len(frame)),
                       frame_gradient_outside_fraction=float(direction.norm()),
                       carrier_reassessment_delta=replay_delta, delay=delay)
    return innovation, next_frame, next_bend, diagnostics


@torch.no_grad()
def commit(op, shadow, decay_point, raw, backlog, innovation, delay):
    proposed = raw + backlog
    adjustment = backlog - delay * innovation
    selected = [s + d for s, d in zip(shadow.shadows, op.parts(adjustment))]
    executed = torch.cat([(w - origin).flatten() for w, origin in zip(selected, decay_point)])
    next_backlog = proposed - executed
    relative_closure = float((executed + next_backlog - proposed).norm() / proposed.norm())
    relative_intervention = float((executed - raw).norm() / raw.norm())
    if not math.isfinite(relative_closure) or relative_closure > 2e-6:
        raise ValueError('Executed update and retained innovation do not conserve the proposal')
    for param, weight in zip(op.params, selected):
        param.copy_(weight)
    return next_backlog, dict(conservation_error=relative_closure, relative_intervention=relative_intervention,
                             reinjected_norm=float(backlog.norm()), retained_norm=float(next_backlog.norm()),
                             raw_norm=float(raw.norm()))


def train(arm, output, trace, base_hash, record_carrier=False):
    output.mkdir(parents=True, exist_ok=True)
    complete = output / 'complete.json'
    final_path = output / 'epoch08.pt'
    if complete.exists():
        value = read(complete)
        assert existing.sha(final_path) == value['checkpoint_sha256']
        return
    existing.seed_all(SEED)
    head = existing.ClassificationHead.load(str(HEADS / 'DTD.pt')).cuda()
    encoder, model, _, flat = build_model(existing, head)
    del flat
    visual = encoder.model.visual
    model.requires_grad_(False)
    visual.requires_grad_(True)
    op = Operator(model, visual)
    shadow = ShadowAdamW(op)
    train_set, val_set = splits('DTD', encoder.train_preprocess, encoder.val_preprocess)
    data = existing.EpochData(train_set, SEED)
    validation = loader(val_set, 8, BATCH_SIZE, SEED)
    steps = math.ceil(len(data) / BATCH_SIZE)
    assert steps == 27
    schedule = cosine_lr(shadow.optimizer, LEARNING_RATE, 500, TASKS['DTD'][0] * steps)
    probe_x, probe_y, fit_x, groups = probe_sets(encoder, trace)
    provenance = dict(seed=SEED, arm=arm, base_sha256=base_hash, head_sha256=existing.sha(HEADS / 'DTD.pt'),
                      probe_groups=groups, trace_sha256=existing.sha(trace / 'trace.json'),
                      epochs=EPOCHS, steps_per_epoch=steps, warmup_steps=WARMUP_STEPS, rank=3,
                      task_weight=TASK_WEIGHT, ridge=RIDGE, frame_rate=FRAME_RATE, bend_rate=BEND_RATE,
                      delay_range=[DELAY_MIN, DELAY_MAX], scheduler_steps=TASKS['DTD'][0] * steps)
    if record_carrier:
        provenance['recorded_steps'] = sorted(AUDIT_STEPS)
    resume = output / 'resume.pt'
    records, epochs, history = [], [], []
    frame = None
    bend = np.array([.08, 0., .08]) if arm == 'online_nonlinear' else None
    backlog = torch.zeros(sum(op.sizes), device='cuda')
    window = torch.zeros_like(backlog)
    if resume.exists():
        state = torch.load(resume, map_location='cpu', mmap=True, weights_only=True)
        if state['provenance'] != provenance:
            raise ValueError('Online carrier resume provenance differs')
        visual.load_state_dict(state['visual'])
        shadow.optimizer.load_state_dict(state['optimizer'])
        records, epochs = state['records'], state['epochs']
        history = [v.cuda() for v in state['history']]
        frame = state['frame'].cuda() if state['frame'] is not None else None
        bend = np.array(state['bend']) if state['bend'] is not None else None
        backlog = state['backlog'].cuda()
        del state
    for epoch in range(len(epochs), EPOCHS):
        started = time.monotonic()
        data.epoch = epoch
        model.train()
        for i, (images, labels, _) in enumerate(loader(data, 8, BATCH_SIZE, SEED + epoch, True)):
            step = epoch * steps + i + 1
            boundary = step % INTERVAL == 0 or i + 1 == steps
            schedule(step - 1)
            shadow.clear()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images.cuda(non_blocking=True))
                loss = F.cross_entropy(logits.float(), labels.cuda(non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(op.params, 1., error_if_nonfinite=True)
            decay, raw = shadow.propose()
            row = dict(step=step, epoch=epoch + 1, training_ce=float(loss.detach()), boundary=boundary)
            if step <= WARMUP_STEPS:
                shadow.apply(shadow.shadows, decay, window)
                if boundary:
                    history.append(window)
                    window = torch.zeros_like(backlog)
            else:
                if frame is None:
                    if len(history) != 5:
                        raise ValueError('Warmup did not record five complete windows')
                    frame = initialize(history)
                    history.clear()
                gradient = torch.cat([p.grad.detach().flatten() for p in op.params])
                proposal = raw + backlog
                innovation, next_frame, next_bend, info = learn(op, fit_x, frame, bend, proposal, gradient, backlog, boundary)
                drain = step == EPOCHS * steps
                delay = 0. if drain else info['delay']
                if record_carrier and step in AUDIT_STEPS:
                    save(output / f'carrier_step{step:03d}.pt', dict(
                        step=step, visual=visual.state_dict(), frame=frame, bend=bend.tolist() if bend is not None else None,
                        proposal=proposal, innovation=innovation, delay=delay))
                if boundary:
                    with probe_mode(model):
                        selected = [s + d for s, d in zip(shadow.shadows, op.parts(backlog - delay * innovation))]
                        control = loss_at(op, shadow.shadows, probe_x, probe_y)
                        trial = loss_at(op, selected, probe_x, probe_y)
                    info['probe_gain_vs_raw'] = (control - trial).reshape(2, 32).mean(1).cpu().tolist()
                backlog, balance = commit(op, shadow, decay, raw, backlog, innovation, delay)
                frame, bend = next_frame, next_bend
                row.update(**info, **balance, terminal_drain=drain, applied_delay=delay)
            records.append(row)
            if boundary:
                print(f'{arm} step {step}/{EPOCHS * steps}, retained={float(backlog.norm()):.5g}', flush=True)
        value = existing.evaluate(model, validation)
        epochs.append(dict(epoch=epoch + 1, seconds=time.monotonic() - started, **value))
        save(resume, dict(provenance=provenance, visual=visual.state_dict(), optimizer=shadow.optimizer.state_dict(),
                          history=history, frame=frame, bend=bend.tolist() if bend is not None else None,
                          backlog=backlog, epochs=epochs, records=records))
        existing.write(output / 'progress.json', dict(epochs=epochs, last=records[-1]))
        print(f'{arm} epoch {epoch + 1}: Val Acc={value["accuracy"]:.6f}, CE={value["ce"]:.6f}', flush=True)
    save(final_path, dict(provenance=provenance, visual=visual.state_dict(), frame=frame,
                          bend=bend.tolist() if bend is not None else None, backlog=backlog, validation=epochs[-1]))
    existing.write(complete, dict(provenance=provenance, checkpoint_sha256=existing.sha(final_path),
                                  epochs=epochs, records=records))
    resume.unlink()


def verify_training(arms, baseline):
    expected = [row['training_ce'] for row in baseline['records'][:WARMUP_STEPS]]
    results = []
    for folder in arms:
        value = read(folder / 'complete.json')
        rows = value['records']
        if [r['training_ce'] for r in rows[:WARMUP_STEPS]] != expected:
            raise ValueError('Warmup differs from the exact AdamW reference')
        active = rows[WARMUP_STEPS:]
        nonlinear = value['provenance']['arm'] == 'online_nonlinear'
        assert len(rows) == 216 and len(active) == 141 and active[-1]['terminal_drain']
        assert all(r['applied_delay'] > 0. for r in active[:-1])
        assert all(r['conservation_error'] <= 2e-6 for r in active)
        assert all(math.isfinite(r['training_ce']) for r in rows)
        assert any(r['relative_intervention'] > 1e-6 for r in active)
        assert any(r['reinjected_norm'] > 0. for r in active[1:])
        assert any(r['nonlinear_relative_norm'] > 0. for r in active) == nonlinear
        results.append(dict(arm=value['provenance']['arm'], final_validation=value['epochs'][-1],
                            intervened_steps=sum(r['relative_intervention'] > 1e-6 for r in active),
                            max_conservation_error=max(r['conservation_error'] for r in active),
                            median_function_explained=median(r['function_explained'] for r in active),
                            median_frame_overlap=median(r['frame_overlap'] for r in active),
                            median_applied_delay=median(r['applied_delay'] for r in active),
                            median_nonlinear_relative_norm=median(r['nonlinear_relative_norm'] for r in active),
                            median_reinjected_norm=median(r['reinjected_norm'] for r in active),
                            median_carrier_reassessment_delta=median(r['carrier_reassessment_delta'] for r in active
                                                                      if r['carrier_reassessment_delta'] is not None),
                            terminal_retained_norm=active[-1]['retained_norm'],
                            checkpoint_sha256=value['checkpoint_sha256']))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='Completed seed-42 AdamW directory from 497767c')
    parser.add_argument('--record-carrier', action='store_true', help='Record selected pre-update states for read-only diagnostics')
    parser.add_argument('--replay-reference', type=Path, help='Verify replay against the existing online run and skip test rescoring')
    args = parser.parse_args()
    if args.replay_reference is not None and not args.record_carrier:
        parser.error('--replay-reference requires --record-carrier')
    args.output.mkdir(parents=True, exist_ok=True)
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with (args.output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base_hash = existing.sha(BASE)
        baseline = read(args.baseline / 'complete.json')
        if (baseline['provenance']['base_sha256'] != base_hash
                or baseline['provenance']['head_sha256'] != existing.sha(HEADS / 'DTD.pt')
                or not baseline['exact_historical_baseline']):
            raise ValueError('Baseline provenance differs')
        arms = [args.output / arm for arm in ARMS]
        for name, path in zip(ARMS, arms):
            train(name, path, args.trace, base_hash, args.record_carrier)
            gc.collect()
            torch.cuda.empty_cache()
        overview = verify_training(arms, baseline)
        existing.write(args.output / 'training_complete.json', dict(arms=list(ARMS)))
        if args.replay_reference is not None:
            for row, folder in zip(overview, arms):
                name = folder.name
                ref_folder = args.replay_reference / name
                ref_path = ref_folder / 'epoch08.pt'
                assert existing.sha(ref_path) == read(ref_folder / 'complete.json')['checkpoint_sha256']
                expected = torch.load(ref_path, map_location='cpu', mmap=True, weights_only=True)
                replay = torch.load(folder / 'epoch08.pt', map_location='cpu', mmap=True, weights_only=True)
                row['exact_visual_replay'] = all(torch.equal(v, replay['visual'][k]) for k,v in expected['visual'].items())
                row['exact_frame_replay'] = torch.equal(expected['frame'], replay['frame'])
                row['exact_backlog_replay'] = torch.equal(expected['backlog'], replay['backlog'])
                row['exact_bend_replay'] = expected['bend'] == replay['bend']
                assert all(row[k] for k in ('exact_visual_replay', 'exact_frame_replay', 'exact_backlog_replay', 'exact_bend_replay'))
                print(f'{name}: replay endpoint exactly matches original', flush=True)
            existing.write(args.output / 'complete.json', dict(arms=overview, recorded_steps=sorted(AUDIT_STEPS)))
            return
        expected = read(ROOT / 'datasets/DTD_test_provenance.json')
        for row in expected['files']:
            assert existing.sha(ROOT / 'datasets/_downloads/tanganke__dtd' / row['path']) == row['sha256']
        test_all(args.output, arms, expected)
        for row, folder in zip(overview, arms):
            row['historical_test'] = read(folder / 'historical_test.json')
        existing.write(args.output / 'summary.json', dict(
            baseline=dict(final_validation=baseline['epochs'][-1],
                          historical_test=read(args.baseline / 'historical_test.json')),
            arms=overview))
        existing.write(args.output / 'complete.json', dict(arms=list(ARMS)))


if __name__ == '__main__':
    main()
