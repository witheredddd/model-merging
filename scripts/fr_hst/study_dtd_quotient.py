"""Discover and test shared DTD update directions in the V/O quotient."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from collect_dtd_updates import batches, load_experiment, save
from train_atlas import embedding
from vo_quotient import VOQuotient


def predictor(model, found):
    names = ['image_encoder.' + name for name, _ in found]
    sizes = [p.numel() for _, p in found]
    shapes = [p.shape for _, p in found]

    def predict(theta, images):
        params = {n: v.view(s) for n, v, s in zip(names, theta.split(sizes), shapes)}
        return torch.func.functional_call(model, params, (images,)).float()

    return predict


@torch.no_grad()
def response(predict, center, directions, images):
    return torch.stack([torch.func.jvp(lambda theta: embedding(predict(theta, images)), (center,), (d,))[1]
                        for d in directions], dim=1)


def canonical(fit_a, fit_b, calibration_a, calibration_b):
    def whiten(x):
        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        keep = s > .02 * s[0]
        return u[:, keep], vh[keep].T / s[keep]

    ua, wa = whiten(fit_a)
    ub, wb = whiten(fit_b)
    left, similarity, right = torch.linalg.svd(ua.T @ ub, full_matrices=False)
    a, b = wa @ left, wb @ right.T
    repeat = F.cosine_similarity(calibration_a @ a, calibration_b @ b, dim=0)
    order = torch.minimum(similarity, repeat).argsort(descending=True)
    return a[:, order], b[:, order], similarity[order], repeat[order]


def gram(rows):
    result = torch.zeros((len(rows), len(rows)), device=rows.device, dtype=torch.double)
    for chunk in rows.split(1048576, dim=1):
        x = chunk.double()
        result += x @ x.T
    return result


def frame(rows, rank):
    values, modes = torch.linalg.eigh(gram(rows))
    if values[-rank] <= values[-1] * 1e-8:
        raise RuntimeError('Requested decoder rank exceeds independent directions')
    return ((modes[:, -rank:] / values[-rank:].sqrt()).T.float() @ rows).contiguous()


def load_bank(path, epoch, base_hash, names):
    manifest = json.loads((path / 'complete.json').read_text())
    if manifest['base_sha256'] != base_hash or manifest['names'] != names or manifest['reserved'] != 144:
        raise RuntimeError(f'Trajectory provenance mismatch: {path.name}')
    data = []
    for file in sorted(path.glob('update_*.pt')):
        state = torch.load(file, weights_only=True, map_location='cpu', mmap=True)
        if state['epoch'] == epoch:
            data.append(state['delta'].cuda())
    if len(data) < 3:
        raise RuntimeError(f'Insufficient update windows: {path.name}')
    return data


@torch.no_grad()
def discover(predict, found, model, center, banks, pools, record, epoch):
    fit, calibration, audit = pools
    quotient = VOQuotient(model, found, center)
    control = quotient.verify_group(model, found, center, audit, every_head=epoch == 1)
    control['projection'] = quotient.check_projection(banks['real42'][0])
    if (control['joint_logit_error'] > 2e-4 or (control['max_head_logit_error'] or 0) > 2e-4
            or control['broken_logit_error'] < 1e-4 or control['minimum_output_gram_eigenvalue'] <= 0
            or control['projection']['idempotence'] > 2e-5):
        raise RuntimeError(f'V/O gauge verification failed: {control}')
    record('gauge', region=epoch, **control)
    evidence = {}
    spaces = {}
    for name, sources, project in [('quotient', ('real42', 'real43'), True),
                                   ('raw', ('real42', 'real43'), False),
                                   ('shuffled', ('sham42', 'sham43'), True)]:
        banks_unit = []
        for source in sources:
            rows = [quotient.horizontal(d) if project else d for d in banks[source]]
            banks_unit.append(torch.stack([d / d.norm() for d in rows]))
        a, b = banks_unit
        fa, fb = (response(predict, center, x, fit) for x in (a, b))
        ca, cb = (response(predict, center, x, calibration) for x in (a, b))
        wa, wb, fit_score, cal_score = canonical(fa, fb, ca, cb)
        accepted = int(((fit_score >= .70) & (cal_score >= .70)).sum())
        # Selection uses only discovery seeds and calibration images.
        record('consensus', region=epoch, method=name, fit=fit_score.tolist(),
               calibration=cal_score.tolist(), qualifying=accepted)
        spaces[name] = .5 * (wa.T.float() @ a + wb.T.float() @ b)
        evidence[name] = accepted
        if name == 'quotient':
            spaces['pca'] = torch.cat([a, b])
            gauge_error = (response(predict, center, [quotient.horizontal(banks['real42'][0])], fit)
                           - response(predict, center, [banks['real42'][0]], fit)).norm()
            scale = response(predict, center, [banks['real42'][0]], fit).norm()
            record('horizontal_response', region=epoch, relative_error=float(gauge_error / scale.clamp_min(1e-10)),
                   removed_energy=[float((d - quotient.horizontal(d)).norm().square() / d.norm().square())
                                   for source in sources for d in banks[source]])
    rank = min(max(evidence['quotient'], 1), 6, *(len(v) for v in spaces.values()))
    decoders = {name: frame(v if name == 'pca' else v[:rank], rank) for name, v in spaces.items()}
    holdout = torch.stack([d / d.norm() for d in banks['real44']])
    target_fit = response(predict, center, holdout, fit)
    target_audit = response(predict, center, holdout, audit)
    for name, basis in decoders.items():
        prediction_fit = response(predict, center, basis, fit)
        prediction_audit = response(predict, center, basis, audit)
        coefficient = torch.linalg.pinv(prediction_fit, rtol=1e-5) @ target_fit
        residual = target_audit - prediction_audit @ coefficient
        energy = 1 - residual.square().sum() / target_audit.square().sum()
        row = dict(heldout_functional_energy=float(energy),
                   per_window_energy=(1 - residual.square().sum(0) / target_audit.square().sum(0)).tolist(),
                   heldout_response_cosine=F.cosine_similarity(target_audit, prediction_audit @ coefficient, dim=0).tolist())
        record('heldout', region=epoch, method=name, rank=rank,
               quotient_rank_qualified=evidence['quotient'], **row)
    # Replicate the leading discovered mode using only the independent trajectory and fit images.
    direction = spaces['quotient'][0]
    observed = response(predict, center, [direction], fit)[:, 0]
    fitted = torch.linalg.pinv(target_fit, rtol=1e-5) @ observed
    independent = quotient.horizontal(fitted.float() @ holdout)
    coupling(predict, found, center, direction, independent, calibration, audit, record, epoch)
    return decoders, spaces['quotient'][:min(3, len(spaces['quotient']))]


@torch.no_grad()
def coupling(predict, found, center, direction, independent, calibration, audit, record, epoch):
    def split(vector):
        blocks = []
        offset = 0
        for name, p in found:
            depth = int(name.split('resblocks.')[1].split('.')[0])
            if depth == len(blocks):
                blocks.append(torch.zeros_like(vector))
            blocks[depth][offset:offset + p.numel()] = vector[offset:offset + p.numel()]
            offset += p.numel()
        return blocks

    indices = torch.triu_indices(12, 12, 1, device=center.device)
    data = []
    for vector in (direction, independent):
        blocks = split(vector)
        for x in (calibration, audit):
            z = F.normalize(response(predict, center, blocks, x), dim=0)
            data.append((z.T @ z)[indices.unbind()])
    evidence = torch.stack(data)
    replicated = (evidence.abs().min(0).values >= .10) & (evidence.sign() == evidence[:1].sign()).all(0)
    edges = [dict(layers=[int(indices[0, i]), int(indices[1, i])], cosine=evidence[:, i].tolist())
             for i in replicated.nonzero().flatten()]
    record('coupling', region=epoch, edges=edges, comparisons=len(replicated),
           cross_seed_audit_cosine=float(F.cosine_similarity(data[1], data[3], dim=0)))


@torch.no_grad()
def overlap(predict, first, second, directions, next_directions, fit, audit, record):
    rank = min(len(directions), len(next_directions), 3)
    if rank < 2:
        record('overlap', tested_rank=rank, reason='Fewer than two independent candidate coordinates')
        return
    a, b = frame(directions[:rank], rank), frame(next_directions[:rank], rank)
    left = first.new_zeros(rank)
    target = embedding(predict(second, fit))
    for _ in range(6):
        center = first + left @ a
        jac = response(predict, center, a, fit)
        step = torch.linalg.pinv(jac, rtol=1e-4) @ (target - embedding(predict(center, fit)))
        left.add_(step.float() * min(1., .5 / max(float(step.norm()), 1e-10)))
    center = first + left @ a
    ja, jb = response(predict, center, a, fit), response(predict, second, b, fit)
    forward = torch.linalg.pinv(jb, rtol=1e-4) @ ja
    reverse = torch.linalg.pinv(ja, rtol=1e-4) @ jb
    singular = torch.linalg.svdvals(forward)
    audit_center_a = embedding(predict(center, audit))
    audit_center_b = embedding(predict(second, audit))
    anchor_error = float((audit_center_a - audit_center_b).norm())
    errors, roundtrip = [], []
    rng = torch.Generator(device=first.device).manual_seed(31)
    for vector in torch.randn(12, rank, device=first.device, generator=rng):
        d = .12 * vector / vector.norm()
        e = (forward @ d.double()).float()
        p = embedding(predict(center + d @ a, audit))
        q = embedding(predict(second + e @ b, audit))
        errors.append(float((p - q).norm() / (p - audit_center_a).norm().clamp_min(1e-10)))
        roundtrip.append(float((reverse @ e.double() - d).norm() / d.norm()))
    record('overlap', tested_rank=rank, fit_transition_rank=int((singular > .01 * singular[0]).sum()),
           singular_values=singular.tolist(), independent_center_distance=anchor_error,
           audit_relative_errors=errors, coordinate_roundtrip_relative_errors=roundtrip,
           accepted=bool(anchor_error < .02 and max(errors) < .20 and max(roundtrip) < .20))


def replay(predict, base, decoders, dataset, train_ids, validation, seed, epochs, record):
    for name, basis in decoders.items():
        code = base.new_zeros(len(basis)).requires_grad_()
        opt = torch.optim.Adam([code], lr=.035)
        for epoch in range(1, epochs + 1):
            for x, y, _ in batches(dataset, train_ids, seed, epoch):
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                opt.zero_grad(set_to_none=True)
                for start in range(0, len(y), 16):
                    theta = base + code @ basis
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = F.cross_entropy(predict(theta, x[start:start + 16]),
                                               y[start:start + 16], reduction='sum') / len(y)
                    loss.backward()
                opt.step()
            with torch.no_grad():
                theta = base + code @ basis
                total, correct, count = 0., 0, 0
                for x, y in validation:
                    z = predict(theta, x.cuda())
                    y = y.cuda()
                    total += float(F.cross_entropy(z, y, reduction='sum'))
                    correct += int((z.argmax(-1) == y).sum())
                    count += len(y)
                record('replay', method=name, seed=seed, epoch=epoch, rank=len(code),
                       ce=total / count, accuracy=correct / count, code=code.tolist())
        yield name, code.detach().cpu()


def run(args):
    existing, model, found, base, dataset, validation, splits, ids, base_hash = load_experiment(args.root, 44)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=False)
    names = [n for n, _ in found]
    predict = predictor(model, found)

    def record(event, **values):
        row = dict(event=event, **values)
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print({k: v for k, v in row.items() if k not in ('head_errors', 'edges')}, flush=True)

    pools = [torch.stack([splits.train_dataset[i][0] for i in ids[start:start + n]]).cuda()
             for start, n in ((0, 16), (48, 16), (96, 24))]
    record('config', base_sha256=base_hash, probe_counts=list(map(len, pools)),
           source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           collector_sha256=hashlib.sha256(Path(__file__).with_name('collect_dtd_updates.py').read_bytes()).hexdigest(),
           quotient_sha256=hashlib.sha256(Path(__file__).with_name('vo_quotient.py').read_bytes()).hexdigest())
    decoders, regions = None, []
    for epoch in (1, 2):
        center = base if epoch == 1 else torch.load(args.trajectories / 'real42/anchor_epoch1.pt',
                                                  weights_only=True, map_location='cpu', mmap=True).cuda()
        bank = {s: load_bank(args.trajectories / s, epoch, base_hash, names)
                for s in ('real42', 'real43', 'real44', 'sham42', 'sham43')}
        candidates, region = discover(predict, found, model, center, bank, pools, record, epoch)
        regions.append(region)
        if epoch == 1:
            decoders = candidates
        del bank, candidates
    overlap(predict, base, center, *regions, pools[0], pools[2], record)
    codes = dict(replay(predict, base, decoders, dataset, ids[144:], validation, 45, args.epochs, record))
    save(args.output / 'decoder.pt', dict(base_sha256=base_hash, names=names,
         modes=decoders['quotient'].cpu(), code=codes['quotient']))
    record('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('At least one replay epoch is required')
    run(args)
