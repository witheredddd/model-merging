"""Fit a task-supervised horizontal continuation at the current frozen DTD model."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from collect_dtd_updates import load_experiment, save
from study_dtd_quotient import frame, load_bank, predictor, replay, response
from vo_quotient import VOQuotient


def target(logits, labels):
    sqrt_p = logits.double().log_softmax(-1).mul(.5).exp()
    one_hot = F.one_hot(labels, logits.shape[-1]).double()
    radial = (sqrt_p * one_hot).sum(dim=-1, keepdim=True)
    # Project the CE descent onto the sphere tangent and cap confidence weighting.
    return ((one_hot - radial * sqrt_p) / radial.clamp_min(.1) / math.sqrt(len(logits))).flatten()


def fit_direction(jacobian, objective, fisher):
    rhs = jacobian.T @ objective
    if fisher:
        gram = jacobian.T @ jacobian
        damping = .10 * gram.diag().mean().clamp_min(1e-12)
        rhs = torch.linalg.solve(gram + damping * torch.eye(len(rhs), device=rhs.device), rhs)
    return F.normalize(rhs, dim=0)


def run(args):
    _, model, found, base, dataset, validation, splits, ids, base_hash = load_experiment(args.root, 45)
    names = [n for n, _ in found]
    predict = predictor(model, found)
    frozen = torch.load(args.study / 'decoder.pt', weights_only=True, map_location='cuda')
    if frozen['base_sha256'] != base_hash or frozen['names'] != names:
        raise RuntimeError('Decoder provenance mismatch')
    center = base + frozen['code'] @ frozen['modes']
    quotient = VOQuotient(model, found, center)
    args.output.mkdir(parents=True, exist_ok=False)

    def record(event, **values):
        row = dict(event=event, **values)
        with (args.output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(row, flush=True)

    record('config', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           base_sha256=base_hash, start='Final primary quotient replay', fit=48, calibration=48, audit=48)
    bank = [quotient.horizontal(d) for s in ('real42', 'real43')
            for d in load_bank(args.trajectories / s, 2, base_hash, names)]
    basis = frame(torch.stack([d / d.norm() for d in bank]), len(bank))
    del bank
    pools = []
    for offset in (0, 48, 96):
        samples = [splits.train_dataset[i] for i in ids[offset:offset + 48]]
        x = torch.stack([a for a, _ in samples]).cuda()
        y = torch.tensor([b for _, b in samples], device='cuda')
        with torch.no_grad():
            z = predict(center, x)
        # Limit CLIP forward AD memory without changing the per-example weighting.
        jac = torch.cat([response(predict, center, basis, batch) * math.sqrt(len(batch) / len(x))
                         for batch in x.split(16)])
        pools.append((x, y, z, jac))

    directions = {}
    rng = torch.Generator(device=center.device).manual_seed(814)
    for name, shuffled, fisher in [('task_fisher', False, True), ('shuffled_fisher', True, True),
                                    ('task_euclidean', False, False)]:
        controls = [y[torch.randperm(len(y), device=y.device, generator=rng)] if shuffled else y
                    for _, y, _, _ in pools[:2]]
        a, b = [fit_direction(jac, target(z, labels), fisher) for (_, _, z, jac), labels in zip(pools, controls)]
        combined = F.normalize(a + b, dim=0)
        direction = combined.float() @ basis
        drops = []
        with torch.no_grad():
            for x, y, z, _ in pools:
                delta = F.cross_entropy(z, y, reduction='none') - F.cross_entropy(
                    predict(center + .20 * direction, x), y, reduction='none')
                drops.append(dict(ce_gain=float(delta.mean()), standard_error=float(delta.std() / len(delta) ** .5)))
        replicated = [float(F.cosine_similarity(jac @ a, jac @ b, dim=0)) for _, _, _, jac in pools]
        passed = min(replicated[:2]) >= .35 and min(x['ce_gain'] for x in drops[:2]) > .001
        record('task_direction', method=name, functional_cosines=replicated, finite_gain=drops,
               supported_by_fit_calibration=passed)
        directions[name] = direction[None, :]
    codes = dict(replay(predict, center, directions, dataset, ids[144:], validation, 46, args.epochs, record))
    save(args.output / 'continuation.pt', dict(base_sha256=base_hash, names=names,
         modes=torch.cat([frozen['modes'], directions['task_fisher']]).cpu(),
         code=torch.cat([frozen['code'].cpu(), codes['task_fisher']])))
    record('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('At least one continuation epoch is required')
    run(args)
