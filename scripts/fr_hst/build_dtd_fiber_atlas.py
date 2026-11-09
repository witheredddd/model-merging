"""Fit and validate local probability charts and restricted DTD fibers."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from fiber_atlas_experts import load_task, probe
from study_dtd_quotient import frame, response
from vo_quotient import VOQuotient


def distance(a, b):
    chord = (a - b).norm(dim=-1).div(2).clamp(max=1)
    return (4 * chord.asin()).square().mean(-1).sqrt()


def load_dtd_points(root, task, required):
    result = {0: task.base.cpu()}
    for seed in sorted({s for s, _ in required}):
        path = root / f'real{seed}'
        info = json.loads((path / 'complete.json').read_text())
        if (info['base_sha256'] != task.base_sha256 or info['shuffled']
                or info['names'] != [n for n, _ in task.found]):
            raise RuntimeError('DTD trajectory provenance mismatch')
        theta = task.base.clone()
        for file in sorted(path.glob('update_*.pt')):
            state = torch.load(file, map_location='cpu', weights_only=True, mmap=True)
            theta.add_(state['delta'].cuda())
            step = state['step']
            if step == info['epoch_1']['step']:
                anchor = torch.load(path / 'anchor_epoch1.pt', map_location='cpu', weights_only=True, mmap=True).cuda()
                if (theta - anchor).abs().max() > 1e-6:
                    raise RuntimeError('DTD trajectory does not reconstruct its anchor')
                theta = anchor
            if (seed, step) in required:
                result[seed, step] = theta.cpu()
    if any(k not in result for k in required):
        raise RuntimeError('Missing checkpoint requested by the atlas')
    return result


def chart(points, metric, index, neighbors):
    indices = metric[index].argsort()[:neighbors]
    a, b = points[index], points[indices]
    cosine = (a * b).sum(-1, keepdim=True).clamp(0, 1)
    tangent = b - cosine * a
    size = tangent.norm(dim=-1, keepdim=True)
    tangent = (2 * torch.atan2(size, cosine) * tangent / size.clamp_min(1e-20)).flatten(1)
    tangent /= points.shape[1] ** .5
    u, sigma, _ = torch.linalg.svd(tangent, full_matrices=False)
    cumulative = sigma.square().cumsum(0) / sigma.square().sum().clamp_min(1e-20)
    reached = (cumulative >= .90).nonzero()
    rank = int(reached[0]) + 1 if len(reached) else 0
    return indices, u[:, :rank] * sigma[:rank], sigma


def normalized_stress(source, target):
    return float((source - target).norm() / target.norm().clamp_min(1e-20))


def transition(a, b):
    ia, x, _ = a
    ib, y, _ = b
    shared = sorted(set(ia.tolist()) & set(ib.tolist()))
    rank = x.shape[1]
    if rank == 0 or rank != y.shape[1] or len(shared) < rank + 2:
        return dict(shared=len(shared), eligible=False, accepted=False)
    x = x[[ia.tolist().index(k) for k in shared]]
    y = y[[ib.tolist().index(k) for k in shared]]

    def fit(source, target):
        source = F.pad(source, (0, 1), value=1.)
        coefficients = torch.linalg.lstsq(source, target, rcond=1e-7).solution
        expected = []
        for i in range(len(source)):
            use = torch.arange(len(source)) != i
            local = torch.linalg.lstsq(source[use], target[use], rcond=1e-7).solution
            expected.append(source[i] @ local)
        error = normalized_stress(torch.stack(expected) - target.mean(0), target - target.mean(0))
        return coefficients, error

    f, f_error = fit(x, y)
    g, g_error = fit(y, x)
    f_rank, g_rank = int(torch.linalg.matrix_rank(f[:-1])), int(torch.linalg.matrix_rank(g[:-1]))
    cycle = normalized_stress(f[:-1] @ g[:-1], torch.eye(rank, dtype=x.dtype))
    return dict(shared=len(shared), eligible=True, rank=rank, forward_rank=f_rank, reverse_rank=g_rank,
                forward_loo=f_error, reverse_loo=g_error, cycle=cycle,
                accepted=f_rank == g_rank == rank and max(f_error, g_error, cycle) <= .30)


@torch.no_grad()
def run(args):
    task = load_task(args.root, 'DTD')
    data = json.loads(args.probability.read_text())
    if data['base_sha256'] != task.base_sha256:
        raise RuntimeError('Probability data have a different base model')
    arrays = np.load(args.probability.with_suffix('.npz'), allow_pickle=False)
    p = [torch.from_numpy(arrays[f'probabilities_{i}']).sqrt() for i in (1, 2)]
    metric = [distance(q[:, None], q[None, :]) for q in p]
    rows, charts, transitions = [], [], []
    for a, b in ((0, 1), (1, 0)):
        fits = [chart(p[a], metric[a], i, 10) for i in range(len(p[a]))]
        charts.append(fits)
        for i, (indices, z, sigma) in enumerate(fits):
            d = torch.cdist(z, z)
            rows.append(dict(probe=a + 1, center=i, indices=indices.tolist(), rank=z.shape[1],
                             rank_8=chart(p[a], metric[a], i, 8)[1].shape[1],
                             rank_12=chart(p[a], metric[a], i, 12)[1].shape[1],
                             spectrum=sigma.tolist(), coordinates=z.tolist(),
                             fit_stress=normalized_stress(d, metric[a][indices][:, indices]),
                             holdout_stress=normalized_stress(d, metric[b][indices][:, indices])))
        for i, (indices, _, _) in enumerate(fits):
            for j in indices.tolist():
                if i < j:
                    transitions.append(dict(probe=a + 1, source=i, target=j, **transition(fits[i], fits[j])))
    cross = [dict(center=i, **transition(a, b)) for i, (a, b) in enumerate(zip(*charts))]
    paths = data['probes'][0]['probability']['paths']
    checkpoints = data['probes'][0]['checkpoints']
    route = paths['42']['indices']
    progress = paths['42']['fr_progress']
    anchors = [route[min(range(len(route)), key=lambda i: abs(progress[i] - t))] for t in (0., .5, 1.)]
    candidate_rows, required = {}, set()
    for i in anchors:
        j = route.index(i)
        closest = [min(paths[s]['indices'][1:], key=lambda k: float(metric[0][i, k])) for s in ('43', '44')]
        nearby = [v for v in route[max(0, j - 1):j + 2] if v != i]
        candidate_rows[i] = closest + nearby
        for k in [i] + candidate_rows[i]:
            point = checkpoints[k]
            if point['end']:
                required.add((point['seed'], point['end']))
    weights = load_dtd_points(args.trajectories, task, required)

    def at(i):
        point = checkpoints[i]
        return weights[(point['seed'], point['end']) if point['end'] else 0].cuda()

    inputs = [probe(task.validation, q['validation_local_indices'])[0] for q in data['probes']]
    fibers = []
    for i in anchors:
        theta = at(i)
        offsets = torch.stack([at(j) - theta for j in candidate_rows[i]])
        size = offsets.norm(dim=1)
        basis = frame(offsets / size[:, None], len(size))
        j0, j1 = [response(task.predict, theta, basis, x) for x in inputs]
        c0, c1 = j0.T @ j0, j1.T @ j1
        eigenvalues, directions = torch.linalg.eigh(c0)
        own = eigenvalues / c0.diag().median().clamp_min(1e-20)
        holdout = (j1 @ directions).square().sum(0) / c1.diag().median().clamp_min(1e-20)
        tentative = (own <= .02) & (holdout <= .02)
        finite = []
        for v in directions[:, tentative].T:
            direction = v.float() @ basis
            shifts = [float(distance(task.predict(theta, x).double().softmax(-1).sqrt(),
                                     task.predict(theta + .1 * direction, x).double().softmax(-1).sqrt()))
                      for x in inputs]
            finite.append(dict(distances=shifts, accepted=max(shifts) <= .03))
        section = VOQuotient(task.model, task.found, theta)
        source = offsets[0] / size[0]
        gauge = source - section.horizontal(source)
        gauge /= gauge.norm()
        control = [float(response(task.predict, theta, [gauge], x).norm()) for x in inputs]
        pairs = [dict(target=k, parameter_distance=float(size[j]),
                      fr=[float(m[i, k]) for m in metric], near=all(float(m[i, k]) <= .05 for m in metric))
                 for j, k in enumerate(candidate_rows[i][:2])]
        fibers.append(dict(center=i, candidates=candidate_rows[i], fit_relative=own.tolist(),
                           audit_relative=holdout.tolist(), empirical_rank=int((own > .02).sum()),
                           tentative=int(tentative.sum()), finite=finite, known_gauge_response=control,
                           section_pairs=pairs))
        print('fiber', fibers[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spectrum = [q['probability']['eigenvalues'][:5] for q in data['probes']]
    task.existing.write(args.output, dict(charts=rows, transitions=transitions, cross_image_transitions=cross,
                                         fibers=fibers, checkpoints=checkpoints, inpca_top5=spectrum))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--probability', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
