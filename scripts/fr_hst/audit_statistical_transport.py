"""Compare DTD parameter angles with prediction tangents at a common base point."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from collect_dtd_updates import load_experiment
from study_dtd_quotient import predictor, response
from train_atlas import embedding
from train_statistical_bundle import transport


def angle(a, b):
    return float(F.cosine_similarity(a, b, dim=0).abs().clamp(0, 1).acos().rad2deg())


def read(path):
    return torch.load(path, weights_only=True, map_location='cpu', mmap=True)


@torch.no_grad()
def audit(args):
    existing, model, found, center, _, validation, _, _, base_hash = load_experiment(args.root, 44)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((args.experiment / 'real44/complete.json').read_text())
    if manifest['names'] != [n for n, _ in found] or manifest['base_sha256'] != base_hash:
        raise RuntimeError('DTD trajectory provenance mismatch')
    ids = torch.randperm(len(validation.dataset), generator=torch.Generator().manual_seed(193)).tolist()[:48]
    probes = torch.stack([validation.dataset[i][0] for i in ids]).cuda()
    predict = predictor(model, found)
    path = args.experiment / 'drift_diagnostic/replay44'
    proof = json.loads((path / 'complete.json').read_text())
    if (proof['seed'] != 44 or proof['base_sha256'] != base_hash
            or any(row['relative_error'] != 0 for row in proof['checks'])):
        raise RuntimeError('Retrospective PCA does not match the original trajectory')
    centers = [[], []]
    reference = []
    previous_mode = None
    rows = []
    for window in sorted((args.experiment / 'real44').glob('update_*.pt')):
        state = read(window)
        end = state['step']
        delta = state['delta'].cuda()
        mode = read(path / f'signals_{end:04d}.pt')['oracle'][-1].cuda()
        angles, path_difference = [], []
        for i, x in enumerate(probes.chunk(2)):
            current = embedding(predict(center, x))
            tangent = response(predict, center, [mode], x)[:, 0]
            centers[i].append(current)
            moved = tangent
            for later, earlier in zip(reversed(centers[i][1:]), reversed(centers[i][:-1])):
                moved = transport(later, earlier, moved, len(x))
            direct = transport(current, centers[i][0], tangent, len(x))
            path_difference.append(float((moved - direct).norm() / tangent.norm().clamp_min(1e-12)))
            if len(reference) > i:
                angles.append(angle(reference[i], moved))
                reference[i] = moved
            else:
                reference.append(moved)
        rows.append(dict(end=end, euclidean_angle=None if previous_mode is None else angle(previous_mode, mode),
                         transported_fisher_angles=angles, path_difference=path_difference))
        previous_mode = mode
        center = center + delta
        if end == manifest['epoch_1']['step']:
            center = read(args.experiment / 'real44/anchor_epoch1.pt').cuda()
    existing.write(args.output, dict(seed=44, base_sha256=base_hash, windows=rows,
                                     validation_local_indices=ids,
                                     source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args())
