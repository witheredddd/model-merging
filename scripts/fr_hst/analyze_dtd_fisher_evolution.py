"""Audit Fisher spectra and direction correspondence using the saved response bank."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


RANKS = (2, 4, 8)


def principal(cross):
    singular = np.linalg.svd(cross, compute_uv=False).clip(0, 1)
    return dict(degrees=np.degrees(np.arccos(singular)).tolist(), mean_squared_overlap=float(np.mean(singular ** 2)))


def compare(a, b):
    cross = a[:, :8].T @ b[:, :8]
    row, col = linear_sum_assignment(-np.abs(cross))
    return dict(angles={r: principal(cross[:r, :r]) for r in RANKS},
                matched_top8=[dict(source=int(i + 1), target=int(j + 1), absolute_cosine=float(abs(cross[i, j])))
                              for i, j in zip(row, col)])


def retention(matrix, old, local):
    values = {}
    for r in RANKS:
        directions = old[:, :r]
        retained = np.trace(directions.T @ matrix @ directions)
        values[r] = dict(response=float(retained), total_fraction=float(retained / np.trace(matrix)),
                         optimal_fraction=float(retained / local[:r].sum()))
    return values


def parameter_compare(gram, steps, vec):
    norms = np.sqrt(np.diag(gram))
    cosine = gram / np.outer(norms, norms)
    rows = []
    for r in RANKS:
        for i in range(0, len(gram) - 2 * r + 1, r):
            # Disjoint blocks of r normalized updates. Orthonormalize their spans in the full parameter metric.
            first, second = slice(i, i + r), slice(i + r, i + 2 * r)
            a, u = np.linalg.eigh(cosine[first, first])
            b, v = np.linalg.eigh(cosine[second, second])
            if min(a.min(), b.min()) <= 1e-10:
                raise ValueError('A matched parameter window has insufficient rank')
            parameter = principal((u / np.sqrt(a)).T @ cosine[first, second] @ (v / np.sqrt(b)))
            fisher = [principal(vec[i + r, g, :, :r].T @ vec[i + 2 * r, g, :, :r]) for g in range(2)]
            rows.append(dict(rank=r, window_steps=[int(steps[k]) for k in (i, i + r, i + 2 * r)],
                             parameter=parameter, fisher=fisher))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.bank) as f:
        grams, steps, param = (f[key] for key in ('fisher_grams', 'steps', 'parameter_gram'))
    assert grams.shape == (17, 2, 48, 48)
    eigenvalues, vectors = np.linalg.eigh(grams)
    values, vectors = eigenvalues[..., ::-1], vectors[..., ::-1]
    np.testing.assert_allclose((vectors * values[:, :, None, :]) @ vectors.swapaxes(-1, -2), grams, atol=1e-9)
    np.testing.assert_array_less(-1e-8, values)
    spectra, pairs, cross_group = [], [], []
    for i, step in enumerate(steps):
        row = []
        for group in (0, 1):
            value = values[i, group]
            same = retention(grams[i, group], vectors[0, group], value)
            transfer = retention(grams[i, group], vectors[i, 1 - group], value)
            base_transfer = retention(grams[i, group], vectors[0, 1 - group], value)
            for r in RANKS:
                same[r]['relative_to_base_response'] = same[r]['response'] / values[0, group, :r].sum()
            row.append(dict(group=group, eigenvalues=value.tolist(), trace=float(value.sum()),
                            cumulative={r: float(value[:r].sum() / value.sum()) for r in RANKS},
                            eigengap_relative={r: float((value[r - 1] - value[r]) / value[0]) for r in RANKS},
                            frozen_base=same, simultaneous_other_group=transfer, frozen_base_other_group=base_transfer))
        spectra.append(dict(step=int(step), groups=row))
        cross_group.append(dict(step=int(step), **compare(vectors[i, 0], vectors[i, 1])))
        for j in range(i + 1, len(steps)):
            matches = [compare(vectors[i, g], vectors[j, g]) for g in (0, 1)]
            pairs.append(dict(start=int(step), end=int(steps[j]), adjacent=j == i + 1, groups=matches))
    result = dict(bank_sha256=hashlib.sha256(args.bank.read_bytes()).hexdigest(), ranks=RANKS,
                  spectra=spectra, state_pairs=pairs, cross_image_group=cross_group,
                  matched_parameter_subspaces=parameter_compare(param, steps, vectors))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'fisher_evolution.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(args.output / 'fisher_eigenpairs.npz', steps=steps, eigenvalues=values, eigenvectors=vectors)
    print(f'{len(steps)} states, {len(pairs)} temporal pairs, {len(cross_group)} cross-image comparisons verified')


if __name__ == '__main__':
    main()
