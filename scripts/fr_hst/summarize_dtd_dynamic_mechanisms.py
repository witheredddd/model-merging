"""Verify DTD dynamic measurements and write a compact, auditable result bank."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def modes(matrix):
    values, directions = np.linalg.eigh(matrix)
    return values[::-1], directions[:, ::-1]


def cosine(a, b):
    return float(np.vdot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def trace_share(matrix, directions):
    return float(np.trace(directions.T @ matrix @ directions) / np.trace(matrix))


def cumulative_spectrum(matrix):
    values, _ = modes(matrix)
    positive = np.maximum(values, 0)
    return dict(eigenvalues=values.tolist(), top4_share=float(positive[:4].sum() / positive.sum()),
                rank90=int(np.searchsorted(np.cumsum(positive) / positive.sum(), .9) + 1))


def parameter_stats(gram, coefficients):
    lengths = np.sqrt(np.diag(gram))
    unit = gram / np.outer(lengths, lengths)
    split = len(gram) // 2
    values, vectors = modes(unit[:split, :split])
    projected = unit[split:, :split] @ (vectors[:, :4] / np.sqrt(values[:4]))
    return dict(raw=cumulative_spectrum(gram), unit=cumulative_spectrum(unit),
                consecutive_cosine=np.diag(unit, k=1).tolist(),
                late_capture_from_early_rank4=np.sum(projected ** 2, axis=1).tolist(),
                fixed_atom_parameter_energy_fraction=(np.sum(coefficients ** 2, axis=1) / lengths ** 2).tolist(),
                norms=lengths.tolist())


def probability_stats(prob):
    roots = np.sqrt(prob)
    overlap = np.clip(np.einsum('tgic,sgic->gtsi', roots, roots), 1e-300, 1)
    divergence = -np.log(overlap).mean(-1)
    fr = np.sqrt(np.mean((2 * np.arccos(overlap)) ** 2, axis=-1))
    np.einsum('gii->gi', divergence)[:] = 0
    np.einsum('gii->gi', fr)[:] = 0
    center = np.eye(len(prob)) - np.ones((len(prob), len(prob))) / len(prob)
    inpca = [cumulative_spectrum(-.5 * center @ d @ center) for d in divergence]
    return fr, divergence, inpca


def anchor_stats(grams, responses, steps, fr):
    fits = [modes(row[0])[1] for row in grams]
    norm = np.linalg.norm(grams, axis=(-2, -1))
    ntk = np.empty((2, len(steps), len(steps)))
    for a, ra in enumerate(responses):
        for b, rb in enumerate(responses):
            cross = ra.transpose(0, 2, 1) @ rb
            ntk[:, a, b] = np.sum(cross ** 2, axis=(1, 2)) / (norm[a] * norm[b])
    np.testing.assert_allclose(np.diagonal(ntk, axis1=1, axis2=2), 1, atol=1e-10)
    results = []
    for t, (step, (a, b), fit) in enumerate(zip(steps, grams, fits)):
        results.append(dict(step=step, spectrum=[cumulative_spectrum(c) for c in (a, b)],
                            image_group_gram_cosine=cosine(a, b),
                            group_b_rank4_local_a=trace_share(b, fit[:, :4]),
                            group_b_rank4_frozen_a=trace_share(b, fits[0][:, :4]),
                            bottom4_group_b_relative_average=12 * trace_share(b, fit[:, -4:]),
                            previous_bottom4_group_b_relative_average=(12 * trace_share(b, fits[t - 1][:, -4:])
                                                                        if t else None),
                            cosine_gram_to_base=[cosine(c, base) for c, base in zip((a, b), grams[0])],
                            cosine_ntk_to_base=ntk[:, t, 0].tolist()))
    off = np.triu_indices(len(steps), k=1)
    relation = [float(np.corrcoef(fr[g][off], 1 - ntk[g][off])[0, 1]) for g in range(2)]
    return results, ntk, relation


def gradient_terms(g):
    denominator = g[3, 3]
    change = float(np.sqrt(denominator))
    norms = np.sqrt(np.maximum(0, g.diagonal()[:3]))
    signed = g[:3, 3] / denominator
    orthogonal = np.sqrt(np.maximum(0, norms ** 2 / denominator - signed ** 2))
    closure_vector = np.array([1., 1., 1., -1.])
    relative_closure = float(np.sqrt(max(0., closure_vector @ g @ closure_vector / denominator)))
    if relative_closure > 1e-6:
        raise ValueError('Gradient terms did not close')
    return dict(gradient_change_norm=change, term_norms=norms.tolist(), signed_share=signed.tolist(),
                orthogonal_share=orthogonal.tolist(), interaction_over_main_norms=float(norms[2] / norms[:2].sum()),
                relative_closure=relative_closure)


def interval_stats(i, j, grad, linear, coeff, responses, grams, prob, fr, ntk, steps):
    exact = 2 / np.sqrt(prob.shape[2]) * (np.sqrt(prob[j]) - np.sqrt(prob[i])).reshape(2, -1)
    groups = []
    upper = np.triu_indices(len(coeff), 1)
    for g, (c, r) in enumerate(zip(grams[i], responses[i])):
        contributions = c * np.outer(coeff, coeff)
        diag = np.trace(contributions)
        terms = 2 * contributions[upper]
        fixed_jvp = r @ coeff
        np.testing.assert_allclose(contributions.sum(), fixed_jvp @ fixed_jvp, rtol=1e-8, atol=1e-12)
        groups.append(dict(gradient=gradient_terms(grad[g]),
                           coupling_ratio=float(contributions.sum() / diag),
                           diagonal_energy=float(diag),
                           positive_interaction_energy=float(terms[terms > 0].sum()),
                           negative_interaction_energy=float(terms[terms < 0].sum()),
                           full_jvp_norm=float(np.linalg.norm(linear[g])),
                           true_chord_norm=float(np.linalg.norm(exact[g])),
                           taylor_relative_error=float(np.linalg.norm(exact[g] - linear[g]) / np.linalg.norm(exact[g])),
                           projected_jvp_relative_error=float(np.linalg.norm(fixed_jvp - linear[g]) / np.linalg.norm(linear[g])),
                           full_jvp_chord_cosine=cosine(linear[g], exact[g]),
                           probability_fr=float(fr[g, i, j]),
                           restricted_ntk_cosine=float(ntk[g, i, j])))
    return dict(start=steps[i], end=steps[j], width=steps[j] - steps[i], groups=groups)


def matched_epochs(fine, coarse):
    records = []
    for wide in coarse:
        parts = [r for r in fine if r['start'] >= wide['start'] and r['end'] <= wide['end']]
        if len(parts) != 2 or sum(r['width'] for r in parts) != wide['width']:
            raise ValueError('Unexpected epoch partition')
        for part in parts:
            groups = []
            for a, b in zip(part['groups'], wide['groups']):
                g, h = a['gradient'], b['gradient']
                groups.append(dict(interaction_norm_ratio=g['term_norms'][2] / h['term_norms'][2],
                                   interaction_relative_ratio=g['interaction_over_main_norms'] / h['interaction_over_main_norms'],
                                   taylor_error_ratio=a['taylor_relative_error'] / b['taylor_relative_error']))
            records.append(dict(start=part['start'], end=part['end'], epoch_start=wide['start'], epoch_end=wide['end'],
                                width_ratio=part['width'] / wide['width'], groups=groups))
    return records


def historical_summary(path):
    rows = read(path)['windows']
    result = []
    for seed, probe in sorted({(r['seed'], r['probe']) for r in rows}):
        group = [r['interaction_geometry'] for r in rows if (r['seed'], r['probe']) == (seed, probe)]
        result.append(dict(seed=seed, probe=probe, n=len(group), positive=sum(r['parallel'] > 0 for r in group),
                           median_parallel=float(np.median([r['parallel'] for r in group])),
                           median_orthogonal=float(np.median([r['orthogonal'] for r in group]))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--historical', type=Path)
    parser.add_argument('--reference', type=Path, help='Earlier independent endpoint Fisher measurements')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.measurements
    meta, completed = read(root / 'manifest.json'), read(root / 'complete.json')
    steps = meta['steps']
    probs, grams, response_rows = [], [], []
    for step in steps:
        with np.load(root / f'anchor_{step:04d}.npz') as f:
            probs.append(f['probability'])
            grams.append(f['gram'])
            response_rows.append(f['response'].reshape(2, -1, 48) * np.sqrt(2))
    prob, grams, responses = map(np.stack, (probs, grams, response_rows))
    assert grams.shape == (17, 2, 48, 48) and prob.shape == (17, 2, 32, 47)
    np.testing.assert_allclose(responses.transpose(0, 1, 3, 2) @ responses, grams, atol=1e-10)
    assert np.linalg.eigvalsh(grams).min() >= -1e-8
    with np.load(root / 'parameter.npz') as f:
        param_gram, coefficients = f['gram'], f['coefficients']
    assert np.linalg.eigvalsh(param_gram).min() >= -1e-8
    fr, divergence, inpca = probability_stats(prob)
    anchors, ntk, relation = anchor_stats(grams, responses, steps, fr)
    intervals, gradients, linear_responses = {}, {}, {}
    for key, pairs in completed.items():
        rows, grads, linear = [], [], []
        for i, j in pairs:
            with np.load(root / f'interval_{steps[i]:04d}_{steps[j]:04d}.npz') as f:
                grad, derivative = f['gradient_gram'], f['derivative']
            rows.append(interval_stats(i, j, grad, derivative, coefficients[i:j].sum(axis=0),
                                       responses, grams, prob, fr, ntk, steps))
            grads.append(grad)
            linear.append(derivative)
        intervals[key], gradients[key], linear_responses[key] = rows, np.stack(grads), np.stack(linear)

    result = dict(provenance=meta, term_order=['residual', 'jacobian', 'interaction', 'net_change'],
                  parameter=parameter_stats(param_gram, coefficients), anchors=anchors, intervals=intervals,
                  matched_epoch_pieces=matched_epochs(intervals['fine'], intervals['coarse']),
                  probability_inpca=inpca, probability_ntk_distance_correlation=relation)
    if args.historical:
        result['historical_three_seed_check'] = historical_summary(args.historical)
        result['historical_source_sha256'] = hashlib.sha256(args.historical.read_bytes()).hexdigest()
    if args.reference:
        previous = read(args.reference)
        assert previous['matrix_names'] == meta['matrix_names']
        assert previous['probe_raw_train_indices']['DTD'] == meta['raw_train_indices']
        reference = np.array([previous['grams'][k]['DTD'] for k in ('base', 'dtd_expert')])
        relative = np.linalg.norm(grams[[0, -1]] - reference, axis=(-2, -1)) / np.linalg.norm(reference, axis=(-2, -1))
        np.testing.assert_array_less(relative, 5e-5)
        result['endpoint_independent_gram_relative_error'] = relative.tolist()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    np.savez_compressed(args.output / 'mechanism_arrays.npz', steps=steps, parameter_gram=param_gram,
                        parameter_coefficients=coefficients, fisher_grams=grams, fisher_responses=responses,
                        probabilities=prob, probability_fr=fr, bhattacharyya=divergence, restricted_ntk_cosine=ntk,
                        fine_indices=completed['fine'], coarse_indices=completed['coarse'],
                        fine_gradient_gram=gradients['fine'], coarse_gradient_gram=gradients['coarse'],
                        fine_full_jvp=linear_responses['fine'], coarse_full_jvp=linear_responses['coarse'])
    print(f'Verified {len(steps)} anchors, {len(intervals["fine"])} windows, {len(intervals["coarse"])} epoch intervals')


if __name__ == '__main__':
    main()
