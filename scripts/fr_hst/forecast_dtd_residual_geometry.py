"""Prequential DTD residual forecasts, finite replay and functional angle checks."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from analyze_dtd_expert_geometry import load_expert
from measure_dtd_dynamic_mechanisms import load_trace
from run_dtd_carrier_diagnostics import Operator, metrics
from train_eight_expert_queue import BASE, HEADS, ROOT, build_model, existing, read, splits


HISTORY = 5
FIT_CARRIER = 4
RANK = 3
METHODS = ('A_fixed', 'B_rolling', 'C_quadratic', 'D_atlas_innovation')


def energy(v, gram):
    return float(v @ gram @ v)


def frame(rows, gram, rank):
    norms = np.sqrt(np.maximum(0., np.einsum('ij,jk,ik->i', rows, gram, rows)))
    unit = rows / np.maximum(norms[:, None], 1e-12)
    values, vectors = np.linalg.eigh(unit @ gram @ unit.T)
    if values[-rank] <= 1e-10:
        raise ValueError('Insufficient independent historical directions')
    return ((vectors[:, -rank:] / np.sqrt(values[-rank:])).T @ unit)[::-1]


def standardize(history, test):
    mean = history.mean(0)
    scale = history.std(0).clip(min=1e-8)
    return (history - mean) / scale, ((test - mean) / scale).clip(-3, 3)


def regression(x, y, target, quadratic=False):
    z, t = standardize(x, target)
    design = np.column_stack((np.ones(len(z)), z))
    test = np.r_[1., t]
    ridge = [0., .25, .25]
    if quadratic:
        design = np.column_stack((design, z[:, 0] ** 2, z[:, 0] * z[:, 1], z[:, 1] ** 2))
        test = np.r_[test, t[0] ** 2, t[0] * t[1], t[1] ** 2]
        ridge += [1., 1., 1.]
    return test @ np.linalg.solve(design.T @ design + np.diag(ridge), design.T @ y)


def in_frame(rows, x, target, basis, gram, quadratic=False):
    coordinates = rows @ gram @ basis.T
    return regression(x, coordinates, target, quadratic) @ basis


def atlas(rows, x, target, gram, utility):
    z, t = standardize(x, target)
    members = (np.arange(3), np.arange(2, 5))
    chart = min(range(2), key=lambda k: np.linalg.norm(t - z[members[k]].mean(0)))
    indices = members[chart]
    local = frame(rows[indices], gram, 2)
    innovations = []
    for i in range(2, len(rows)):
        earlier = frame(rows[:i], gram, 2)
        prediction = in_frame(rows[:i], x[:i], x[i], earlier, gram, quadratic=True)
        error = rows[i] - prediction
        innovations.append(error - (local @ gram @ error) @ local)
    innovations = np.stack(innovations)
    norms = np.sqrt(np.maximum(0., np.einsum('ij,jk,ik->i', innovations, gram, innovations)))
    if not (norms > 1e-10).any():
        raise ValueError('Historical innovation channel has no nonzero direction')
    # Prior residual-only CE is a proxy for historical usefulness, not the CE of
    # the projected novelty itself. Past nonpositive/withheld values get zero weight.
    weights = np.maximum(utility[2:], 0) * (norms > 1e-10)
    fallback = not weights.any()
    if fallback:
        weights = (norms > 1e-10).astype(float)
    pool = innovations / np.maximum(norms[:, None], 1e-12)
    pool *= np.sqrt(weights / weights.max())[:, None]
    values, vectors = np.linalg.eigh(pool @ gram @ pool.T)
    innovation = vectors[:, -1] @ pool / np.sqrt(values[-1])
    local_prediction = in_frame(rows[indices], x[indices], target, local, gram, quadratic=True)
    innovation_prediction = float(regression(x, rows @ gram @ innovation, target)) * innovation
    return np.vstack((local, innovation)), local_prediction + innovation_prediction, local_prediction, dict(
        chart=chart, chart_history_positions=indices.tolist(), utility_weights=weights.tolist(),
        uniform_fallback=fallback, historical_innovation_norms=norms.tolist())


def state_features(probability, widths):
    entropy = -(probability * np.log(np.maximum(probability, 1e-300))).sum(-1).mean(-1)
    overlap = (np.sqrt(probability[1:]) * np.sqrt(probability[:-1])).sum(-1).clip(0, 1)
    velocity = np.sqrt(np.mean((2 * np.arccos(overlap)) ** 2, axis=-1)) / widths[:, None]
    return np.stack((entropy, np.vstack((np.zeros((1, 2)), velocity))), axis=-1)


def coefficients(gram):
    atoms = np.eye(len(gram))
    basis = frame(atoms[:FIT_CARRIER], gram, RANK)
    carrier = (atoms @ gram @ basis.T) @ basis
    return carrier, atoms - carrier


def predict_window(k, group, gram, carrier, residual, states, widths, usefulness, previous_d):
    # All fitting routines receive only a k-by-k prefix and rows supported on that prefix.
    historical = np.arange(k - HISTORY, k)
    metric = gram[:k, :k]
    x, target = states[historical, group], states[k, group]
    rates = residual[historical, :k] / widths[historical, None]
    fixed = residual[FIT_CARRIER:FIT_CARRIER + HISTORY, :k] / widths[FIT_CARRIER:FIT_CARRIER + HISTORY, None]
    carrier_forecast = regression(x, carrier[historical, :k] / widths[historical, None], target) * widths[k]
    results = {}

    def add(name, basis, forecast, start, **details):
        np.testing.assert_allclose(basis @ metric @ basis.T, np.eye(len(basis)), atol=1e-7)
        results[name] = dict(basis=basis.tolist(), residual=(forecast * widths[k]).tolist(),
                             fit_seconds=time.perf_counter() - start, **details)

    start = time.perf_counter()
    basis = frame(fixed, metric, RANK)
    add(METHODS[0], basis, in_frame(rates, x, target, basis, metric), start)
    start = time.perf_counter()
    basis = frame(rates, metric, RANK)
    add(METHODS[1], basis, in_frame(rates, x, target, basis, metric), start)
    start = time.perf_counter()
    basis = frame(rates, metric, RANK)
    add(METHODS[2], basis, in_frame(rates, x, target, basis, metric, quadratic=True), start)
    start = time.perf_counter()
    utility = usefulness[historical, group]
    basis, prediction, local, details = atlas(rates, x, target, metric, utility)
    add(METHODS[3], basis, prediction, start, local_residual=(local * widths[k]).tolist(), **details)
    feedback = None
    if previous_d is not None:
        shadow = rates.copy()
        previous = np.asarray(previous_d)
        shadow[-1] = 0.
        shadow[-1, :len(previous)] = previous / widths[k - 1]
        shadow_utility = utility.copy()
        shadow_utility[-1] = 0.
        alt, alt_prediction, _, details = atlas(shadow, x, target, metric, shadow_utility)
        feedback = dict(basis=alt.tolist(), residual=(alt_prediction * widths[k]).tolist(), **details)
    return dict(index=k, fit_group=group, audit_group=1 - group, historical_indices=historical.tolist(),
                fixed_basis_indices=list(range(FIT_CARRIER, FIT_CARRIER + HISTORY)),
                state=target.tolist(), known_width=int(widths[k]), carrier=carrier_forecast.tolist(),
                methods=results, feedback_control=feedback)


def forecast(args):
    with np.load(args.bank) as data:
        steps, gram, probability = (data[key] for key in ('steps', 'parameter_gram', 'probabilities'))
    widths = np.diff(steps)
    assert len(widths) == 16 and probability.shape == (17, 2, 32, 47)
    carrier, residual = coefficients(gram)
    history = read(args.replay)['windows']
    usefulness = np.zeros((len(widths), 2))
    for i, row in enumerate(history, FIT_CARRIER):
        assert (row['start'], row['end']) == (steps[i], steps[i + 1])
        np.testing.assert_allclose(energy(carrier[i], gram) / gram[i, i], row['carrier_energy_fraction'], atol=1e-5)
        usefulness[i] = [r['conditions']['residual']['ce_gain'] for r in row['groups']]
    features = state_features(probability, widths)
    forecasts, previous = [], [None, None]
    for k in range(FIT_CARRIER + HISTORY, len(widths)):
        for group in (0, 1):
            prediction = predict_window(k, group, gram, carrier, residual, features, widths, usefulness, previous[group])
            # The same result must be obtained with all unseen observations removed.
            prefix = predict_window(k, group, gram[:k, :k], carrier[:k, :k], residual[:k, :k],
                                    features[:k + 1], widths[:k + 1], usefulness[:k], previous[group])
            for method in METHODS:
                np.testing.assert_array_equal(prediction['methods'][method]['residual'], prefix['methods'][method]['residual'])
                np.testing.assert_array_equal(prediction['methods'][method]['basis'], prefix['methods'][method]['basis'])
            np.testing.assert_array_equal(prediction['carrier'], prefix['carrier'])
            if prediction['feedback_control']:
                np.testing.assert_array_equal(prediction['feedback_control']['residual'], prefix['feedback_control']['residual'])
            prediction.update(start=int(steps[k]), end=int(steps[k + 1]))
            forecasts.append(prediction)
            previous[group] = prediction['methods'][METHODS[3]]['residual']
    return dict(steps=steps.tolist(), bank_sha256=existing.sha(args.bank), replay_sha256=existing.sha(args.replay),
                history=HISTORY, rank=RANK, base_carrier_windows=FIT_CARRIER, prefix_invariance_verified=True,
                forecasts=forecasts)


def principal(a, b, metric=None):
    cross = a @ b.T if metric is None else a @ metric @ b.T
    singular = np.linalg.svd(cross, compute_uv=False).clip(0, 1)
    return dict(degrees=np.degrees(np.arccos(singular)).tolist(), mean_squared_overlap=float(np.mean(singular ** 2)))


def output_basis(responses):
    vectors, singular, _ = np.linalg.svd(responses.T, full_matrices=False)
    return vectors[:, singular > singular[0] * 1e-8].T


def response(op, x, vector):
    logits, jvp = op.logit_jvp(x, vector)
    p = logits.double().softmax(-1)
    return (p.sqrt() * (jvp - (p * jvp).sum(-1, keepdim=True))).flatten().cpu().numpy() / np.sqrt(len(x))


def evaluate_window(op, states, images, bank, gram, residual, plan):
    k = plan['index']
    size = len(gram)
    x, y = images[plan['audit_group']]
    op.load(states[k])
    old = op.evaluate(x)
    actual = op.evaluate(x, bank[k])
    if not torch.allclose(actual.double().softmax(-1).cpu(), states[k + 1]['probability'][plan['audit_group']],
                          atol=2e-5, rtol=0):
        raise ValueError(f'Replay did not reproduce step {states[k + 1]["step"]}')
    full = metrics(old, actual, y)
    known = np.zeros(size)
    known[:k] = plan['carrier']
    true_residual = residual[k]

    def decode(a):
        return torch.as_tensor(a, dtype=bank.dtype, device=bank.device) @ bank

    # Future states enter only this scoring routine, after the forecast file is written.
    audit = output_basis(np.stack([response(op, x, decode(r)) for r in residual[k:k + RANK]]))
    target_response = response(op, x, decode(true_residual))
    cache = {}

    def direction_response(a):
        key = np.asarray(a).tobytes()
        if key not in cache:
            cache[key] = response(op, x, decode(a))
        return cache[key]

    def score(row, with_angles=True):
        predicted = np.zeros(size)
        predicted[:k] = row['residual']
        estimate = predicted + known
        exact = np.eye(size)[k]
        gain = metrics(old, op.evaluate(x, decode(estimate)), y)
        residual_gain = metrics(old, op.evaluate(x, decode(predicted)), y)
        basis = np.zeros((RANK, size))
        basis[:, :k] = row['basis']
        result = dict(update_relative_error=np.sqrt(energy(estimate - exact, gram) / gram[k, k]),
                      residual_relative_error=np.sqrt(energy(predicted - true_residual, gram) / energy(true_residual, gram)),
                      parameter_energy_capture=float(np.sum((basis @ gram @ true_residual) ** 2) / energy(true_residual, gram)),
                      full_forecast=gain, residual_only=residual_gain)
        if full['ce_gain'] > 0:
            result['ce_retention'] = gain['ce_gain'] / full['ce_gain']
        if with_angles:
            current = output_basis(np.stack([direction_response(v) for v in basis]))
            response_estimate = direction_response(predicted)
            cosine = target_response @ response_estimate / (np.linalg.norm(target_response) * np.linalg.norm(response_estimate))
            result.update(future_functional_angles=principal(current, audit),
                          forecast_response_angle_deg=float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))),
                          next_response_energy_capture=float(np.sum((current @ target_response) ** 2) / np.sum(target_response ** 2)),
                          future_functional_reference_rank=len(audit))
        return result

    results = {method: score(row) for method, row in plan['methods'].items()}
    actual_d = plan['methods'][METHODS[3]]
    chart_only = known.copy()
    chart_only[:k] += actual_d['local_residual']
    results[METHODS[3]]['chart_only'] = metrics(old, op.evaluate(x, decode(chart_only)), y)
    feedback = None
    if plan['feedback_control']:
        control = plan['feedback_control']
        a, b = [np.pad(np.asarray(row['basis']), ((0, 0), (0, size - k))) for row in (actual_d, control)]
        output_a, output_b = [output_basis(np.stack([direction_response(v) for v in basis])) for basis in (a, b)]
        no_feedback = score(control, with_angles=False)
        feedback = dict(no_real_feedback=no_feedback, with_real_feedback_chart=actual_d['chart'],
                        no_real_feedback_chart=control['chart'], parameter_angles=principal(a, b, gram),
                        functional_angles=principal(output_a, output_b),
                        ce_gain_delta=results[METHODS[3]]['full_forecast']['ce_gain']
                        - no_feedback['full_forecast']['ce_gain'])
    return dict(start=plan['start'], end=plan['end'], fit_group=plan['fit_group'], audit_group=plan['audit_group'],
                true_full_update=full, carrier_only=metrics(old, op.evaluate(x, decode(known)), y),
                methods=results, feedback=feedback)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = forecast(args)
    existing.write(args.output / 'forecast_plan.json', plan)
    print('Locked 14 forecasts after strict prefix checks', flush=True)
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, states = load_trace(args.trace)
    assert [s['step'] for s in states] == plan['steps']
    head = existing.ClassificationHead.load(str(HEADS / 'DTD.pt')).cuda().eval()
    encoder, model, found, flat = build_model(existing, head)
    del flat
    model.requires_grad_(False).eval()
    base_hash = existing.sha(BASE)
    expert, expert_manifest = load_expert(args.queue, 'DTD', [name for name, _ in found], base_hash)
    visual = encoder.model.visual
    for key, value in visual.state_dict().items():
        if not torch.equal(value.cpu(), states[0]['visual'][key]) or not torch.equal(expert[key], states[-1]['visual'][key]):
            raise ValueError('Trajectory endpoint differs from the expert provenance')
    op = Operator(model, visual)
    fit, _ = splits('DTD', encoder.val_preprocess, encoder.val_preprocess)
    lookup = {raw: i for i, raw in enumerate(fit.indices)}
    images = []
    for ids, labels in zip(manifest['raw_train_indices'], manifest['labels']):
        batch = [fit[lookup[i]] for i in ids]
        assert [y for _, y in batch] == labels
        images.append((torch.stack([x for x, _ in batch]).cuda(), torch.tensor(labels, device='cuda')))
    with np.load(args.bank) as source:
        gram = source['parameter_gram']
    _, residual = coefficients(gram)
    bank = torch.stack([torch.cat([s['delta_visual'][key].flatten().cuda() for key in op.keys]) for s in states[1:]])
    measured_norms = bank.norm(dim=1).double().cpu().numpy() ** 2
    np.testing.assert_allclose(measured_norms, np.diag(gram), rtol=5e-5)
    records = []
    for row in plan['forecasts']:
        result = evaluate_window(op, states, images, bank, gram, residual, row)
        records.append(result)
        print(f'Window {row["start"]}–{row["end"]}, fit group {row["fit_group"]} measured', flush=True)
    existing.write(args.output / 'results.json', dict(
        forecast_plan_sha256=existing.sha(args.output / 'forecast_plan.json'), trace_sha256=existing.sha(args.trace / 'trace.json'),
        base_sha256=base_hash, expert_sha256=expert_manifest['checkpoint_sha256'],
        image_indices=manifest['raw_train_indices'], methods=METHODS,
        common_scoring_budget=dict(forward_per_main_method=2, jvp_per_main_method_upper_bound=4),
        rows=records))


if __name__ == '__main__':
    main()
