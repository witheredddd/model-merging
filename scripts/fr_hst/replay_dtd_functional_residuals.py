"""Replay Fisher and Euclidean decompositions of held-out DTD residuals."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from forecast_dtd_residual_geometry import FIT_CARRIER, HISTORY, coefficients, replay_context, response
from run_dtd_carrier_diagnostics import metrics
from train_eight_expert_queue import existing, read


BETAS = (0., .25, .5, .75, 1.)
RELATIVE_RIDGE = .001


def solve(matrix):
    gram = matrix[:-1, :-1]
    cross = matrix[:-1, -1]
    ridge = float(RELATIVE_RIDGE * np.trace(gram) / len(gram))
    regularized = gram + ridge * np.eye(len(gram))
    alpha = np.linalg.solve(regularized, cross)
    np.testing.assert_allclose(regularized @ alpha, cross, rtol=1e-8, atol=1e-12)
    return alpha, dict(ridge=ridge, condition_number=float(np.linalg.cond(regularized)),
                       eigenvalues=np.linalg.eigvalsh(gram).tolist())


def energy_split(matrix, alpha):
    parallel = np.r_[alpha, 0.]
    innovation = np.r_[-alpha, 1.]
    scale = matrix[-1, -1]
    out = dict(total=float(scale), parallel_fraction=float(parallel @ matrix @ parallel / scale),
               innovation_fraction=float(innovation @ matrix @ innovation / scale),
               cross_fraction=float(2 * parallel @ matrix @ innovation / scale))
    np.testing.assert_allclose(sum(out[k] for k in ('parallel_fraction', 'innovation_fraction', 'cross_fraction')),
                               1., atol=1e-8)
    return out


def score(old, full, logits, labels):
    row = metrics(old, logits, labels)
    row.update(ce=float(F.cross_entropy(logits.double(), labels)),
               accuracy=float((logits.argmax(-1) == labels).double().mean()),
               kl_from_full=metrics(full, logits, labels)['kl_from_start'])
    return row


def replay(op, images, labels, old, full, carrier, parallel, innovation, baseline):
    result = {
        'parallel_only': score(old, full, op.evaluate(images, parallel), labels),
        'innovation_only': score(old, full, op.evaluate(images, innovation), labels),
        'carrier_plus_innovation': score(old, full, op.evaluate(images, carrier + innovation), labels),
    }
    sweep = []
    for beta in BETAS:
        update = carrier + parallel + beta * innovation
        logits = op.evaluate(images, update)
        if beta == 1.:
            torch.testing.assert_close(logits.double().softmax(-1), full.double().softmax(-1), atol=2e-5, rtol=0)
        row = score(old, full, logits, labels)
        row.update(beta=beta, ce_gain_delta_vs_full=row['ce_gain'] - baseline['full']['ce_gain'],
                   ce_gain_delta_vs_carrier=row['ce_gain'] - baseline['carrier']['ce_gain'])
        if baseline['full']['ce_gain'] > 0:
            row['ce_retention'] = row['ce_gain'] / baseline['full']['ce_gain']
        sweep.append(row)
    interaction = (sweep[-1]['ce_gain'] - sweep[0]['ce_gain']
                   - result['carrier_plus_innovation']['ce_gain'] + baseline['carrier']['ce_gain'])
    return dict(components=result, beta_sweep=sweep, ce_interaction=interaction)


def measure_window(ctx, k, gram, carrier_rows, residual_rows, previous):
    q = min(HISTORY, k - FIT_CARRIER)
    history = np.arange(k - q, k)
    indices = np.r_[history, k]
    rows = residual_rows[indices]
    parameter_gram = rows @ gram @ rows.T
    directions = torch.as_tensor(rows, dtype=ctx.bank.dtype, device=ctx.bank.device) @ ctx.bank
    carrier = torch.as_tensor(carrier_rows[k], dtype=ctx.bank.dtype, device=ctx.bank.device) @ ctx.bank
    op = ctx.op
    op.load(ctx.states[k])
    responses, scores = [], []
    for g, (images, labels) in enumerate(ctx.images):
        local = np.stack([response(op, images, v) for v in directions])
        responses.append(local @ local.T)
        old, full = op.evaluate(images), op.evaluate(images, ctx.bank[k])
        torch.testing.assert_close(full.double().softmax(-1).cpu(), ctx.states[k + 1]['probability'][g], atol=2e-5, rtol=0)
        baseline = dict(carrier=score(old, full, op.evaluate(images, carrier), labels), full=score(old, full, full, labels))
        historical = previous['groups'][g]['conditions']
        for name, key in [('carrier', 'carrier'), ('full', 'carrier_plus_1')]:
            np.testing.assert_allclose(baseline[name]['ce_gain'], historical[key]['ce_gain'], atol=2e-5)
        scores.append((old, full, baseline))
    groups = []
    euclidean = solve(parameter_gram)
    for fit in (0, 1):
        audit = 1 - fit
        images, labels = ctx.images[audit]
        old, full, baseline = scores[audit]
        outcomes = {}
        for name, (alpha, stats) in [('fisher', solve(responses[fit])), ('euclidean', euclidean)]:
            parallel = torch.as_tensor(alpha, dtype=directions.dtype, device=directions.device) @ directions[:-1]
            innovation = directions[-1] - parallel
            outcomes[name] = dict(alpha=alpha.tolist(), solver=stats,
                                  parameter=energy_split(parameter_gram, alpha),
                                  fit_function=energy_split(responses[fit], alpha),
                                  audit_function=energy_split(responses[audit], alpha),
                                  **replay(op, images, labels, old, full, carrier, parallel, innovation, baseline))
        groups.append(dict(fit_group=fit, audit_group=audit, baseline=baseline, projections=outcomes))
    return dict(start=ctx.states[k]['step'], end=ctx.states[k + 1]['step'],
                historical_windows=[[ctx.states[i]['step'], ctx.states[i + 1]['step']] for i in history],
                history_size=q, primary=q == HISTORY, parameter_gram=parameter_gram.tolist(),
                functional_grams=[r.tolist() for r in responses], groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.bank) as data:
        steps, gram = data['steps'], data['parameter_gram']
    ctx = replay_context(args.trace, args.queue, steps, gram)
    previous = read(args.replay)['windows']
    assert len(previous) == len(gram) - FIT_CARRIER
    carrier, residual = coefficients(gram)
    provenance = dict(bank_sha256=existing.sha(args.bank), replay_sha256=existing.sha(args.replay),
                      trace_sha256=existing.sha(args.trace / 'trace.json'), base_sha256=ctx.base_sha256,
                      expert_sha256=ctx.expert_sha256, image_indices=ctx.image_indices,
                      carrier_windows=FIT_CARRIER, target_history=HISTORY, ridge_relative=RELATIVE_RIDGE, betas=BETAS)
    rows = []
    for k in range(FIT_CARRIER + 1, len(gram)):
        prior = previous[k - FIT_CARRIER]
        assert (prior['start'], prior['end']) == (steps[k], steps[k + 1])
        result = measure_window(ctx, k, gram, carrier, residual, prior)
        rows.append(result)
        print(f'Window {steps[k]}–{steps[k + 1]}: Fisher and Euclidean replays, q={result["history_size"]}', flush=True)
    existing.write(args.output / 'results.json', dict(provenance=provenance, windows=rows))


if __name__ == '__main__':
    main()
