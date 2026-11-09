"""Verify and package the DTD rolling innovation training comparison."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def inspect(folder, value, test):
    p = value['provenance']
    assert value['checkpoint_sha256'] == test['checkpoint_sha256'] == digest(folder / 'epoch08.pt')
    assert test['split'] == 'historical_dtd_test' and test['n'] == 1880
    rows = value['records']
    assert len(rows) == 216 and [r['step'] for r in rows] == list(range(1, 217))
    assert [r['epoch'] for r in value['epochs']] == list(range(1, 9))
    ends = []
    for row in rows:
        step = row['step']
        boundary = step % 25 == 0 or step % 27 == 0
        assert row['boundary'] == boundary and row['history_end_steps'] == ends[-5:]
        assert math.isfinite(row['training_ce'])
        expected = 1.
        if p['arm'] != 'adamw' and len(ends) >= 5:
            checks = row['checks']
            assert [r['beta'] for r in checks] == [.875, .75]
            for r in checks:
                assert r['passed'] == all(x > p['ce_margin'] for x in r['lower_95'])
            expected = next((r['beta'] for r in checks if r['passed']), 1.)
        else:
            assert 'checks' not in row
        assert row['beta'] == expected
        if row['beta'] < 1.:
            assert 'against_scalar' in row
        if boundary:
            ends.append(step)
    assert len(ends) == 16 and ends[4] == 75
    assert len(set(i for group in p['probe_groups'] for i in group)) == 96
    assert all(len(g) == 32 for g in p['probe_groups'])
    on = [r for r in rows if r['beta'] < 1.]
    gate_diagnostics = {}
    for beta in (.875, .75):
        trials = [check for r in rows for check in r.get('checks', []) if check['beta'] == beta]
        if trials:
            gate_diagnostics[str(beta)] = dict(
                evaluated=len(trials),
                both_mean_positive=sum(all(x > 0 for x in c['mean_gain']) for c in trials),
                mean_sign_disagreement=sum(sum(x > 0 for x in c['mean_gain']) == 1 for c in trials),
                passed=sum(c['passed'] for c in trials),
                best_min_lower_bound=max(min(c['lower_95']) for c in trials),
            )
    boundary_rows = [r for r in rows if 'executed_function_explained' in r]
    candidate = [r['candidate_function_explained'] for r in boundary_rows]
    accepted = [r['executed_function_explained'] for r in boundary_rows]
    final = value['epochs'][-1]
    scalar = [r['against_scalar'] for r in on]
    return dict(seed=p['seed'], arm=p['arm'], final_validation=dict(accuracy=final['accuracy'], ce=final['ce']),
                historical_test=dict(accuracy=test['accuracy'], ce=test['ce'], n=test['n']),
                historical_test_target_met=test['accuracy'] >= .8,
                beta_counts={str(b): sum(r['beta'] == b for r in rows) for b in (1., .875, .75)},
                gate_diagnostics=gate_diagnostics,
                gated_steps_by_epoch=[sum(r['epoch'] == e for r in on) for e in range(1, 9)],
                first_intervention=on[0]['step'] if on else None,
                candidate_function_explained_median=median(candidate),
                executed_function_explained_median=median(accepted),
                first_three_executed_function_explained=median(accepted[:3]),
                last_three_executed_function_explained=median(accepted[-3:]),
                functional_boundary_count=len(boundary_rows),
                norm_matched_scalar_strictly_beaten=sum(r['passed'] for r in scalar),
                norm_matched_scalar_positive_both=sum(all(x > 0 for x in r['mean_gain']) for r in scalar),
                norm_matched_scalar_median_mean_gains=[median(r['mean_gain'][i] for r in scalar) for i in (0, 1)] if scalar else None,
                sum_epoch_seconds=sum(e['seconds'] for e in value['epochs']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    folders = read(args.run / 'complete.json')['arms']
    assert folders == read(args.run / 'training_complete.json')['arms']
    records, overview = [], []
    for name in folders:
        folder = args.run / name
        value, test = read(folder / 'complete.json'), read(folder / 'historical_test.json')
        overview.append(inspect(folder, value, test))
        records.append(dict(training=value, historical_test=test))
    seeds = sorted({r['seed'] for r in overview})
    assert seeds in ([42], [42, 43, 44])
    assert records[0]['training']['exact_historical_baseline']
    assert all(r['training']['provenance']['probe_groups'] == records[0]['training']['provenance']['probe_groups'] for r in records)
    comparisons = []
    for seed in seeds:
        rows = [r for r in records if r['training']['provenance']['seed'] == seed]
        assert [r['training']['provenance']['arm'] for r in rows] == ['adamw', 'fisher', 'euclidean']
        losses = [tuple(s['training_ce'] for s in r['training']['records'][:75]) for r in rows]
        assert losses[0] == losses[1] == losses[2]
        a, b, c = [r for r in overview if r['seed'] == seed]
        import torch

        prefix = args.run / f'seed{seed}'
        anchor = torch.load(prefix / 'adamw/epoch08.pt', weights_only=True, mmap=True, map_location='cpu')['visual']
        for gate_arm in (b, c):
            if gate_arm['first_intervention'] is None:
                candidate = torch.load(prefix / gate_arm['arm'] / 'epoch08.pt', weights_only=True, mmap=True,
                                       map_location='cpu')['visual']
                gate_arm['no_gate_endpoint_tensor_exact_to_adamw'] = all(torch.equal(v, candidate[k]) for k, v in anchor.items())
                assert gate_arm['no_gate_endpoint_tensor_exact_to_adamw']
        comparisons.append(dict(seed=seed, fisher_minus_adamw_test=b['historical_test']['accuracy'] - a['historical_test']['accuracy'],
                                fisher_minus_euclidean_test=b['historical_test']['accuracy'] - c['historical_test']['accuracy'],
                                fisher_minus_adamw_val=b['final_validation']['accuracy'] - a['final_validation']['accuracy'],
                                fisher_minus_euclidean_val=b['final_validation']['accuracy'] - c['final_validation']['accuracy']))
    seed42 = next(r for r in overview if r['seed'] == 42 and r['arm'] == 'fisher')
    assert (seeds == [42, 43, 44]) == (seed42['first_intervention'] is not None)
    result = dict(seeds=seeds, seed42_adamw_tensor_exact=True, all_warmup_training_losses_identical=True,
                  arms=overview, paired=comparisons)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    (args.output / 'training_records.json').write_text(json.dumps(records, separators=(',', ':'), allow_nan=False),
                                                     encoding='utf-8')
    print(f'Verified {len(overview)} training arms and historical test evaluations')


if __name__ == '__main__':
    main()
