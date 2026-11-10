"""Summarize the paired DTD allocation experiment from its two results files."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def correlation(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def summarize(arm):
    records = [r for r in arm['records'] if 'applied_sources' in r]
    sources = [r['applied_sources'] for r in records]
    samples = [(r['step'], s) for r,s in zip(records,sources) if 'ce_gains' in s]
    independent = [s for _,s in samples if s['independent_of_selection']]
    norms = np.array([s['norms'] for s in sources])
    ce_gains = np.array([s['ce_gains'] for s in independent]).mean(axis=0)
    gram = np.array([s['fisher_gram'] for s in independent]).mean(axis=0)
    # These signed terms split the response energy, including the cross terms.
    fisher_shares = gram.sum(axis=1)/gram.sum() if gram.sum() > 0 else np.zeros(3)
    audit = arm['learned']['coordinate_audit']['rows']
    expected = [[r['scores'][k]['fisher_prediction'] for r in audit] for k in (0,1)]
    actual = [[r['scores'][k]['actual_response'] for r in audit] for k in (0,1)]
    result = dict(validation=arm['validation'],historical_test=arm['test'],
                  components=arm['learned']['components'],dimension=arm['learned']['coordinate_dimension'],
                  post_warmup_steps=len(records),source_norms_sum=norms.sum(axis=0).tolist(),
                  independent_sample_steps=[step for step,s in samples if s['independent_of_selection']],
                  mean_sequential_ce_gain=ce_gains.tolist(),mean_source_fisher_gram=gram.tolist(),
                  signed_source_fisher_shares=fisher_shares.tolist(),
                  terminal_predicted_vs_actual=[correlation(p,a) for p,a in zip(expected,actual)],
                  terminal_actual_across_image_groups=correlation(*actual),
                  birth_steps=[r['step'] for r in records if r.get('birth') and r['birth']['added']])
    allocations = [r['allocation'] for r in records if 'allocation' in r]
    if allocations:
        result['beta_counts'] = dict(Counter(str(a['beta']) for a in allocations))
        result['fallback_counts'] = dict(Counter(a['fallback'] for a in allocations if a['fallback']))
        result['evolved_steps'] = sum(a['evolved'] for a in allocations)
        result['legacy_birth_proposal_steps'] = [r['step'] for r in records
                                                 if r.get('birth') and r['birth']['proposed_added']]
        result['trust_rejections'] = {key:sum(a['rejections'][key] for a in allocations)
                                      for key in allocations[0]['rejections']}
        frozen = [next((row for row in a['candidates'] if row['beta']==0 and not row['evolved']),None)
                  for a in allocations]
        result['frozen_pure_candidates'] = sum(row is not None for row in frozen)
        result['frozen_pure_improves'] = sum(row is not None and row['loss']<a['ce_zero']
                                             for a,row in zip(allocations,frozen))
        result['frozen_pure_reaches_safety_line'] = sum(
            row is not None and a['ce_limit'] is not None and row['loss']<=a['ce_limit']
            for a,row in zip(allocations,frozen))
        ratios = [max(0.,a['ce_zero']-row['loss'])/(a['ce_zero']-a['ce_adamw'])
                  for a,row in zip(allocations,frozen) if row is not None and a['ce_adamw']<a['ce_zero']]
        result['frozen_pure_vs_positive_adamw_gain'] = dict(
            n=len(ratios),quartiles=np.quantile(ratios,[.25,.5,.75]).tolist() if ratios else [])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('allocation',type=Path)
    args = parser.parse_args()
    a = json.loads(args.baseline.read_text(encoding='utf-8'))['arms']['adaptive_dynamics']
    b = json.loads(args.allocation.read_text(encoding='utf-8'))['arms']['adaptive_allocation']
    for key in ('steps','seed','warmup','train_n','validation_split','validation_n','birth_penalty',
                'max_components','base_sha256','head_sha256'):
        if a['provenance'][key] != b['provenance'][key]:
            raise ValueError(f'Incompatible A/B provenance for {key}')
    print(json.dumps(dict(source_order=['coordinates','axis_evolution','high_dimensional'],
                          A=summarize(a),B=summarize(b)),indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
