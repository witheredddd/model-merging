"""Require declared sources, rather than invent domain, budgets or branches."""
import json
from pathlib import Path


def load_protocol(path):
    p = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    for key in ['protocol_id', 'domain', 'ec', 'max_charts', 'branch',
                'population_transition', 'learn_at_ec_anchors']:
        if key not in p:
            raise ValueError(f'Missing explicit run protocol field: {key}')
    for obj in [p['domain'], p['ec'], p['branch'], p['population_transition']]:
        if not obj.get('source'):
            raise ValueError('All scheduling choices require a stated source')
    if p['domain'].get('kind') != 'original_radius_joint_motion':
        raise ValueError('This run requires the explicitly chosen joint-motion domain')
    for field in ['pop_size', 'seed', 'termination']:
        if field not in p['ec']:
            raise ValueError('Complete original EC budget configuration is required')
    t = p['ec']['termination']
    if t['kind'] not in ['n_eval', 'n_gen'] or int(t['value']) <= 0:
        raise ValueError('Unsupported explicit EC termination')
    if p['ec']['pop_size'] <= 0 or p['max_charts'] <= 0:
        raise ValueError('Invalid execution budget')
    if t['kind'] == 'n_eval' and p['ec']['pop_size'] > t['value']:
        raise ValueError('Population exceeds declared evaluation budget')
    if p['branch']['kind'] not in ['none', 'candidate_ids', 'objective_extreme', 'pareto_arc']:
        raise ValueError('Branch policy must be explicit, not mean-accuracy fallback')
    if p['max_charts'] > 1 and p['branch']['kind'] == 'none':
        raise ValueError('Multiple charts require an explicit branch policy')
    if p['population_transition']['kind'] not in ['fresh', 'reproject']:
        raise ValueError('Population transition is not specified')
    return p


def select_branch(front, branch, transition_index, tasks):
    if branch['kind'] == 'pareto_arc':
        import numpy as np
        unique = {}
        for row in front:
            unique.setdefault(tuple(row['ec_objectives']), row)
        ordered = sorted(unique.values(), key=lambda r: r['top1'][tasks.index('DTD')])
        F = np.array([r['ec_objectives'] for r in ordered])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(F, axis=0), axis=1))]
        return ordered[int(np.argmin(np.abs(arc - branch['fraction']*arc[-1])))]
    if branch['kind'] == 'candidate_ids':
        desired = branch['ids'][transition_index]
        return next(r for r in front if r['candidate_id'] == desired)
    if branch['kind'] == 'objective_extreme':
        task = branch['task']
        # Ties retain evaluation/archive order; no average objective or extra score.
        return min(front, key=lambda r: r['ec_objectives'][tasks.index(task)])
    raise ValueError('No branch selected')

