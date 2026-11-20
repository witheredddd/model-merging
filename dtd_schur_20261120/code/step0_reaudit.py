"""Audit the existing DTD Schur step-0 outputs without any new CG solves.

Read-only with respect to the original experiment.  New audit files are written
under <output>/reaudit_step0/. Requires the excluded binary .pt payloads to be
present on the original A100.  No claim of discovered physical supports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bootstrap import CORE, LEGACY  # noqa: E402
from carrier import TrainingPositions  # noqa: E402
from geometry import FullFunctionalGeometry  # noqa: E402
from math_core import class_weights, metric_stats  # noqa: E402
from reaudit_math import (common_scale, metric_retention, orthonormalize,
                           pooled_step, reduced_optimum, scale_to_radius)  # noqa: E402
import task_runtime  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf8')
    tmp.replace(path)


def gradient(context, point, batches, batch_size):
    total = sum(len(y) for _, y in batches)
    if total <= 0:
        raise ValueError('empty demand')
    result = torch.zeros_like(point)
    for x, y in batches:
        for k in range(0, len(y), batch_size):
            xx, yy = x[k:k + batch_size], y[k:k + batch_size]
            with torch.enable_grad():
                leaf = point.detach().requires_grad_(True)
                logits = context.predict(leaf, xx, 'DTD')
                loss = F.cross_entropy(logits, yy, reduction='sum') / total
                result += torch.autograd.grad(loss, leaf)[0].detach().double()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True, help='Existing step0_anchor0 output on the A100')
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--probes', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--functional-radii', type=float, nargs='+', default=[.0025, .01])
    a = p.parse_args()
    if a.batch_size <= 0 or not a.functional_radii or any(r <= 0 for r in a.functional_radii):
        raise ValueError('invalid batch size / radii')
    root = a.output.resolve()
    original = json.loads((root / 'CONFIG.json').read_text())
    quadratic = json.loads((root / 'QUADRATIC.json').read_text())
    demand_meta = json.loads((root / 'DEMANDS.json').read_text())
    selection = json.loads((root / 'SELECTION.json').read_text())
    mode_meta = json.loads((root / 'MODES.json').read_text())
    active_meta = json.loads((root / 'ACTIVE_DOMAIN.json').read_text())
    if original['anchor'] != 0 or demand_meta['probe'] != 0:
        raise ValueError('only original anchor0 / probe0 is supported')
    for input_path in (a.inventory, a.probes, a.trace / 'trace.json'):
        saved = original['inputs'].get(str(input_path))
        if saved is None or saved != digest(input_path):
            raise ValueError(f'original input identity does not match: {input_path}')
    for name in ('motions.pt', 'candidate_modes.pt', 'compensation.pt'):
        if not (root / name).is_file():
            raise FileNotFoundError(f'Need original excluded binary: {root / name}')
    lam, beta = float(quadratic['lambda_value']), float(quadratic['beta'])
    if lam <= 0 or beta < 0 or lam != original['lam'] or beta != original['beta']:
        raise ValueError('old quadratic configuration is inconsistent')

    audit_dir = root / 'reaudit_step0'
    audit_dir.mkdir(exist_ok=True)
    runtime = audit_dir / 'runtime'
    runtime.mkdir(exist_ok=True)
    shutil.copy2(a.probes, runtime / 'probes.json')
    inventory = [r for r in json.loads(a.inventory.read_text())['tasks'] if r['task'] in ('DTD', 'Cars')]
    old_task_names = task_runtime.TASKS
    task_runtime.TASKS = ('DTD', 'Cars')
    try:
        base = task_runtime.EightTaskContext(runtime, inventory)
    finally:
        task_runtime.TASKS = old_task_names
    for key, batches in list(base.batches.items()):
        base.batches[key] = [(x[k:k+a.batch_size],y[k:k+a.batch_size])
                              for x,y in batches for k in range(0,len(y),a.batch_size)]

    # Exactly the same stratified 2/3 split used by run_step0.py.
    xx = torch.cat([x for x,y in base.batches['DTD',1]])
    yy = torch.cat([y for x,y in base.batches['DTD',1]])
    split = [[], []]
    for cls in yy.unique().tolist():
        ids = (yy == cls).nonzero().flatten().tolist()
        for k, i in enumerate(ids):
            split[k % 2].append(i)
    if set(split[0]) & set(split[1]):
        raise ValueError('selection / final split overlaps')
    for probe, ids in zip((2,3), split):
        base.batches['DTD',probe] = [(xx[ids[k:k+a.batch_size]],yy[ids[k:k+a.batch_size]])
                                      for k in range(0,len(ids),a.batch_size)]
    del xx, yy
    pos = TrainingPositions(a.trace,base.keys,audit_dir)
    pos.verify_normal_recipe()
    idx = next(i for i,r in enumerate(pos.rows) if r['step']==0)
    point = pos.flat(idx).cuda().double()
    geo = FullFunctionalGeometry(base,point,0,precision='float64')
    ctx = geo.c
    active = torch.zeros(ctx.D,dtype=torch.bool,device='cuda')
    active_names = []
    at = 0
    for name,size in zip(ctx.keys,ctx.sizes):
        if name in active_meta['names']:
            active[at:at+size] = True
            active_names.append(name)
        at += size
    if active_names != active_meta['names'] or int(active.sum()) != active_meta['coordinates']:
        raise ValueError('original active domain no longer matches the model')
    def project(v):return v * active
    def get_grad(probe):return gradient(ctx,point,ctx.batches['DTD',probe],a.batch_size)
    def j(v):return geo.jvp('DTD',v)
    def h(v):return lam*v+beta*geo.vjp('DTD',j(v))

    # Stored motions are the four independent p0 task-regularized full optima.
    saved_steps = torch.load(root/'motions.pt',map_location='cpu',mmap=True,weights_only=True)
    steps = [x.to('cuda',dtype=torch.float64) for x in saved_steps['motions']]
    counts = [d['images'] for d in demand_meta['chunks']]
    mean_step,weights = pooled_step(steps, counts)
    full_g0 = steps[0]
    del steps
    g0 = get_grad(0)
    g2 = get_grad(2)  # the sole selection demand; probe3 is final evaluation only
    mean_grad_mode = orthonormalize([g0],project)
    mean_step_mode = orthonormalize([mean_step],project)
    # candidate_modes.pt also stores numpy eigenvalue arrays, which the weights_only
    # unpickler rejects. The file is produced by this experiment itself.
    raw_modes = torch.load(root/'candidate_modes.pt',map_location='cpu',mmap=True,weights_only=False)
    candidate = [v.to('cuda',dtype=torch.float64) for v in raw_modes['modes']]
    for k,u in enumerate(candidate):
        if float(torch.linalg.vector_norm(u-project(u))) > 1e-9:
            raise ValueError(f'stored candidate {k} leaks outside active domain')
    candidate = orthonormalize(candidate, project)
    if len(candidate) != mode_meta['kept_modes']:
        raise ValueError('stored candidate basis rank changed')

    # Because H is common across the four demands, their weighted mean optimum
    # must solve the pooled-CE quadratic. This is a strict numerical identity.
    pooled_residual = project(g0 + h(mean_step))
    pooled_ratio = float(torch.linalg.vector_norm(pooled_residual) / max(
        float(torch.linalg.vector_norm(project(g0))),np.finfo(float).tiny))
    if pooled_ratio > .003:
        raise RuntimeError(f'pooled-motion identity failed: {pooled_ratio:.5g}')
    active_g = project(g0)
    kappa_active = float(torch.dot(j(active_g),j(active_g)) / torch.dot(active_g,active_g))
    kappa_original = float(quadratic['kappa'])

    # Correct missing test: how much of the *full* selection-demand optimum can
    # each low-dimensional span achieve on its own? This is NOT Schur deletion.
    full_q = float(selection['rows'][0]['full_q'])
    groups = {
        'candidate_r1': candidate[:1],
        'candidate_r2': candidate[:2],
        'candidate_r3': candidate[:3],
        'mean_gradient_raw_r1': mean_grad_mode,
        'mean_regularized_r1': mean_step_mode,
        'mean_regularized_plus_candidate_r2': orthonormalize(mean_step_mode + candidate[:1],project),
        'mean_regularized_plus_candidate_r3': orthonormalize(mean_step_mode + candidate[:2],project),
    }
    selection_records = []
    finite_directions = {'mean_regularized_full':mean_step}
    for name,modes in groups.items():
        v2,q2,_ = reduced_optimum(modes,g2,j,lam,beta)
        v0,q0,_ = reduced_optimum(modes,g0,j,lam,beta)
        selection_records.append(dict(family=name,rank=len(modes),selection_q=q2,
                full_q=full_q,selection_retention=metric_retention(q2,full_q),
                construction_q=q0))
        finite_directions[name] = v0
    # The best single-DTD full-domain direction is always rank 1 for its own
    # construction demand. That is why selection on a different demand matters.
    report = dict(
        numerical=dict(pooled_stationarity_ratio=pooled_ratio,
            original_kappa=kappa_original,active_gradient_kappa=kappa_active,
            lambda_value=lam,beta=beta),
        construction_counts=counts,construction_weights=weights,
        selection_rows=selection_records,
        old_rank=selection['selected_rank'],
        interpretation='retention is restricted-domain sufficiency, not Schur deletion; rank is not selected from probe3',
        full_q_provenance='SELECTION.json full optimum, probe2 gradient with fixed probe0 H',
        old_random_control='dense isotropic random directions, not random coordinate supports',
        no_new_CG=True,
    )
    write_json(audit_dir/'sufficiency.json',report)

    # Physical finite interventions: mode construction is frozen before any
    # probe3 evaluation.  All directions receive exactly the same radius in
    # the individually normalized experiment; no objective-gap interpretation.
    comp = torch.load(root/'compensation.pt',map_location='cpu',mmap=True,weights_only=True)
    rest_g0 = comp['complement_at_zero'].to('cuda',dtype=torch.float64)
    half_joint = .5*(full_g0 + rest_g0)
    baseline = {probe:ctx.baseline('DTD',probe,point)[0] for probe in (2,3)}
    rows = []
    for probe in (2,3):
        base,b_labels = ctx.baseline('DTD',probe,point)
        mm = metric_stats(base.numpy(),b_labels.numpy(),class_weights(base.shape[1],'mean'))
        rows.append(dict(protocol='actual_anchor',family='baseline',probe=probe,
            parameter_norm=0.,functional_radius=None,
            CE=mm['CE'],Top1=mm['Top1'],delta_CE=0.,delta_Top1=0.,
            paired_CI_lower=0.,paired_CI_upper=0.))

    for f_radius in a.functional_radii:
        physical = f_radius / np.sqrt(kappa_original)
        physical_comp = [full_g0,rest_g0,half_joint]
        common, tau = common_scale(physical_comp,physical)
        arms = []
        for name,v in finite_directions.items():
            if float(torch.linalg.vector_norm(v)) > 0:
                arms.append(('equal_radius',name,scale_to_radius(v,physical),None))
        for name,v in zip(('g0_full','g0_restricted','g0_half_joint'),physical_comp):
            arms.append(('equal_radius',name,scale_to_radius(v,physical),None))
        for name,v in zip(('g0_full','g0_restricted','g0_half_joint'),common):
            arms.append(('common_tau',name,v,tau))
        for protocol,name,delta,tau_used in arms:
            theta = point + delta
            for probe in (2,3):
                logits,labels = ctx.baseline('DTD',probe,theta)
                base_logits = baseline[probe]
                met = metric_stats(logits.numpy(),labels.numpy(),
                    class_weights(logits.shape[1],'mean'),base_logits.numpy())
                rows.append(dict(protocol=protocol,family=name,probe=probe,
                    functional_radius=f_radius,parameter_norm=float(torch.linalg.vector_norm(delta)),
                    common_tau=tau_used,CE=met['CE'],Top1=met['Top1'],
                    delta_CE=met['delta_CE'],delta_Top1=met['delta_Top1'],
                    paired_CI_lower=met['paired_weighted_CE_approx95'][0],
                    paired_CI_upper=met['paired_weighted_CE_approx95'][1]))
            del theta,delta
    path = audit_dir/'finite_small_steps.csv'
    fields = list(dict.fromkeys(k for x in rows for k in x))
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write_json(audit_dir/'RUN_AUDIT.json',dict(
        source='Original step0 raw vectors, no new Schur/CG solve',
        all_p3_cases_predeclared=True,
        p3_not_used_for_model_or_rank_selection=True,
        original_config_sha256=digest(root/'CONFIG.json'),
        parameter_domain='48 Transformer matrices',
        radii_functional=a.functional_radii,
        original_beta_used=True,
        active_kappa_audit_only=True,
        changed_original_files=False,
        finite_rows=len(rows),
        warnings=['Independent normalization is a direction comparison, not a common-quadratic Schur gap',
                  'Pooled task motion is rank one by construction; only independent demands can establish reusability',
                  'The held-out data are training-side development samples, not official test'],
        ))
    print('AUDIT COMPLETE',audit_dir,'\n'+json.dumps({
        'pooled_stationarity_ratio':pooled_ratio,
        'kappa_active':kappa_active,
        'radii':a.functional_radii,'finite_rows':len(rows)},indent=2))

if __name__ == '__main__':main()
