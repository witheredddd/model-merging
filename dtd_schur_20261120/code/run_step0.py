"""Step 0: one real DTD anchor, the 48 Transformer weight matrices only.

Chain: real CE demands -> full-domain regularized motions -> H-metric candidate
motion space -> Schur conditional deletion -> fixed-driver compensation ->
finite CE on held-out images.

Nothing here discovers physical parameter supports. The candidate space is a
subspace of full-coordinate movements and may cross all 48 matrices.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bootstrap import CORE, LEGACY, R3, SPACE, FLOW  # noqa: E402
from carrier import TrainingPositions  # noqa: E402
from io_state import atomic_json, digest  # noqa: E402
from geometry import FullFunctionalGeometry  # noqa: E402
import task_runtime  # noqa: E402
from math_core import class_weights, metric_stats  # noqa: E402
from conditional_schur_core import (  # noqa: E402
    mask_projector, orthogonal_complement_projector,
    construct_metric_motion_modes, hessian_action, quadratic_objective)
from schur_resumable import (solve_resumable, project_onto, mode_diagnostics,  # noqa: E402
                             tensor_digest)

ACTIVE_SUFFIX = ('attn.in_proj_weight', 'attn.out_proj.weight',
                 'mlp.c_fc.weight', 'mlp.c_proj.weight')
EXPECTED_ACTIVE_DIM = 84934656
EXPECTED_ACTIVE_TENSORS = 48

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--inventory', type=Path, required=True)
p.add_argument('--probes', type=Path, required=True)
p.add_argument('--trace', type=Path, required=True)
p.add_argument('--flow-source', type=Path, required=True)
p.add_argument('--anchor', type=int, default=0)
p.add_argument('--demands', type=int, default=4)
p.add_argument('--select-demands', type=int, default=1)
p.add_argument('--max-rank', type=int, default=3)
p.add_argument('--alpha', type=float, default=1.)
p.add_argument('--lam', type=float, default=1.)
p.add_argument('--drivers', type=float, nargs='+', default=[.25, .5, 1.])
p.add_argument('--batch-size', type=int, default=16)
p.add_argument('--rtol', type=float, default=1e-3)
p.add_argument('--maxiter', type=int, default=64)
a = p.parse_args()

out = a.output
out.mkdir(parents=True, exist_ok=True)
(out / 'solves').mkdir(exist_ok=True)
(out / 'vectors').mkdir(exist_ok=True)
STARTED = time.monotonic()
GEO = None


def status(stage, **extra):
    payload = dict(stage=stage, anchor=a.anchor, elapsed_seconds=time.monotonic() - STARTED, **extra)
    if GEO is not None:
        payload['operator'] = GEO.operator.audit()
    atomic_json(out / 'status.json', payload)
    print(json.dumps({k: v for k, v in payload.items() if k != 'operator'}), flush=True)


def save_vector(name, vector):
    path = out / 'vectors' / (name + '.pt')
    tmp = path.with_suffix('.tmp')
    torch.save(dict(vector=vector.detach().cpu().double()), tmp)
    h = digest(tmp)
    final = path.with_name(path.stem + '_' + h[:12] + '.pt')
    if final.exists():
        tmp.unlink()
    else:
        tmp.replace(final)
    return dict(file=str(final.relative_to(out)), sha256=h, dimension=vector.numel())


# ---------------------------------------------------------------- configuration
status('CONFIGURING')

runtime = out / 'runtime'
runtime.mkdir(exist_ok=True)
shutil.copy2(a.probes, runtime / 'probes.json')

inventory = [row for row in json.loads(a.inventory.read_text())['tasks']
             if row['task'] in ('DTD', 'Cars')]
default_tasks = task_runtime.TASKS
task_runtime.TASKS = ('DTD', 'Cars')
try:
    base_context = task_runtime.EightTaskContext(runtime, inventory)
finally:
    task_runtime.TASKS = default_tasks
for key, batches in list(base_context.batches.items()):
    base_context.batches[key] = [(x[i:i + a.batch_size], y[i:i + a.batch_size])
                                 for x, y in batches for i in range(0, len(y), a.batch_size)]

# Disjoint selection (probe 2) and final evaluation (probe 3) pools from probe 1.
_xt = torch.cat([x for x, y in base_context.batches['DTD', 1]])
_yt = torch.cat([y for x, y in base_context.batches['DTD', 1]])
_split = [[], []]
for cls in _yt.unique().tolist():
    ids = (_yt == cls).nonzero().flatten().tolist()
    for k, i in enumerate(ids):
        _split[k % 2].append(i)
for probe, ids in zip((2, 3), _split):
    base_context.batches['DTD', probe] = [
        (_xt[ids[i:i + a.batch_size]], _yt[ids[i:i + a.batch_size]])
        for i in range(0, len(ids), a.batch_size)]
if set(_split[0]) & set(_split[1]):
    raise ValueError('Selection and evaluation images overlap')
del _xt, _yt

positions = TrainingPositions(a.trace, base_context.keys, out)
provenance = positions.verify_normal_recipe()
index = next(i for i, row in enumerate(positions.rows) if row['step'] == a.anchor)
point = positions.flat(index).cuda().double()

GEO = FullFunctionalGeometry(base_context, point, 0, precision='float64')
CTX = GEO.c
N_CLASSES = int(CTX.baseline('DTD', 0, point)[0].shape[1])
status('ANCHOR_READY', step=a.anchor, classes=N_CLASSES)

# ------------------------------------------------------- active coordinate mask
active = torch.zeros(CTX.D, dtype=torch.bool, device='cuda')
offset = 0
active_names = []
for name, size in zip(CTX.keys, CTX.sizes):
    if any(name.endswith(suffix) for suffix in ACTIVE_SUFFIX):
        active[offset:offset + size] = True
        active_names.append(name)
    offset += size
if int(active.sum()) != EXPECTED_ACTIVE_DIM or len(active_names) != EXPECTED_ACTIVE_TENSORS:
    raise ValueError(f'Active domain mismatch: {int(active.sum())} values, '
                     f'{len(active_names)} tensors')
P_D = mask_projector(active)
MASK_ID = 'mask:' + hashlib.sha256(
    np.ascontiguousarray(active.cpu().numpy()).tobytes()).hexdigest()
POINT_ID = tensor_digest(point)
MODE_IDS = {}


def subspace_id(family, modes):
    key = (family, len(modes))
    if key not in MODE_IDS:
        MODE_IDS[key] = f'{family}[{len(modes)}]:' + tensor_digest(list(modes))
    return f'{MASK_ID}|{MODE_IDS[key]}'


active_report = dict(tensors=len(active_names), coordinates=int(active.sum()),
                     total_coordinates=int(CTX.D), names=active_names,
                     selection_rule='Transformer attention/MLP weight matrices only; '
                                    'no native-region ranking beyond this declared family')
atomic_json(out / 'ACTIVE_DOMAIN.json', active_report)
status('ACTIVE_DOMAIN_READY', coordinates=int(active.sum()))

# ------------------------------------------------------------ numerical gates
status('ADJOINTNESS_CHECK')
gen = torch.Generator(device='cuda').manual_seed(7)
direction = torch.zeros(CTX.D, dtype=torch.float64, device='cuda')
direction.normal_(0., 1., generator=gen)
direction /= torch.linalg.vector_norm(direction)
_direction_output = GEO.jvp('DTD', direction)
_weight = torch.zeros_like(_direction_output)
_weight.normal_(0., 1., generator=gen)
lhs = float(torch.dot(_direction_output, _weight))
rhs = float(torch.dot(direction, GEO.vjp('DTD', _weight)))
adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(float).tiny)
del _direction_output, _weight, direction
torch.cuda.empty_cache()
atomic_json(out / 'NUMERICS.json', dict(
    adjointness=dict(jvp_dot=lhs, vjp_dot=rhs, relative_error=adjoint_error,
                     passed=bool(adjoint_error < 1e-10)),
    precision='float64 model, inputs, parameters, logits and embedding',
    probe='constructor probe 0 (DTD construction pool) for every JVP/VJP in this run'))
if adjoint_error >= 1e-10:
    raise ValueError('Adjointness gate failed: ' + str(adjoint_error))
status('ADJOINTNESS_OK', relative_error=adjoint_error)

# ------------------------------------------------------------------- demands
status('DEMANDS')


def stratified_chunks(task, probe, parts):
    xs = torch.cat([x for x, _ in CTX.batches[task, probe]])
    ys = torch.cat([y for _, y in CTX.batches[task, probe]])
    ids = [[] for _ in range(parts)]
    for cls in ys.unique().tolist():
        members = (ys == cls).nonzero().flatten().tolist()
        for k, i in enumerate(members):
            ids[k % parts].append(int(i))
    return [[(xs[i], ys[i])] for i in ids]


def ce_gradient(batches, task):
    """Mean CE gradient over these images, reduced in fixed-size sub-batches."""
    total = sum(len(y) for _, y in batches)
    gradient = torch.zeros(CTX.D, dtype=torch.float64, device='cuda')
    for x, y in batches:
        for start in range(0, len(y), a.batch_size):
            xs = x[start:start + a.batch_size]
            ys = y[start:start + a.batch_size]
            with torch.enable_grad():
                leaf = point.detach().requires_grad_(True)
                logits = CTX.predict(leaf, xs, task)
                loss = F.cross_entropy(logits, ys, reduction='sum') / total
                gradient += torch.autograd.grad(loss, leaf)[0].detach().double()
    return gradient


construct_chunks = stratified_chunks('DTD', 0, a.demands)
gradients = []
chunk_meta = []
for j, batches in enumerate(construct_chunks):
    g = ce_gradient(batches, 'DTD')
    gradients.append(g)
    chunk_meta.append(dict(index=j, images=sum(len(y) for _, y in batches),
                           gradient_norm=float(torch.linalg.vector_norm(g)),
                           active_gradient_norm=float(torch.linalg.vector_norm(P_D(g)))))
cosines = [[float(F.cosine_similarity(gradients[i], gradients[j], dim=0))
            for j in range(len(gradients))] for i in range(len(gradients))]
atomic_json(out / 'DEMANDS.json', dict(
    chunks=chunk_meta, cosine_between_demands=cosines, probe=0,
    role='independent data-batch estimates of one DTD task, not distinct semantic functions'))
status('DEMANDS_READY', demands=len(gradients))

# ------------------------------------------------------------------ quadratic
full_pool = [(torch.cat([x for x, _ in CTX.batches['DTD', 0]]),
              torch.cat([y for _, y in CTX.batches['DTD', 0]]))]
mean_gradient = ce_gradient(full_pool, 'DTD')
Jg = GEO.jvp('DTD', mean_gradient)
kappa = float(torch.dot(Jg, Jg)) / max(float(torch.dot(mean_gradient, mean_gradient)),
                                       np.finfo(float).tiny)
beta = a.alpha / kappa
del Jg
torch.cuda.empty_cache()
atomic_json(out / 'QUADRATIC.json', dict(
    lambda_value=a.lam, alpha=a.alpha, kappa=kappa, beta=beta,
    kappa_definition='||J g||^2 / ||g||^2 on the DTD construction probe',
    objective='g^T v + lambda/2 ||v||^2 + beta/2 ||Jv||^2',
    hessian='lambda I + beta J^T J, applied matrix-free',
    not_the_true_network_CE_hessian=True))
status('QUADRATIC_READY', kappa=kappa, beta=beta)


def solve(label, g, projector, note=None, projector_id=None):
    # A checkpoint is reusable only under the exact system that produced it.
    identity = dict(task='DTD', probe=0, anchor=a.anchor, lam=a.lam, beta=beta,
                    gradient=tensor_digest(g), subspace=projector_id or MASK_ID,
                    parameter_point=POINT_ID,
                    output_coordinates=GEO.output_coordinate_ids['DTD'],
                    dimension=int(g.numel()))
    result, report = solve_resumable(
        GEO, g, projector, lam=a.lam, beta=beta, task='DTD',
        rtol=a.rtol, maxiter=a.maxiter, checkpoint=out / 'solves' / (label + '.npz'),
        task_name=label, audit=GEO.operator.audit, identity=identity,
        progress=lambda extra: status('SOLVING', case=label, **extra))
    report['note'] = note
    atomic_json(out / 'solves' / (label + '.json'), report)
    return result, report


# ------------------------------------------------------- full-domain motions
status('FULL_MOTIONS')
motions = []
for j, g in enumerate(gradients):
    result, report = solve(f'g{j}_full', g, P_D,
                           note='unrestricted motion inside the 48 matrices')
    motions.append(result)
    record = dict(index=j, q=result.q,
                  norm=float(torch.linalg.vector_norm(result.vector)),
                  stationarity_ratio=report['relative_stationarity'],
                  iterations=report['iterations'], **save_vector(f'motion_g{j}', result.vector))
    print(json.dumps(record), flush=True)
torch.save(dict(motions=[m.vector.detach().cpu() for m in motions],
                quality=[m.q for m in motions], beta=beta, kappa=kappa,
                lam=a.lam, anchor=a.anchor, probe=0), out / 'motions.pt')

# ------------------------------------------------- candidate motion subspaces
status('MOTION_MODES')
candidate = construct_metric_motion_modes(GEO, [m.vector for m in motions],
                                          lam=a.lam, beta=beta, task='DTD')
modes = candidate.modes[:min(len(candidate.modes), a.max_rank)]
mode_report = mode_diagnostics(modes, P_D)
atomic_json(out / 'MODES.json', dict(
    h_gram_eigenvalues=candidate.h_gram_eigenvalues.tolist(),
    cumulative_fraction=candidate.cumulative_fraction.tolist(),
    kept_modes=len(modes), rank_bound=len(motions),
    rank_is_bounded_by_the_number_of_construction_demands=True, **mode_report))
torch.save(dict(modes=[m.detach().cpu() for m in modes],
                eigenvalues=candidate.h_gram_eigenvalues,
                cumulative_fraction=candidate.cumulative_fraction),
           out / 'candidate_modes.pt')
status('MODES_READY', modes=len(modes),
       max_orthonormality_error=mode_report['max_orthonormality_error'])

motions = motions[:1]
gradients = gradients[:1]
torch.cuda.empty_cache()

# ------------------------------------------------------- reference subspaces


def orthonormalize(vectors, count):
    """Euclidean-orthonormal modes that live strictly inside the active domain.

    P_N(v) = P_D(v) - sum_i u_i (u_i . P_D(v)) is an orthogonal projector only
    when P_D u_i = u_i. A raw gradient or a Fischer direction carries a large
    component outside the 48 matrices, and using it unprojected makes P_N
    non-idempotent: the restricted quadratic then has no minimiser and the CG
    residual never settles.
    """
    basis = []
    for vector in vectors[:count]:
        b = P_D(vector)
        for _ in range(2):
            for u in basis:
                b = b - torch.dot(u, b) * u
        b = P_D(b)
        length = float(torch.linalg.vector_norm(b))
        if length > 1e-13:
            basis.append(b / length)
    return basis


def require_in_domain(modes, label):
    for index, mode in enumerate(modes):
        leak = float(torch.linalg.vector_norm(mode - P_D(mode)))
        if leak > 1e-9 * max(float(torch.linalg.vector_norm(mode)), 1.):
            raise ValueError(f'{label} mode {index} leaks outside the active domain: {leak}')


current = json.loads((a.flow_source / 'CURRENT.json').read_text())
raw = (a.flow_source / current['state']).read_bytes()
if hashlib.sha256(raw).hexdigest() != current['sha256']:
    raise ValueError('Flow state hash mismatch')
flow_state = json.loads(raw)
first_transition = json.loads((a.flow_source / flow_state['transitions'][0]).read_text())
source_point = torch.load(a.flow_source / first_transition['theta_k']['file'],
                          map_location='cpu', mmap=True, weights_only=True)['vector']
if not torch.equal(source_point.reshape(-1).double(), point.detach().cpu().double().reshape(-1)):
    raise ValueError('Fisher baseline was built at a different parameter anchor')
fisher_vectors = []
for ref in first_transition['basis_k']:
    path = a.flow_source / ref['file']
    if digest(path) != ref['sha256']:
        raise ValueError('Fisher direction checksum changed')
    fisher_vectors.append(torch.load(path, map_location='cuda', mmap=True,
                                     weights_only=True)['vector'].double())
fisher_modes = orthonormalize(fisher_vectors, 2)
mean_mode = orthonormalize([mean_gradient], 1)
rng = np.random.default_rng(20261120)
random_candidates = []
for _ in range(2):
    v = torch.zeros(CTX.D, dtype=torch.float64, device='cuda')
    v.normal_(0., 1., generator=torch.Generator(device='cuda').manual_seed(
        int(rng.integers(1 << 30))))
    random_candidates.append(P_D(v))
random_modes = orthonormalize(random_candidates, 2)
del fisher_vectors, source_point, random_candidates
torch.cuda.empty_cache()

require_in_domain(modes, 'candidate')
require_in_domain(mean_mode, 'mean_gradient_1d')
require_in_domain(fisher_modes, 'existing_fisher')
require_in_domain(random_modes, 'random_coordinates')

baselines = {'mean_gradient_1d': mean_mode,
             'existing_fisher': fisher_modes,
             'random_coordinates': random_modes}
atomic_json(out / 'BASELINES.json', {
    'mean_gradient_1d': dict(modes=len(mean_mode),
                             source='mean CE gradient on the DTD construction probe'),
    'existing_fisher': dict(modes=len(fisher_modes),
                            source=str(a.flow_source / flow_state['transitions'][0])),
    'random_coordinates': dict(modes=len(random_modes), seed=20261120,
                               note='active-domain random directions; a null control'),
    'all_built_without_selection_or_evaluation_images': True})

# ------------------------------------- rank selection on independent demands
status('INDEPENDENT_SELECTION')
select_chunks = stratified_chunks('DTD', 2, a.select_demands)
selection_rows = []
for k, batches in enumerate(select_chunks):
    g_select = ce_gradient(batches, 'DTD')
    full_select, _ = solve(f'select{k}_full', g_select, P_D,
                           note='independent demand, unrestricted')
    families = {'candidate': modes}
    families.update(baselines)
    for family, pool in families.items():
        upper = min(len(pool), 2 if family in ('existing_fisher', 'random_coordinates')
                    else a.max_rank)
        for r in range(1, upper + 1):
            P_N = orthogonal_complement_projector(P_D, pool[:r])
            restricted, report = solve(f'select{k}_{family}_r{r}', g_select, P_N,
                                       note=f'forbid {r} {family} direction(s)',
                                       projector_id=subspace_id(family, pool[:r]))
            gap = restricted.q - full_select.q
            difference = restricted.vector - full_select.vector
            schur = .5 * float(torch.dot(difference, hessian_action(GEO, difference,
                                                                   a.lam, beta, 'DTD')))
            selection_rows.append(dict(
                select_demand=k, images=sum(len(y) for _, y in batches), family=family, rank=r,
                restricted_q=restricted.q, full_q=full_select.q,
                proxy_deletion=gap, schur_energy=schur, schur_identity_error=gap - schur,
                stationarity_ratio=report['relative_stationarity'],
                converged=bool(report['converged']),
                iterations=report['iterations']))
            print(json.dumps(selection_rows[-1]), flush=True)
    del g_select
    torch.cuda.empty_cache()

candidate_rows = [row for row in selection_rows if row['family'] == 'candidate']
if not candidate_rows:
    raise RuntimeError('No candidate-rank results')
tiny = np.finfo(float).tiny
for row in selection_rows:
    row['normalized_deletion'] = row['proxy_deletion'] / max(abs(row['full_q']), tiny)

# Proxy deletion grows monotonically with the size of the forbidden subspace, so
# it can never select a rank by itself. Rank is chosen where the candidate space
# is most distinctive against dimension-matched alternatives built from
# construction-only data. Only ranks at which EVERY baseline family is present
# are comparable: otherwise the comparison set changes with the rank and the
# advantage stops meaning anything.
family_at_rank = {}
for row in selection_rows:
    family_at_rank.setdefault(row['rank'], set()).add(row['family'])
baseline_families = {row['family'] for row in selection_rows if row['family'] != 'candidate'}
complete_ranks = sorted(rank for rank, families in family_at_rank.items()
                        if baseline_families <= families)
advantages = []
for rank in sorted(family_at_rank):
    own = next(row for row in candidate_rows if row['rank'] == rank)['proxy_deletion']
    matched = [row['proxy_deletion'] for row in selection_rows
               if row['family'] != 'candidate' and row['rank'] == rank]
    advantages.append(dict(rank=rank, candidate_deletion=own,
                           best_matched_baseline=max(matched) if matched else None,
                           advantage=(own - max(matched)) if rank in complete_ranks else None,
                           comparable=rank in complete_ranks))
scored = [row for row in advantages if row['comparable']]
selected_rank = int(max(scored, key=lambda row: row['advantage'])['rank']) if scored \
    else int(min(row['rank'] for row in candidate_rows))
atomic_json(out / 'SELECTION.json', dict(
    rows=selection_rows, rank_advantages=advantages,
    comparable_ranks=complete_ranks, selected_rank=selected_rank,
    rule='largest proxy-deletion advantage over dimension-matched baselines at ranks '
         'where every baseline family exists; deletion itself is monotone in rank '
         'and never selects',
    selection_probe=2, evaluation_probe=3, official_test_opened=False,
    baselines_rank_matched_on_the_same_demand=True,
    h_spectrum_is_flat_so_no_low_rank_claim=True))

# ------------------------------------------------------------- Schur deletion
status('SCHUR_DELETION')
selected_modes = modes[:selected_rank]
P_N = orthogonal_complement_projector(P_D, selected_modes)
g0 = gradients[0]
full0 = motions[0]
rest0, rest0_report = solve('deletion_g0_restricted', g0, P_N,
                            note='forbid the selected candidate motion space',
                            projector_id=subspace_id('candidate', selected_modes))
gap = rest0.q - full0.q
difference = rest0.vector - full0.vector
schur_energy = .5 * float(torch.dot(difference, hessian_action(GEO, difference,
                                                               a.lam, beta, 'DTD')))
deletion = dict(
    selected_rank=selected_rank, full_q=full0.q, restricted_q=rest0.q,
    proxy_deletion=gap, schur_energy=schur_energy, schur_identity_error=gap - schur_energy,
    full_norm=float(torch.linalg.vector_norm(full0.vector)),
    restricted_norm=float(torch.linalg.vector_norm(rest0.vector)),
    forbidden_motion_norm=float(torch.linalg.vector_norm(difference)),
    full_stationarity=full0.stationarity_ratio,
    restricted_stationarity=rest0_report['relative_stationarity'],
    restricted_converged=bool(rest0_report['converged']),
    identity_holds_only_for_converged_common_quadratic=True)
atomic_json(out / 'deletions' / 'g0_selected_rank.json', deletion)
save_vector('complement_at_zero', rest0.vector)
status('SCHUR_DELETION_READY', proxy_deletion=gap,
       schur_identity_error=deletion['schur_identity_error'])

# ------------------------------------------------- conditional compensation
status('COMPENSATION')
x_unit = project_onto(selected_modes, full0.vector)
x_norm = float(torch.linalg.vector_norm(x_unit))
x_leak = float(torch.linalg.vector_norm(P_N(x_unit)))
if x_leak > 1e-9 * max(x_norm, 1.):
    raise ValueError('Driver is not fully inside the forbidden motion space')
effective_g = g0 + hessian_action(GEO, x_unit, a.lam, beta, 'DTD')
nx_unit, nx_report = solve('driver_g0_at_x', effective_g, P_N,
                           note='complement optimum when the full candidate motion is forced',
                           projector_id=subspace_id('candidate', selected_modes))
del effective_g
compensation_unit = nx_unit.vector - rest0.vector

half_x = .5 * x_unit
half_g = g0 + hessian_action(GEO, half_x, a.lam, beta, 'DTD')
nx_half, _ = solve('driver_g0_at_half_x', half_g, P_N,
                   note='linearity verification solve',
                   projector_id=subspace_id('candidate', selected_modes))
del half_g
predicted_half = rest0.vector + .5 * compensation_unit
linearity_error = float(torch.linalg.vector_norm(nx_half.vector - predicted_half)) / max(
    float(torch.linalg.vector_norm(nx_half.vector)), np.finfo(float).tiny)
del nx_half
torch.cuda.empty_cache()

driver_rows = []
joint_vectors = {}
for t in a.drivers:
    x_t = t * x_unit
    n_t = rest0.vector + t * compensation_unit
    joint = x_t + n_t
    joint_vectors[t] = dict(x=x_t, complement=n_t, joint=joint)
    q_joint = quadratic_objective(GEO, g0, joint, a.lam, beta, 'DTD')
    driver_rows.append(dict(
        amplitude=t, driver_norm=float(torch.linalg.vector_norm(x_t)),
        complement_norm=float(torch.linalg.vector_norm(n_t)),
        compensation_norm=float(torch.linalg.vector_norm(t * compensation_unit)),
        compensation_over_driver=float(torch.linalg.vector_norm(t * compensation_unit)) / max(
            float(torch.linalg.vector_norm(x_t)), np.finfo(float).tiny),
        q_joint=q_joint, q_full=full0.q, q_complement_only=rest0.q,
        joint_minus_full=q_joint - full0.q))
atomic_json(out / 'COMPENSATION.json', dict(
    rows=driver_rows,
    driver_source='orthogonal projection of the unrestricted g0 motion onto the selected modes',
    driver_norm=x_norm, driver_leak_into_complement=x_leak,
    complement_at_zero_norm=float(torch.linalg.vector_norm(rest0.vector)),
    compensation_unit_norm=float(torch.linalg.vector_norm(compensation_unit)),
    linearity_relative_error=linearity_error,
    linearity_model='n(t x) = n(0) + t (n(x) - n(0)) for one common quadratic',
    compensation_separate_from_complement_own_motion=True,
    amplitude_sweep_uses_one_compensator_solve=True))
torch.save(dict(x_unit=x_unit.detach().cpu(),
                complement_at_zero=rest0.vector.detach().cpu(),
                complement_at_x=nx_unit.vector.detach().cpu(),
                compensation=compensation_unit.detach().cpu(),
                amplitudes=a.drivers,
                joints=[joint_vectors[t]['joint'].detach().cpu() for t in a.drivers]),
           out / 'compensation.pt')
for t in a.drivers:
    save_vector(f'joint_amplitude{t:g}', joint_vectors[t]['joint'])
status('COMPENSATION_READY', linearity_error=linearity_error)

# ------------------------------------------------------- finite evaluation
status('FINITE_EVALUATION')
base_logits = {probe: CTX.baseline('DTD', probe, point)[0] for probe in (0, 2, 3)}


def finite_rows(name, delta_vector, proxy_q, note):
    theta = point + delta_vector
    rows = []
    for probe in (0, 2, 3):
        logits, labels = CTX.baseline('DTD', probe, theta)
        base = base_logits[probe]
        m = metric_stats(logits.numpy(), labels.numpy(),
                         class_weights(base.shape[1], 'mean'), base.numpy())
        rows.append(dict(state=name, probe=probe, images=int(m['n']), CE=m['CE'],
                         Top1=m['Top1'], delta_CE=m['delta_CE'],
                         delta_weighted_CE=m['delta_weighted_CE'],
                         delta_Top1=m['delta_Top1'],
                         prediction_flip_rate=m['prediction_flip_rate'],
                         paired_CI_lower=m['paired_weighted_CE_approx95'][0],
                         paired_CI_upper=m['paired_weighted_CE_approx95'][1],
                         parameter_norm=float(torch.linalg.vector_norm(delta_vector)),
                         proxy_q=proxy_q, note=note))
    return rows


evaluation = [dict(state='anchor', probe=None, images=None, CE=0., Top1=0., delta_CE=0.,
                   delta_weighted_CE=0., delta_Top1=0., prediction_flip_rate=0.,
                   paired_CI_lower=0., paired_CI_upper=0., parameter_norm=0.,
                   proxy_q=0., note='reference anchor, no movement')]
evaluation += finite_rows('full_optimum', full0.vector, full0.q,
                          'unrestricted quadratic optimum')
evaluation += finite_rows('restricted_optimum', rest0.vector, rest0.q,
                          'selected motion space forbidden')
for t in a.drivers:
    joint = joint_vectors[t]['joint']
    evaluation += finite_rows(f'joint_amplitude_{t:g}', joint,
                              quadratic_objective(GEO, g0, joint, a.lam, beta, 'DTD'),
                              'compensated joint update')
    evaluation += finite_rows(f'driver_only_{t:g}', joint_vectors[t]['x'],
                              quadratic_objective(GEO, g0, joint_vectors[t]['x'],
                                                  a.lam, beta, 'DTD'),
                              'forced motion without complement compensation')

audit = GEO.operator.audit()
for row in evaluation:
    row['operator_JVP_calls'] = audit['JVP_calls']
    row['operator_VJP_calls'] = audit['VJP_calls']
    row['operator_J_seconds'] = audit['seconds']['J']
    row['operator_JT_seconds'] = audit['seconds']['JT']
columns = list(dict.fromkeys(key for row in evaluation for key in row))
with (out / 'evaluation.csv').open('w', newline='', encoding='utf8') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(evaluation)

# ------------------------------------------------------------------- report
lines = ['# DTD Schur conditional compensation, step 0', '',
         f'Anchor {a.anchor}; active domain {len(active_names)} attention/MLP weight '
         f'matrices ({int(active.sum())} of {CTX.D} coordinates).', '',
         '## Numerical gates', '',
         f'- JVP/VJP adjointness relative error: {adjoint_error:.3e}',
         f'- kappa = ||Jg||^2/||g||^2 = {kappa:.4f}; beta = {beta:.3e}',
         f'- Candidate modes: {len(modes)}; max orthonormality error '
         f"{mode_report['max_orthonormality_error']:.3e}",
         f'- Compensation linearity relative error: {linearity_error:.3e}', '',
         '## Independent rank selection (probe 2; probe 3 never used)', '',
         '| demand | family | rank | proxy deletion | / abs(q_full) | Schur identity error |',
         '|---|---|---:|---:|---:|---:|']
for row in selection_rows:
    lines.append(f"| {row['select_demand']} | {row['family']} | {row['rank']} | "
                 f"{row['proxy_deletion']:.6g} | {row['normalized_deletion']:.3e} | "
                 f"{row['schur_identity_error']:.3e} |")
lines += ['', 'Rank advantages against dimension-matched baselines:', '',
          '| rank | candidate deletion | best matched baseline | advantage |',
          '|---:|---:|---:|---:|']
for row in advantages:
    base = 'absent' if row['best_matched_baseline'] is None else f"{row['best_matched_baseline']:.6g}"
    adv = 'not comparable' if not row['comparable'] else f"{row['advantage']:.6g}"
    lines.append(f"| {row['rank']} | {row['candidate_deletion']:.6g} | {base} | {adv} |")
lines += ['', f'Selected rank: {selected_rank}. Deletion grows monotonically with the '
          'size of the forbidden subspace, so the rank is chosen by the advantage over '
          'matched alternatives, not by deletion itself. The H-spectrum of the '
          'construction motions is flat (see MODES.json), so nothing here supports a '
          'low-rank movement claim.', '',
          '## Schur deletion on the construction demand', '',
          f'- q(full) = {full0.q:.6f}; q(restricted) = {rest0.q:.6f}',
          f'- proxy deletion = {gap:.6f}',
          f"- Schur identity error = {deletion['schur_identity_error']:.3e}", '',
          '## Conditional compensation', '',
          '| amplitude | driver norm | compensation norm | compensation/driver '
          '| q(joint) - q(full) |', '|---|---:|---:|---:|---:|']
for row in driver_rows:
    lines.append(f"| {row['amplitude']:g} | {row['driver_norm']:.6g} | "
                 f"{row['compensation_norm']:.6g} | {row['compensation_over_driver']:.4f} | "
                 f"{row['joint_minus_full']:.6g} |")
lines += ['', '## Finite DTD outcomes (delta against the anchor)', '',
          '| state | probe | CE change | Top-1 change | flip rate | parameter norm |',
          '|---|---:|---:|---:|---:|---:|']
for row in evaluation:
    if row['probe'] is None:
        continue
    lines.append(f"| {row['state']} | {row['probe']} | {row['delta_CE']:.6g} | "
                 f"{row['delta_Top1']:.6g} | {row['prediction_flip_rate']:.4f} | "
                 f"{row['parameter_norm']:.6g} |")
lines += ['',
          'Probes 0/2/3 are disjoint training-side development images. Probe 3 is used '
          'only for the table above. The quadratic proxy is not the network CE Hessian: '
          'a large proxy deletion with no held-out CE difference indicates local proxy '
          'structure, not an irreplaceable task parameter.']
(out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf8')

sources = {str(path): digest(path) for folder in (HERE, CORE, LEGACY, R3, SPACE, FLOW)
           for path in Path(folder).glob('*.py')}
atomic_json(out / 'CONFIG.json', dict(
    anchor=a.anchor, demands=a.demands, select_demands=a.select_demands,
    max_rank=a.max_rank, alpha=a.alpha, lam=a.lam, beta=beta, kappa=kappa,
    rtol=a.rtol, maxiter=a.maxiter, batch_size=a.batch_size, drivers=a.drivers,
    active_domain=active_report, provenance=provenance, sources=sources,
    inputs={str(path): digest(path) for path in (a.inventory, a.probes,
                                                 a.trace / 'trace.json')},
    official_test_opened=False, training_side_development_probes=True,
    no_new_expert_training=True, no_online_writeback=True))
status('COMPLETE', selected_rank=selected_rank)
print(json.dumps(dict(complete=True, selected_rank=selected_rank, proxy_deletion=gap,
                      linearity_error=linearity_error, audit=audit)), flush=True)
