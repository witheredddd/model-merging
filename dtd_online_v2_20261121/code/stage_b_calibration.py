"""Stage B: predeclared trust-radius calibration. Training-side only, no CG.

Protocol (fixed before looking at any number):
  direction   AdamW's first proposal from the gradient on update-side batch U0
  radii       [0.005, 0.02, 0.05, 0.09]; only if none pass, [0.0017]
  probe       the frozen p0 functional probe; Jacobian taken at the anchor
  metrics     eps_lin = ||phi(t+d) - phi(t) - J d|| / ||phi(t+d) - phi(t)||
              finite CE change on the update-side batch U1 (never used to build
              the direction) and on the training-side selection split p2
  gate        CE improves on both, and eps_lin <= 0.20
  choice      the largest radius that passes
p3 is never read and no inverse CG is solved.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RADII = [.005, .02, .05, .09]
FALLBACK = [.0017]
TOLERANCE = .20


def main():
    ap = argparse.ArgumentParser()
    for key in ('output', 'inventory', 'probes', 'trace', 'runtime-code-dir'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--anchor', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--direction-batch', type=int, default=0)
    ap.add_argument('--evaluation-batch', type=int, default=1)
    ap.add_argument('--tolerance', type=float, default=TOLERANCE)
    a = ap.parse_args()

    sys.path.insert(0, str(a.runtime_code_dir))
    import bootstrap  # noqa: F401
    from carrier import TrainingPositions
    from geometry import FullFunctionalGeometry
    import task_runtime
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from controller import AdamWProposal
    from core import norm

    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime = out / 'runtime'
    runtime.mkdir(exist_ok=True)
    shutil.copy2(a.probes, runtime / 'probes.json')
    inventory = [r for r in json.loads(a.inventory.read_text())['tasks']
                 if r['task'] in ('DTD', 'Cars')]
    old = task_runtime.TASKS
    task_runtime.TASKS = ('DTD', 'Cars')
    try:
        base = task_runtime.EightTaskContext(runtime, inventory)
    finally:
        task_runtime.TASKS = old
    for key, rows in list(base.batches.items()):
        base.batches[key] = [(x[i:i + a.batch_size], y[i:i + a.batch_size])
                             for x, y in rows for i in range(0, len(y), a.batch_size)]

    positions = TrainingPositions(a.trace, base.keys, out)
    provenance = positions.verify_normal_recipe()
    index = next(i for i, r in enumerate(positions.rows) if r['step'] == a.anchor)
    point = positions.flat(index).cuda().double()

    # Exactly the split the online driver uses; p2 is the training-side selection
    # split and p3 is never touched here.
    xx = torch.cat([x for x, _ in base.batches['DTD', 1]])
    yy = torch.cat([y for _, y in base.batches['DTD', 1]])
    splits = [[], []]
    for cls in yy.unique().tolist():
        ids = (yy == cls).nonzero().flatten().tolist()
        for k, i in enumerate(ids):
            splits[k % 2].append(int(i))
    for probe, ids in zip((2, 3), splits):
        base.batches['DTD', probe] = [(xx[ids[k:k + a.batch_size]], yy[ids[k:k + a.batch_size]])
                                      for k in range(0, len(ids), a.batch_size)]
    del xx, yy

    active = torch.zeros(point.numel(), dtype=torch.bool, device='cuda')
    offset = 0
    for name, size in zip(base.keys, base.sizes):
        if any(name.endswith(t) for t in ('attn.in_proj_weight', 'attn.out_proj.weight',
                                          'mlp.c_fc.weight', 'mlp.c_proj.weight')):
            active[offset:offset + size] = True
        offset += size

    geo = FullFunctionalGeometry(base, point, 0, precision='float64')
    ctx = geo.c
    probe_images = sum(len(y) for _, y in base.batches['DTD', 0])

    update_batches = base.batches['DTD', 0]
    if max(a.direction_batch, a.evaluation_batch) >= len(update_batches):
        raise ValueError('update-side batch index out of range')
    direction_batch = update_batches[a.direction_batch]
    evaluation_batch = update_batches[a.evaluation_batch]

    def gradient(theta, batches):
        total = sum(len(y) for _, y in batches)
        g = torch.zeros_like(theta)
        for x, y in batches:
            for k in range(0, len(y), a.batch_size):
                with torch.enable_grad():
                    leaf = theta.detach().requires_grad_(True)
                    logits = ctx.predict(leaf, x[k:k + a.batch_size], 'DTD')
                    loss = F.cross_entropy(logits, y[k:k + a.batch_size], reduction='sum') / total
                    g += torch.autograd.grad(loss, leaf)[0].detach().double()
        return g

    def phi(theta):
        rows = []
        for x, _ in base.batches['DTD', 0]:
            logits = ctx.predict(theta, x, 'DTD')
            rows.append((2 * logits.softmax(-1).sqrt() / np.sqrt(probe_images)).detach().reshape(-1))
        return torch.cat(rows)

    def ce(theta, batches):
        values, correct = [], []
        for x, y in batches:
            logits = ctx.predict(theta, x, 'DTD')
            values.append(F.cross_entropy(logits, y.long(), reduction='none').detach().double().cpu())
            correct.append((logits.argmax(-1) == y).double().cpu())
        return float(torch.cat(values).mean()), float(torch.cat(correct).mean())

    g = gradient(point, [direction_batch])
    optim = AdamWProposal(point, active, lr=1e-5)
    u = optim.propose(point, g)
    u_norm = norm(u)
    direction = u / u_norm

    phi0 = phi(point)
    ju = geo.jvp('DTD', direction)
    ce_anchor_eval, top_anchor_eval = ce(point, [evaluation_batch])
    ce_anchor_p2, top_anchor_p2 = ce(point, base.batches['DTD', 2])

    def evaluate(radius):
        delta = radius * direction
        phi1 = phi(point + delta)
        actual = phi1 - phi0
        predicted = radius * ju
        denominator = max(float(torch.linalg.vector_norm(actual)), 1e-30)
        eps = float(torch.linalg.vector_norm(actual - predicted)) / denominator
        ce_eval, top_eval = ce(point + delta, [evaluation_batch])
        ce_p2, top_p2 = ce(point + delta, base.batches['DTD', 2])
        del phi1, actual, predicted, delta
        torch.cuda.empty_cache()
        return dict(radius=radius,
                    delta_norm=radius,
                    jvp_norm=radius * float(torch.linalg.vector_norm(ju)),
                    actual_phi_norm=denominator,
                    eps_linear=eps,
                    ce_eval_batch=ce_eval, delta_ce_eval_batch=ce_eval - ce_anchor_eval,
                    top1_eval_batch=top_eval,
                    ce_selection_p2=ce_p2, delta_ce_selection_p2=ce_p2 - ce_anchor_p2,
                    top1_selection_p2=top_p2,
                    ce_improves=bool(ce_eval < ce_anchor_eval and ce_p2 < ce_anchor_p2),
                    linearisation_ok=bool(eps <= a.tolerance))

    rows = [evaluate(r) for r in RADII]
    passes = [r for r in rows if r['ce_improves'] and r['linearisation_ok']]
    fallback_rows = []
    if not passes:
        fallback_rows = [evaluate(r) for r in FALLBACK]
        passes = [r for r in fallback_rows if r['ce_improves'] and r['linearisation_ok']]
    chosen = max((r['radius'] for r in passes), default=None)

    report = dict(
        anchor=a.anchor, tolerance=a.tolerance,
        predefined_radii=RADII, fallback_radii=FALLBACK,
        direction_source=dict(probe='p0', batch_index=a.direction_batch,
                              images=int(len(direction_batch[1])),
                              adamw_lr=1e-5, proposal_norm=u_norm,
                              unit_direction_norm=1.),
        evaluation=dict(update_side_batch_index=a.evaluation_batch,
                        images=int(len(evaluation_batch[1])),
                        selection_split='p2',
                        p2_images=int(sum(len(y) for _, y in base.batches['DTD', 2])),
                        functional_probe='p0', functional_probe_images=probe_images),
        anchor_metrics=dict(ce_eval_batch=ce_anchor_eval, top1_eval_batch=top_anchor_eval,
                            ce_selection_p2=ce_anchor_p2, top1_selection_p2=top_anchor_p2),
        rows=rows, fallback_rows=fallback_rows,
        chosen_radius=chosen,
        gate='finite CE improves on both evaluation sets AND eps_linear <= tolerance',
        p3_never_opened=True, no_cg_solved=True, radius_not_tuned_per_arm=True,
        provenance=provenance)
    (out / 'STAGE_B.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('chosen_radius', 'anchor_metrics')}, indent=2))
    for r in rows + fallback_rows:
        print(f"  r={r['radius']:<7g} |d|={r['delta_norm']:.3e} eps={r['eps_linear']:.4f} "
              f"dCE_eval={r['delta_ce_eval_batch']:+.3e} dCE_p2={r['delta_ce_selection_p2']:+.3e} "
              f"ce_ok={r['ce_improves']} lin_ok={r['linearisation_ok']}")


if __name__ == '__main__':
    main()
