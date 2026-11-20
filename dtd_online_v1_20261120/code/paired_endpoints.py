"""Paired per-image comparison of the three arm endpoints on probe 3.

Reads only the saved ENDPOINT.pt files and the frozen probes. No training, no CG.
The three arms saw the same p3 images, so the differences are paired and their
standard error must be computed per image, not from independent-run CIs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runtime-code-dir', type=Path, required=True)
    ap.add_argument('--inventory', type=Path, required=True)
    ap.add_argument('--probes', type=Path, required=True)
    ap.add_argument('--trace', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--anchor', type=int, default=0)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('runs', type=Path, nargs='+', help='directories containing ENDPOINT.pt')
    a = ap.parse_args()

    sys.path.insert(0, str(a.runtime_code_dir))
    import bootstrap  # noqa: F401
    from carrier import TrainingPositions
    from geometry import FullFunctionalGeometry
    import task_runtime

    import shutil
    a.output.mkdir(parents=True, exist_ok=True)
    runtime = a.output / 'runtime'
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

    positions = TrainingPositions(a.trace, base.keys, a.output)
    index = next(i for i, r in enumerate(positions.rows) if r['step'] == a.anchor)
    anchor = positions.flat(index).cuda().double()
    geo = FullFunctionalGeometry(base, anchor, 0, precision='float64')
    ctx = geo.c

    def per_image_ce(theta, probe):
        ce, lab = [], []
        for x, y in ctx.batches['DTD', probe]:
            logits = ctx.predict(theta, x, 'DTD')
            ce.append(F.cross_entropy(logits, y.long(), reduction='none').detach().double().cpu())
            lab.append(y.cpu())
        return torch.cat(ce).numpy(), torch.cat(lab).numpy()

    points = {'anchor': anchor}
    for run in a.runs:
        state = torch.load(run / 'ENDPOINT.pt', map_location='cpu', weights_only=True)
        points[run.name] = state['theta'].cuda().double()

    ce = {}
    labels = None
    for name, theta in points.items():
        ce[name], labels = per_image_ce(theta, 3)

    names = [r.name for r in a.runs]
    rows = []
    base_ce = ce['anchor']
    for name in names:
        d = ce[name] - base_ce
        rows.append(dict(state=name, mean_delta=float(d.mean()),
                         se=float(d.std(ddof=1) / np.sqrt(len(d))),
                         ci_low=float(d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                         ci_high=float(d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d))),
                         improved=int((d < 0).sum()), worsened=int((d > 0).sum())))
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = ce[names[i]] - ce[names[j]]
            se = float(d.std(ddof=1) / np.sqrt(len(d)))
            pairs.append(dict(first=names[i], second=names[j], mean_difference=float(d.mean()),
                              se=se, t=float(d.mean() / se) if se else None,
                              ci_low=float(d.mean() - 1.96 * se),
                              ci_high=float(d.mean() + 1.96 * se),
                              first_better=int((d < 0).sum()),
                              second_better=int((d > 0).sum())))

    report = dict(probe=3, images=int(len(labels)), anchor_step=a.anchor,
                  arms=rows, paired_differences=pairs,
                  note='negative delta means lower CE than the anchor; '
                       'differences are paired on identical p3 images')
    (a.output / 'PAIRED_P3.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
