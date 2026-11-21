"""Export the physical-parameter inputs the Pareto-balance pilot needs.

Produces, at one common anchor theta_a (the pretrained visual tower, which is the
base that task arithmetic is defined against):

  g_d_train.npy, g_c_train.npy      CE gradients on each task's primary probe
  g_d_heldout.npy, g_c_heldout.npy  CE gradients on a disjoint probe, same anchor
  u_d.npy, u_c.npy                  expert visual weights minus theta_a

All six are flat float64 in one fixed key order (the model's named-parameter
order), with a manifest recording keys, shapes, offsets and sha256.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--inventory', type=Path, required=True)
    ap.add_argument('--probes', type=Path, required=True)
    ap.add_argument('--runtime-code-dir', type=Path, required=True)
    ap.add_argument('--tasks', nargs='+', default=['DTD', 'Cars'])
    ap.add_argument('--train-probe', type=int, default=0)
    ap.add_argument('--heldout-probe', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=16)
    a = ap.parse_args()
    if len(a.tasks) != 2:
        raise ValueError('this pilot is two-task only')

    sys.path.insert(0, str(a.runtime_code_dir))
    import bootstrap  # noqa: F401
    from geometry import FullFunctionalGeometry  # noqa: F401  (ensures the paths resolve)
    from precision import Float64Context
    import task_runtime

    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime = out / 'runtime'
    runtime.mkdir(exist_ok=True)
    shutil.copy2(a.probes, runtime / 'probes.json')
    inventory = [r for r in json.loads(a.inventory.read_text())['tasks'] if r['task'] in a.tasks]
    order = tuple(a.tasks)
    old = task_runtime.TASKS
    task_runtime.TASKS = order
    try:
        base = task_runtime.EightTaskContext(runtime, inventory)
    finally:
        task_runtime.TASKS = old
    for key, rows in list(base.batches.items()):
        base.batches[key] = [(x[i:i + a.batch_size], y[i:i + a.batch_size])
                             for x, y in rows for i in range(0, len(y), a.batch_size)]

    ctx = Float64Context(base)
    anchor = base.base.cuda().double()          # pretrained visual tower, key order fixed
    dimension = anchor.numel()

    def gradient(task, probe):
        batches = ctx.batches[task, probe]
        total = sum(len(y) for _, y in batches)
        g = torch.zeros_like(anchor)
        for x, y in batches:
            with torch.enable_grad():
                leaf = anchor.detach().requires_grad_(True)
                logits = ctx.predict(leaf, x, task)
                g += torch.autograd.grad(
                    F.cross_entropy(logits, y, reduction='sum') / total, leaf)[0].detach().double()
        return g

    def expert_delta(task):
        row = next(r for r in inventory if r['task'] == task)
        state = torch.load(row['expert_path'], map_location='cpu', weights_only=False)['visual']
        flat = torch.cat([state[k].reshape(-1) for k in base.keys]).double()
        if flat.numel() != dimension:
            raise ValueError(f'{task} expert has {flat.numel()} values, expected {dimension}')
        return flat - anchor.detach().cpu().double()

    arrays = {}
    for task in a.tasks:
        arrays[f'g_{task[0].lower()}_train'] = gradient(task, a.train_probe)
        arrays[f'g_{task[0].lower()}_heldout'] = gradient(task, a.heldout_probe)
        arrays[f'u_{task[0].lower()}'] = expert_delta(task)
        print(f'{task}: |g_train|={float(torch.linalg.vector_norm(arrays[f"g_{task[0].lower()}_train"])):.4f} '
              f'|g_held|={float(torch.linalg.vector_norm(arrays[f"g_{task[0].lower()}_heldout"])):.4f} '
              f'|u|={float(torch.linalg.vector_norm(arrays[f"u_{task[0].lower()}"])):.4f}', flush=True)

    files = {}
    for name, tensor in arrays.items():
        path = out / f'{name}.npy'
        np.save(path, tensor.detach().cpu().numpy().astype(np.float64))
        files[name] = dict(file=path.name, sha256=digest(path),
                           norm=float(torch.linalg.vector_norm(tensor)))
        del tensor
        torch.cuda.empty_cache()

    offsets, position = [], 0
    for name, size in zip(base.keys, base.sizes):
        offsets.append(dict(name=name, shape=list(base.shapes[len(offsets)]),
                            offset=position, numel=int(size)))
        position += size
    manifest = dict(
        anchor='pretrained visual tower (models/checkpoints/ViT-B-16/pretrained.pt)',
        anchor_note='task arithmetic is defined as theta_0 + sum_k (theta_k - theta_0), '
                    'so both gradients and both deltas live at this one point',
        tasks=list(a.tasks), train_probe=a.train_probe, heldout_probe=a.heldout_probe,
        probe_disjoint=True, dimension=int(dimension), tensors=len(base.keys),
        key_order=base.keys, tensors_offsets=offsets, arrays=files,
        gradient_definition='mean cross entropy over the probe',
        head_note='each task is evaluated with its own trained classification head',
        inputs={str(p): digest(p) for p in (a.inventory, a.probes)},
        dataset='training-side development probes only; no official validation or test')
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print('wrote', out)


if __name__ == '__main__':
    main()
