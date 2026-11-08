"""Replay existing DTD trajectories and separate gradient and optimizer signals."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from collect_dtd_updates import batches, flat, load_experiment, save
from study_dtd_quotient import frame


def replay(args):
    manifest = json.loads((args.trajectory / 'complete.json').read_text())
    if manifest['shuffled']:
        raise ValueError('Expected a real-label DTD trajectory')
    seed = manifest['seed']
    existing, model, found, base, dataset, _, _, ids, base_hash = load_experiment(args.root, seed)
    if manifest['base_sha256'] != base_hash or manifest['names'] != [n for n, _ in found]:
        raise RuntimeError('Trajectory provenance mismatch')
    paths = {int(path.stem.split('_')[-1]): path for path in args.trajectory.glob('update_*.pt')}
    args.output.mkdir(parents=True, exist_ok=False)
    params = [p for _, p in found]
    for parameter in params:
        parameter.requires_grad_(True)
    lr, wd = 1e-5, .1
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd, fused=True)
    before = window_start = base
    raw = torch.zeros_like(base)
    decay = torch.zeros_like(base)
    per_step = []
    first = None
    step = 0
    checks = []
    for epoch in range(1, manifest['epochs'] + 1):
        for x, y, _ in batches(dataset, ids[144:], seed, epoch):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(y), 16):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = model(x[offset:offset + 16]).float()
                    loss = F.cross_entropy(z, y[offset:offset + 16], reduction='sum') / len(y)
                loss.backward()
            g = torch.cat([p.grad.flatten() for p in params])
            raw.add_(g, alpha=-lr)
            if first is None:
                first = -lr * g
            decay.add_(before, alpha=lr * wd)
            optimizer.step()
            after = flat(found)
            if args.oracle:
                per_step.append(after - before)
            before = after
            step += 1
            if step in paths:
                saved = torch.load(paths[step], map_location='cpu', weights_only=True, mmap=True)['delta'].cuda()
                delta = after - window_start
                relative = float((delta - saved).norm() / saved.norm())
                if relative > 1e-4:
                    raise RuntimeError(f'Could not reproduce seed {seed}, window {step}: {relative:.3g}')
                record = dict(step=step, raw_gradient=raw.cpu(), first_gradient=first.cpu(),
                              without_decay=(delta + decay).cpu(),
                              decay_relative_norm=float(decay.norm() / delta.norm()), replay_relative_error=relative)
                if args.oracle:
                    record['oracle'] = frame(torch.stack(per_step), 2).cpu()
                save(args.output / f'signals_{step:04d}.pt', record)
                checks.append(dict(step=step, relative_error=relative))
                raw.zero_()
                decay.zero_()
                window_start, first = after, None
                per_step.clear()
                print(dict(seed=seed, step=step, relative_error=relative), flush=True)
    if len(checks) != len(paths):
        raise RuntimeError('Missing replay windows')
    existing.write(args.output / 'complete.json', dict(seed=seed, base_sha256=base_hash, lr=lr, weight_decay=wd,
                                                       checks=checks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--oracle', action='store_true')
    replay(parser.parse_args())
