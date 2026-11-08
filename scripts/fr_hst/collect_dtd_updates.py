"""Collect realized AdamW displacements for an independent DTD geometry experiment."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def load_experiment(root, seed):
    sys.path[:0] = [str(root), str(root / 'scripts'), str(root / 'scripts/tfpml')]
    import dtd_grassmann4 as existing
    import runtime

    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    existing.ROOT = root
    existing.seed_all(seed)
    path = root / 'models/checkpoints/ViT-B-16/head_DTDVal.rebuilt.pt'
    metadata = json.loads(path.with_suffix('.json').read_text())
    if existing.sha(path) != metadata['sha256']:
        raise RuntimeError('DTD head checksum mismatch')
    head = existing.ClassificationHead.load(str(path))
    encoder, model, found, base = runtime.build_model(existing, head)
    model.requires_grad_(False).eval()
    dataset, _, validation, splits = runtime.training_data(existing, encoder, seed)
    if [c.replace('_', ' ') for c in splits.train_dataset.dataset.classes] != metadata['classnames']:
        raise RuntimeError('DTD class order mismatch')
    ids = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(901)).tolist()
    base_hash = existing.sha(path.parent / 'pretrained.pt')
    return existing, model, found, base, dataset, validation, splits, ids, base_hash


def batches(dataset, ids, seed, epoch):
    dataset.epoch = epoch - 1
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    batch_ids = [[ids[i] for i in order[j:j + 64]] for j in range(0, len(order), 64)]
    return torch.utils.data.DataLoader(dataset, batch_sampler=batch_ids, num_workers=4, pin_memory=True)


def flat(found):
    return torch.cat([p.detach().flatten() for _, p in found])


def save(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def collect(args):
    existing, model, found, base, dataset, validation, splits, ids, base_hash = load_experiment(args.root, args.seed)
    train_ids = ids[144:]
    for _, parameter in found:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([p for _, p in found], lr=1e-5, weight_decay=.1, fused=True)
    labels = torch.tensor([splits.train_dataset.dataset.targets[splits.train_dataset.indices[i]]
                           for i in train_ids])
    # Shuffle once at the individual example level, keeping exact label counts.
    shuffled = torch.empty(len(dataset), dtype=torch.long)
    shuffled[train_ids] = labels[torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(args.seed + 1300))]
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(seed=args.seed, shuffled=args.shuffled, base_sha256=base_hash, names=[n for n, _ in found],
                    train=len(train_ids), reserved=144, window=args.window, epochs=args.epochs,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    existing.write(args.output / 'config.json', manifest)
    start = base
    step = 0
    for epoch in range(1, args.epochs + 1):
        loader = batches(dataset, train_ids, args.seed, epoch)
        for i, (images, labels, indices) in enumerate(loader, 1):
            x = images.cuda(non_blocking=True)
            y = (shuffled[indices] if args.shuffled else labels).cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(y), 16):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = model(x[offset:offset + 16]).float()
                    loss = F.cross_entropy(z, y[offset:offset + 16], reduction='sum') / len(y)
                loss.backward()
            optimizer.step()
            step += 1
            if i % args.window == 0 or i == len(loader):
                current = flat(found)
                save(args.output / f'update_{step:04d}.pt', dict(epoch=epoch, step=step, delta=(current - start).cpu()))
                start = current
        score = existing.evaluate(model, validation)
        if epoch == 1:
            save(args.output / 'anchor_epoch1.pt', start.cpu())
        manifest[f'epoch_{epoch}'] = dict(step=step, **score)
        existing.write(args.output / 'config.json', manifest)
        print(dict(seed=args.seed, epoch=epoch, step=step, **score), flush=True)
    existing.write(args.output / 'complete.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--shuffled', action='store_true')
    args = parser.parse_args()
    if args.window < 1 or args.epochs < 2:
        parser.error('At least two trajectory regions are required')
    collect(args)
