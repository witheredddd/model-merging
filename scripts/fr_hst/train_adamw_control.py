"""DTD full-weight AdamW control with the atlas training samples and batch order."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def train(args):
    sys.path[:0] = [str(args.root), str(args.root / 'scripts'), str(args.root / 'scripts/tfpml')]
    import dtd_grassmann4 as existing
    import runtime

    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    existing.seed_all(args.seed)
    existing.ROOT = args.root
    path = args.root / 'models/checkpoints/ViT-B-16/head_DTDVal.rebuilt.pt'
    if existing.sha(path) != json.loads(path.with_suffix('.json').read_text())['sha256']:
        raise RuntimeError('DTD head checksum mismatch')
    head = existing.ClassificationHead.load(str(path))
    encoder, model, found, initial = runtime.build_model(existing, head)
    del initial
    model.requires_grad_(False).eval()
    for _, parameter in found:
        parameter.requires_grad_(True)
    trainset, _, validation, _ = runtime.training_data(existing, encoder, args.seed)
    train_ids = torch.randperm(len(trainset), generator=torch.Generator().manual_seed(900)).tolist()[96:]
    optimizer = torch.optim.AdamW([p for _, p in found], lr=args.lr, weight_decay=.1, fused=True)
    args.output.mkdir(parents=True, exist_ok=False)

    def record(epoch, step):
        row = dict(seed=args.seed, epoch=epoch, step=step, lr=args.lr, **existing.evaluate(model, validation))
        with (args.output / 'validation.jsonl').open('a') as out:
            out.write(json.dumps(row) + '\n')
        print(row, flush=True)

    step = 0
    record(0, step)
    for epoch in range(1, args.epochs + 1):
        trainset.epoch = epoch - 1
        order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        batches = [[train_ids[i] for i in order[start:start + args.batch]]
                   for start in range(0, len(order), args.batch)]
        loader = torch.utils.data.DataLoader(trainset, batch_sampler=batches, num_workers=4, pin_memory=True)
        for x, y, _ in loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            for j in range(0, len(y), 16):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(x[j:j + 16]).float()
                    loss = F.cross_entropy(pred, y[j:j + 16], reduction='sum') / len(y)
                loss.backward()
            optimizer.step()
            step += 1
        record(epoch, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.lr <= 0:
        parser.error('Invalid optimizer or training budget')
    train(args)
