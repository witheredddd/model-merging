"""Independent DTD/SUN397 experts and reproducible probes for the fiber atlas."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import ImageFile
from torch.nn import functional as F
from torchvision.datasets import ImageFolder

from collect_dtd_updates import batches, flat, save
from study_dtd_quotient import predictor


def load_task(root, task, seed=42):
    sys.path[:0] = [str(root), str(root / 'scripts'), str(root / 'scripts/tfpml')]
    import dtd_grassmann4 as existing
    import runtime

    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    existing.ROOT = root
    existing.seed_all(seed)
    base_path = root / 'models/checkpoints/ViT-B-16/pretrained.pt'
    if task == 'DTD':
        head_path = base_path.parent / 'head_DTDVal.rebuilt.pt'
    else:
        head_path = root / 'results/functional_geometry_v2_no_expert/assets/heads/SUN397.pt'
        # Match the existing SUN397 decoder audit for its recoverable JPEG.
        ImageFile.LOAD_TRUNCATED_IMAGES = True
    metadata = json.loads(head_path.with_suffix('.json').read_text())
    if existing.sha(head_path) != metadata['sha256']:
        raise RuntimeError(f'{task} classification head provenance mismatch')
    encoder, model, found, base = runtime.build_model(existing, existing.ClassificationHead.load(str(head_path)))
    model.requires_grad_(False).eval()
    folder = root / 'datasets' / task.lower() / 'train'
    training = ImageFolder(folder, transform=encoder.train_preprocess)
    evaluation = ImageFolder(folder, transform=encoder.val_preprocess)
    count = len(training)
    if count != {'DTD': 3760, 'SUN397': 19850}[task] or len(training.classes) != model.classification_head.out_features:
        raise RuntimeError(f'{task} dataset counts or class count mismatch')
    if task == 'DTD' and [x.replace('_', ' ') for x in training.classes] != metadata['classnames']:
        raise RuntimeError('DTD head label order mismatch')
    train, val = torch.utils.data.random_split(training, [count - count // 10, count // 10],
                                                generator=torch.Generator().manual_seed(0))
    dataset = existing.EpochData(train, seed)
    fit = torch.utils.data.Subset(evaluation, train.indices)
    validation = torch.utils.data.Subset(evaluation, val.indices)
    order = torch.randperm(len(fit), generator=torch.Generator().manual_seed(901)).tolist()
    reserved = 144 if task == 'DTD' else 288
    return SimpleNamespace(existing=existing, model=model, found=found, base=base, dataset=dataset,
                           fit=fit, validation=validation, train_ids=order[reserved:], fit_ids=order[:48],
                           audit_ids=order[48:96], ec_ids=order[96:reserved], base_sha256=existing.sha(base_path),
                           head_sha256=metadata['sha256'], predict=predictor(model, found), task=task)


def probe(data, ids):
    samples = [data[i] for i in ids]
    return torch.stack([x for x, _ in samples]).cuda(), torch.tensor([y for _, y in samples], device='cuda')


@torch.no_grad()
def score(task, theta, data):
    loss, hits, count = 0., 0, 0
    for x, y in torch.utils.data.DataLoader(data, batch_size=64, num_workers=4, pin_memory=True):
        z, y = task.predict(theta, x.cuda()), y.cuda()
        loss += float(F.cross_entropy(z, y, reduction='sum'))
        hits += int((z.argmax(-1) == y).sum())
        count += len(y)
    return dict(ce=loss / count, accuracy=hits / count, n=count)


def train(args):
    task = load_task(args.root, args.task, args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    params = [p for _, p in task.found]
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=.1, fused=True)
    x, _ = probe(task.fit, task.fit_ids[:24] + task.audit_ids[:24])
    metadata = dict(task=args.task, seed=args.seed, epochs=args.epochs, train=len(task.train_ids),
                    base_sha256=task.base_sha256, head_sha256=task.head_sha256, names=[n for n, _ in task.found],
                    probe_indices=task.fit_ids[:24] + task.audit_ids[:24], validations=[])

    @torch.no_grad()
    def checkpoint(step):
        theta = flat(task.found)
        probability = task.predict(theta, x).double().softmax(-1).cpu()
        save(output / f'checkpoint_{step:04d}.pt', dict(weights=theta.cpu(), probability=probability, step=step))
        return theta

    step = 0
    theta = checkpoint(0)
    initial = score(task, theta, task.validation)
    metadata['validations'].append(dict(epoch=0, step=step, **initial))
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        loader = batches(task.dataset, task.train_ids, args.seed, epoch)
        for i, (inputs, y, _) in enumerate(loader, 1):
            inputs, y = inputs.cuda(non_blocking=True), y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            for j in range(0, len(y), 16):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = F.cross_entropy(task.model(inputs[j:j + 16]).float(), y[j:j + 16], reduction='sum') / len(y)
                loss.backward()
            optimizer.step()
            step += 1
            if i == len(loader) or i % max(1, len(loader) // 4) == 0:
                theta = checkpoint(step)
        row = dict(epoch=epoch, step=step, **score(task, theta, task.validation))
        metadata['validations'].append(row)
        if row['ce'] < best:
            best = row['ce']
            metadata['selected'] = row
        task.existing.write(output / 'progress.json', metadata)
        print(row, flush=True)
    task.existing.write(output / 'complete.json', metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--task', choices=('DTD', 'SUN397'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('At least one complete epoch is required')
    train(args)
