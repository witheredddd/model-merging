"""Sequential full-visual CLIP experts, selected on validation accuracy."""

import argparse
import copy
import fcntl
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import ImageFile
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder, MNIST

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'scripts/tfpml')]

import dtd_grassmann4 as existing
from runtime import build_model
from src.datasets.cars import PytorchStanfordCars
from src.datasets.gtsrb import PyTorchGTSRB
from src.datasets.resisc45 import RESISC45Dataset
from src.datasets.svhn import PyTorchSVHN
from src.utils.utils import cosine_lr

TASKS = {
    'Cars': (35, 7330, 814), 'DTD': (76, 3384, 376), 'EuroSAT': (12, 16200, 5400),
    'GTSRB': (11, 23976, 2664), 'MNIST': (5, 55000, 5000), 'RESISC45': (15, 17010, 1890),
    'SVHN': (4, 68257, 5000), 'SUN397': (14, 17865, 1985),
}
BATCH_SIZE = 128
SEED = 42
BASE = ROOT / 'models/checkpoints/ViT-B-16/pretrained.pt'
HEADS = ROOT / 'results/functional_geometry_v2_no_expert/assets/heads'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    temp = path.with_suffix('.pt.tmp')
    torch.save(value, temp)
    temp.replace(path)


def splits(task, train_transform=None, val_transform=None):
    root = ROOT / 'datasets'
    if task == 'EuroSAT':
        train = ImageFolder(root / 'EuroSAT_splits/train', transform=train_transform)
        val = ImageFolder(root / 'EuroSAT_splits/val', transform=val_transform)
        if train.class_to_idx != val.class_to_idx:
            raise ValueError(f'{task} train and validation class mappings differ')
        return train, val

    datasets = {
        'Cars': lambda: PytorchStanfordCars(str(root), 'train'),
        'DTD': lambda: ImageFolder(root / 'dtd/train'),
        'GTSRB': lambda: PyTorchGTSRB(str(root), split='train'),
        'MNIST': lambda: MNIST(str(root), train=True),
        'RESISC45': lambda: RESISC45Dataset(str(root), split='train'),
        'SVHN': lambda: PyTorchSVHN(str(root / 'svhn'), split='train'),
        'SUN397': lambda: ImageFolder(root / 'sun397/train'),
    }
    train = datasets[task]()
    val = copy.copy(train)
    transform_attr = 'transforms' if task == 'RESISC45' else 'transform'
    setattr(train, transform_attr, train_transform)
    setattr(val, transform_attr, val_transform)
    size = len(train)
    n_val = min(size // 10, 5000)
    fit, held = random_split(train, [size - n_val, n_val], generator=torch.Generator().manual_seed(0))
    return fit, Subset(val, held.indices)


def preflight(task):
    epochs, n_train, n_val = TASKS[task]
    train, val = splits(task)
    if (len(train), len(val)) != (n_train, n_val):
        raise ValueError(f'{task}: unexpected split sizes {(len(train), len(val))}')
    head = read(HEADS / f'{task}.json')
    if head['kind'] != 'original_pipeline_zero_shot' or existing.sha(HEADS / f'{task}.pt') != head['sha256']:
        raise ValueError(f'{task}: unexpected classification head')
    train_set = train.dataset if isinstance(train, Subset) else train
    if hasattr(train_set, 'classes') and len(train_set.classes) != head['classes']:
        raise ValueError(f'{task}: classification head size does not match dataset')
    ref = read(ROOT / 'results/expert_references' / f'{task}.json')
    reference_split = 'validation' if 'validation' in ref else 'test'
    if reference_split == 'validation' and ref['validation']['n'] != n_val:
        raise ValueError(f'{task}: validation reference size does not match dataset')
    accuracy = ref[reference_split]['accuracy']
    validation = 'reconstructed_eurosat_val' if task == 'EuroSAT' else 'train_holdout_seed_0'
    return dict(task=task, status='queued', epoch_budget=epochs, train_n=n_train, validation_n=n_val,
                validation_split=validation, head_sha256=head['sha256'], reference_split=reference_split,
                reference_accuracy=accuracy, threshold_99=.99 * accuracy, threshold_98=.98 * accuracy)


def loader(data, workers, batch_size, seed, shuffle=False):
    return DataLoader(data, batch_size=batch_size, num_workers=workers, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), pin_memory=True)


def train(task, row, output, workers, base_hash, publish, trace=None, stop_epoch=None):
    folder = output / task
    folder.mkdir(exist_ok=True)
    best_path, resume_path = folder / 'best.pt', folder / 'resume.pt'
    existing.seed_all(SEED)
    head = existing.ClassificationHead.load(str(HEADS / f'{task}.pt'))
    encoder, model, found, base_flat = build_model(existing, head)
    del base_flat
    visual = encoder.model.visual
    model.requires_grad_(False)
    visual.requires_grad_(True)
    params = list(visual.parameters())
    train_data, val_data = splits(task, encoder.train_preprocess, encoder.val_preprocess)
    train_data = existing.EpochData(train_data, SEED)
    val_loader = loader(val_data, workers, BATCH_SIZE, SEED)
    optimizer = torch.optim.AdamW(params, lr=1e-5, weight_decay=.1, fused=True)
    steps = math.ceil(len(train_data) / BATCH_SIZE)
    schedule = cosine_lr(optimizer, 1e-5, 500, row['epoch_budget'] * steps)
    provenance = dict(seed=SEED, base_sha256=base_hash, head_sha256=row['head_sha256'],
                      task=task, visual_scope='full', matrix_names=[name for name, _ in found],
                      validation_split=row['validation_split'], batch_size=BATCH_SIZE, train_precision='bf16_autocast',
                      validation_precision='fp32', optimizer='AdamW', lr=1e-5, weight_decay=.1,
                      warmup_steps=500, grad_clip_norm=1., epoch_budget=row['epoch_budget'])
    history, best = [], None
    if resume_path.exists():
        resumed = torch.load(resume_path, map_location='cpu', weights_only=True)
        if resumed['provenance'] != provenance:
            raise ValueError(f'{task}: resume provenance mismatch')
        visual.load_state_dict(resumed['visual'])
        optimizer.load_state_dict(resumed['optimizer'])
        history, best = resumed['history'], resumed['best']
        del resumed

    if trace is not None:
        trace.start(model, visual, optimizer)
    epochs = row['epoch_budget'] if stop_epoch is None else stop_epoch
    for epoch in range(len(history), epochs):
        if best and best['accuracy'] >= row['threshold_99']:
            break
        started = time.monotonic()
        train_data.epoch = epoch
        model.train()
        for i, (images, labels, _) in enumerate(loader(train_data, workers, BATCH_SIZE, SEED + epoch, True)):
            step = epoch * steps + i
            schedule(step)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images.cuda(non_blocking=True))
                loss = F.cross_entropy(logits.float(), labels.cuda(non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            if trace is not None:
                trace.record(epoch + 1, step + 1, epoch_end=(i + 1 == steps))
            if i == 0 or (i + 1) % 25 == 0:
                print(f'{task} epoch {epoch + 1}/{row["epoch_budget"]} step {i + 1}/{steps} loss={loss.item():.5f}', flush=True)

        metrics = existing.evaluate(model, val_loader)
        result = dict(epoch=epoch + 1, seconds=round(time.monotonic() - started, 1), **metrics)
        history.append(result)
        current = dict(provenance=provenance, visual=visual.state_dict(), validation=result)
        if best is None or (result['accuracy'], -result['ce']) > (best['accuracy'], -best['ce']):
            best = result
            save(best_path, current)
        save(resume_path, dict(**current, optimizer=optimizer.state_dict(), history=history, best=best))
        existing.write(folder / 'validation.json', dict(provenance=provenance, history=history, best=best))
        row.update(epoch=epoch + 1, best=best)
        publish()
        print(f'{task} validation epoch {epoch + 1}: accuracy={result["accuracy"]:.6f}, best={best["accuracy"]:.6f}', flush=True)

    ratio = best['accuracy'] / row['reference_accuracy']
    gate = 'target_99' if ratio >= .99 else 'floor_98' if ratio >= .98 else 'below_floor'
    row.update(status='complete', gate=gate, ratio=ratio, best=best, epoch=len(history),
               qualified=gate != 'below_floor' and row['reference_split'] == 'validation',
               best_checkpoint=str(best_path), checkpoint_sha256=existing.sha(best_path))
    existing.write(folder / 'complete.json', row)
    resume_path.unlink(missing_ok=True)
    publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with (output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base_hash = existing.sha(BASE)
        rows = [preflight(task) for task in TASKS]

        def publish():
            existing.write(output / 'queue.json', dict(pid=os.getpid(), base_sha256=base_hash, tasks=rows))

        publish()
        print(f'Validated dataset sizes and head hashes for {len(rows)} tasks', flush=True)
        for row in rows:
            task = row['task']
            done = output / task / 'complete.json'
            if done.exists():
                completed = read(done)
                if existing.sha(Path(completed['best_checkpoint'])) != completed['checkpoint_sha256']:
                    raise ValueError(f'{task}: completed checkpoint hash mismatch')
                row.update(completed)
                publish()
                continue

            # Match the existing decoder audit for the recoverable SUN397 JPEG.
            ImageFile.LOAD_TRUNCATED_IMAGES = task == 'SUN397'
            row['status'] = 'running'
            publish()
            try:
                train(task, row, output, args.workers, base_hash, publish)
            except Exception as error:
                row.update(status='failed', error=str(error))
                publish()
                raise
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
