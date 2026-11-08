"""Lock the minimum-validation-CE task continuation and evaluate it once."""

import argparse
import json
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder

from collect_dtd_updates import load_experiment, save


def run(args):
    existing, model, found, base, _, validation, splits, _, base_hash = load_experiment(args.root, 46)
    rows = [json.loads(line) for line in (args.run / 'metrics.jsonl').read_text().splitlines()]
    if rows[-1]['event'] != 'complete':
        raise RuntimeError('Continuation must finish before selection')
    candidates = [row for row in rows if row['event'] == 'replay' and row['method'] == 'task_fisher']
    selected = min(candidates, key=lambda r: r['ce'])
    bundle = torch.load(args.run / 'continuation.pt', weights_only=True, map_location='cpu', mmap=True)
    if bundle['base_sha256'] != base_hash or bundle['names'] != [n for n, _ in found]:
        raise RuntimeError('Continuation provenance mismatch')
    code = bundle['code'].clone()
    code[-1] = selected['code'][0]
    bundle['code'] = code
    theta = base + code.cuda() @ bundle['modes'].cuda()
    existing.assign_flat(found, theta)
    validation_check = existing.evaluate(model, validation)
    if abs(validation_check['ce'] - selected['ce']) > 1e-5 or validation_check['accuracy'] != selected['accuracy']:
        raise RuntimeError('Selected checkpoint did not reconstruct validation results')
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'selected.pt'
    save(path, bundle)
    passed = selected['accuracy'] >= .80
    lock = dict(epoch=selected['epoch'], validation=validation_check, passed=passed,
                frozen_sha256=existing.sha(path), base_sha256=base_hash)
    existing.write(args.output / 'selection.json', lock)
    print(lock, flush=True)
    if args.test and passed:
        provenance = json.loads((args.root / 'datasets/DTD_test_provenance.json').read_text())
        dataset = ImageFolder(args.root / 'datasets/dtd/val', transform=model.image_encoder.val_preprocess)
        if dataset.classes != splits.train_dataset.dataset.classes or len(dataset) != provenance['n']:
            raise RuntimeError('DTD test split does not match its provenance')
        score = existing.evaluate(model, existing.loader(dataset))
        existing.write(args.output / 'test.json', dict(**score, test_provenance=provenance,
                       frozen_sha256=lock['frozen_sha256'], historically_exposed=True))
        print(score, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--test', action='store_true')
    run(parser.parse_args())
