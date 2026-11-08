"""Test a discovered cross-block relation by intervening on a frozen DTD decoder."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from collect_dtd_updates import load_experiment
from study_dtd_quotient import predictor


def interval(values, seed):
    rng = torch.Generator().manual_seed(seed)
    sampled = values[torch.randint(len(values), (4000, len(values)), generator=rng)].mean(dim=1)
    return dict(mean=float(values.mean()), interval95=sampled.quantile(torch.tensor([.025, .975])).tolist())


@torch.no_grad()
def audit(args):
    _, model, found, base, _, _, splits, ids, base_hash = load_experiment(args.root, 45)
    predict = predictor(model, found)
    frozen = torch.load(args.study / 'decoder.pt', weights_only=True, map_location='cuda')
    if frozen['base_sha256'] != base_hash or frozen['names'] != [n for n, _ in found]:
        raise RuntimeError('Frozen decoder provenance mismatch')
    study = [json.loads(line) for line in (args.study / 'metrics.jsonl').read_text().splitlines()]
    discovery = next(r for r in study if r['event'] == 'coupling' and r['region'] == 1)
    report = dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  test='Paired loss contrasts after removing blocks of the learned decoder')
    if not discovery['edges']:
        report['tested'] = False
        report['reason'] = 'No replicated edge reached the predeclared response threshold'
        return report
    edge = max(discovery['edges'], key=lambda e: min(map(abs, e['cosine'])))
    report.update(tested=True, layers=edge['layers'], selection_cosines=edge['cosine'])
    update = frozen['code'] @ frozen['modes']
    expert = base + update
    by_layer = {}
    offset = 0
    for name, p in found:
        layer = int(name.split('resblocks.')[1].split('.')[0])
        by_layer.setdefault(layer, []).append(slice(offset, offset + p.numel()))
        offset += p.numel()

    def removed(layers):
        changed = expert.clone()
        for layer in layers:
            for section in by_layer[layer]:
                changed[section] = base[section]
        return changed

    i, j = edge['layers']
    variants = [expert, removed([i]), removed([j]), removed([i, j])]
    report['results'] = []
    # These ranges lie outside every probe slice in study_dtd_quotient.py.
    for k, start in enumerate((16, 64)):
        samples = [splits.train_dataset[index] for index in ids[start:start + 24]]
        images = torch.stack([x for x, _ in samples]).cuda()
        labels = torch.tensor([y for _, y in samples], device='cuda')
        full, first, second, joint = [F.cross_entropy(predict(w, images), labels, reduction='none').cpu()
                                     for w in variants]
        report['results'].append(dict(n=len(labels), first_drop=interval(first - full, k),
                                      second_drop=interval(second - full, k), joint_drop=interval(joint - full, k),
                                      interaction=interval(joint - first - second + full, k)))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    output = args.study / 'intervention.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
