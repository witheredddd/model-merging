"""Repeat the DTD expert recipe while recording full visual and probability states."""

import argparse
import json
from pathlib import Path

import torch

from train_eight_expert_queue import BASE, ROOT, existing, preflight, read, save, splits, train


class Trace:
    def __init__(self, folder, interval):
        self.folder = folder
        self.interval = interval
        self.rows = []

    def start(self, model, visual, optimizer):
        self.model, self.visual, self.optimizer = model, visual, optimizer
        data, _ = splits('DTD', model.val_preprocess, model.val_preprocess)
        ids = torch.randperm(len(data), generator=torch.Generator().manual_seed(20261109))[:64]
        self.probe_ids = [[data.indices[i] for i in group] for group in ids.split(32)]
        batches = [[data[i] for i in group] for group in ids.split(32)]
        self.images = [torch.stack([x for x, _ in batch]).cuda() for batch in batches]
        self.labels = [torch.tensor([y for _, y in batch]) for batch in batches]
        self.previous = self.previous_epoch = None
        self.record(0, 0, epoch_end=True)

    @torch.no_grad()
    def record(self, epoch, step, epoch_end):
        if not epoch_end and step % self.interval:
            return
        was_training = self.model.training
        self.model.eval()
        probabilities = [self.model(x).double().softmax(-1).cpu() for x in self.images]
        self.model.train(was_training)
        weights = {k: v.detach().cpu() for k, v in self.visual.state_dict().items()}
        delta = None if self.previous is None else {k: v - self.previous[k] for k, v in weights.items()}
        path = self.folder / f'step_{step:04d}.pt'
        save(path, dict(epoch=epoch, step=step, visual=weights, delta_visual=delta, probability=probabilities))
        row = dict(epoch=epoch, step=step, checkpoint=path.name, sha256=existing.sha(path), epoch_end=epoch_end)
        if epoch_end:
            epoch_path = self.folder / f'epoch_{epoch:02d}_adamw.pt'
            epoch_delta = None if self.previous_epoch is None else {k: v - self.previous_epoch[k] for k, v in weights.items()}
            save(epoch_path, dict(epoch=epoch, step=step, adamw=self.optimizer.state_dict(), delta_visual=epoch_delta,
                                  torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state()))
            row.update(epoch_state=epoch_path.name, epoch_state_sha256=existing.sha(epoch_path))
            self.previous_epoch = weights
        self.previous = weights
        self.rows.append(row)
        existing.write(self.folder / 'trace.json', dict(interval=self.interval, raw_train_indices=self.probe_ids,
                       labels=[v.tolist() for v in self.labels], rows=self.rows))
        print(f'trace saved: epoch={epoch}, step={step}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=25, choices=range(20, 51))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    folder = output / 'trace'
    folder.mkdir()
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = read(args.reference / 'complete.json')
    base_hash = existing.sha(BASE)
    trace = Trace(folder, args.interval)
    # Keep the original 76-epoch schedule, but record only through the selected eighth epoch.
    row = preflight('DTD')
    train('DTD', row, output, 8, base_hash, lambda: None, trace=trace, stop_epoch=manifest['best']['epoch'])
    original_path = args.reference / 'best.pt'
    if existing.sha(original_path) != manifest['checkpoint_sha256']:
        raise ValueError('Reference checkpoint hash mismatch')
    original = torch.load(original_path, map_location='cpu', weights_only=True, mmap=True)
    replayed = torch.load(folder / trace.rows[-1]['checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    exact = all(torch.equal(v, replayed['visual'][k]) for k, v in original['visual'].items())
    error = max(float((v - replayed['visual'][k]).abs().max()) for k, v in original['visual'].items())
    existing.write(output / 'comparison.json', dict(source=manifest, replay=row, base_sha256=base_hash,
                   final_step=replayed['step'], exact_tensor_match=exact, max_abs_difference=error))
    print(json.dumps(dict(final_step=replayed['step'], exact_tensor_match=exact, max_abs_difference=error)), flush=True)


if __name__ == '__main__':
    main()
