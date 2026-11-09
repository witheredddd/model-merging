"""Train matched DTD AdamW controls and rolling-history innovation gates."""

import argparse
from collections import deque
from contextlib import contextmanager
import fcntl
import gc
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import ImageFolder

from forecast_dtd_residual_geometry import response
from replay_dtd_functional_residuals import solve
from run_dtd_carrier_diagnostics import Operator
from train_eight_expert_queue import (
    BASE, BATCH_SIZE, HEADS, ROOT, TASKS, build_model, cosine_lr, existing, loader, preflight, read, save, splits,
)


EPOCHS = 8
HISTORY = 5
INTERVAL = 25
LEARNING_RATE = 1e-5
DECAY = .1
BETAS = (1., .875, .75)
CE_MARGIN = 1e-5
# One-sided 95% paired t bound, with 32 examples in each probe group.
T_CRITICAL = 1.695519
ARMS = ('adamw', 'fisher', 'euclidean')


@contextmanager
def probe_mode(model):
    training = model.training
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    model.eval()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
        model.train(training)


def probe_sets(encoder, trace):
    fit, _ = splits('DTD', encoder.val_preprocess, encoder.val_preprocess)
    indices = read(trace / 'trace.json')['raw_train_indices']
    used = {i for group in indices for i in group}
    remainder = [i for i in fit.indices if i not in used]
    order = torch.randperm(len(remainder), generator=torch.Generator().manual_seed(431916))[:32]
    geometry = [remainder[i] for i in order]
    by_raw = {raw: i for i, raw in enumerate(fit.indices)}
    groups = []
    for group in [*indices, geometry]:
        assert len(group) == 32 and len(set(group)) == 32
        rows = [fit[by_raw[i]] for i in group]
        groups.append((torch.stack([x for x, _ in rows]).cuda(), torch.tensor([y for _, y in rows], device='cuda')))
    assert len(used | set(geometry)) == 96
    return torch.cat([x for x, _ in groups[:2]]), torch.cat([y for _, y in groups[:2]]), groups[2][0], [*indices, geometry]


def paired_ce(old, new):
    a = (old - new).reshape(2, 32)
    means = a.mean(1)
    lower = means - T_CRITICAL * a.std(1, unbiased=True) / math.sqrt(a.shape[1])
    return dict(mean_gain=means.cpu().tolist(), lower_95=lower.cpu().tolist(),
                passed=bool((lower > CE_MARGIN).all()))


class ShadowAdamW:
    """Only shadow tensors are written by the fused optimizer before selection."""

    def __init__(self, op):
        self.op = op
        self.shadows = [p.detach().clone() for p in op.params]
        self.optimizer = torch.optim.AdamW(self.shadows, lr=LEARNING_RATE, weight_decay=DECAY, fused=True)

    def clear(self):
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.op.params:
            p.grad = None

    @torch.no_grad()
    def propose(self):
        for shadow, actual in zip(self.shadows, self.op.params):
            shadow.copy_(actual)
            shadow.grad = actual.grad
        self.optimizer.step()
        shrink = 1. - self.optimizer.param_groups[0]['lr'] * DECAY
        decay_point = tuple(p.detach() * shrink for p in self.op.params)
        raw = torch.cat([(shadow - origin).flatten() for shadow, origin in zip(self.shadows, decay_point)])
        return decay_point, raw

    @torch.no_grad()
    def parameters(self, innovation, beta):
        if beta == 1.:
            return self.shadows
        return [shadow - (1. - beta) * part for shadow, part in zip(self.shadows, self.op.parts(innovation))]

    @torch.no_grad()
    def apply(self, selected, decay_point, window):
        for param, weight, origin, accumulator in zip(self.op.params, selected, decay_point, self.op.parts(window)):
            # Accumulate the actual accepted FP32 displacement, excluding the separate decay.
            accumulator.add_(weight - origin)
            param.copy_(weight)


@torch.no_grad()
def loss_at(op, weights, images, labels):
    logits = torch.func.functional_call(op.model, dict(zip(op.names, weights)), (images,)).double()
    return F.cross_entropy(logits, labels, reduction='none')


def functional_rows(op, images, bank, update):
    return np.stack([*(response(op, images, row) for row in bank), response(op, images, update)])


def functional_explained(rows, executed):
    joined = np.vstack((rows[:-1], executed))
    alpha, _ = solve(joined @ joined.T)
    return float(1. - np.linalg.norm(executed - alpha @ rows[:-1]) ** 2 / np.linalg.norm(executed) ** 2)


def choose(op, optimizer, arm, bank, raw, gate_x, gate_y, fisher_x, boundary):
    result = dict(beta=1.)
    if bank is None or (arm == 'adamw' and not boundary):
        return optimizer.shadows, result
    with probe_mode(op.model):
        rows = functional_rows(op, fisher_x, bank, raw) if arm == 'fisher' or boundary else None
        if boundary:
            result['candidate_function_explained'] = functional_explained(rows, rows[-1])
        if arm == 'adamw':
            result['executed_function_explained'] = result['candidate_function_explained']
            return optimizer.shadows, result
        if arm == 'fisher':
            alpha, stats = solve(rows @ rows.T)
        else:
            joint = torch.vstack((bank, raw))
            alpha, stats = solve((joint @ joint.T).double().cpu().numpy())
        innovation = raw - torch.as_tensor(alpha, dtype=raw.dtype, device=raw.device) @ bank
        original = loss_at(op, optimizer.shadows, gate_x, gate_y)
        checks = []
        beta = 1.
        for trial in BETAS[1:]:
            candidate = loss_at(op, optimizer.parameters(innovation, trial), gate_x, gate_y)
            record = paired_ce(original, candidate)
            checks.append(dict(beta=trial, **record))
            if beta == 1. and record['passed']:
                beta = trial
        selected = optimizer.parameters(innovation, beta)
        result.update(beta=beta, alpha=alpha.tolist(), solver=stats, checks=checks,
                      original_probe_ce=original.reshape(2, 32).mean(1).cpu().tolist(),
                      parameter_innovation_energy_fraction=float(innovation.square().sum() / raw.square().sum()))
        if boundary:
            selected_response = beta * rows[-1] + (1. - beta) * (alpha @ rows[:-1])
            result['executed_function_explained'] = functional_explained(rows, selected_response)
        if beta < 1.:
            # Diagnostic only. Compare the selected combination against a pure
            # rescaling of the raw adaptive update at the same adaptive norm.
            adaptive = raw - (1. - beta) * innovation
            scalar = float(adaptive.norm() / raw.norm())
            scalar_ce = loss_at(op, optimizer.parameters(raw, scalar), gate_x, gate_y)
            selected_ce = loss_at(op, selected, gate_x, gate_y)
            result.update(norm_matched_scalar=scalar, against_scalar=paired_ce(scalar_ce, selected_ce))
    return selected, result


def train_arm(seed, arm, output, base_hash, trace, reference):
    output.mkdir(parents=True, exist_ok=True)
    complete = output / 'complete.json'
    final = output / 'epoch08.pt'
    if complete.exists():
        result = read(complete)
        if existing.sha(final) != result['checkpoint_sha256']:
            raise ValueError(f'Finished checkpoint changed at {output}')
        return result
    existing.seed_all(seed)
    baseline = preflight('DTD')
    head = existing.ClassificationHead.load(str(HEADS / 'DTD.pt')).cuda()
    encoder, model, _, flat = build_model(existing, head)
    del flat
    visual = encoder.model.visual
    model.requires_grad_(False)
    visual.requires_grad_(True)
    op = Operator(model, visual)
    shadow = ShadowAdamW(op)
    train, val = splits('DTD', encoder.train_preprocess, encoder.val_preprocess)
    data = existing.EpochData(train, seed)
    validation = loader(val, 8, BATCH_SIZE, seed)
    steps = math.ceil(len(data) / BATCH_SIZE)
    schedule = cosine_lr(shadow.optimizer, LEARNING_RATE, 500, TASKS['DTD'][0] * steps)
    gate_x, gate_y, fisher_x, ids = probe_sets(encoder, trace)
    provenance = dict(seed=seed, arm=arm, base_sha256=base_hash, head_sha256=baseline['head_sha256'],
                      train_n=len(train), validation_n=len(val), batch_size=BATCH_SIZE, epochs=EPOCHS,
                      scheduler_steps=TASKS['DTD'][0] * steps, history=HISTORY, interval=INTERVAL,
                      lr=LEARNING_RATE, decay=DECAY, betas=list(BETAS), ce_margin=CE_MARGIN, t_critical=T_CRITICAL,
                      probe_groups=ids, trace_sha256=existing.sha(trace / 'trace.json'))
    resume_path = output / 'resume.pt'
    bank, ends = deque(maxlen=HISTORY), deque(maxlen=HISTORY)
    window = torch.zeros(sum(op.sizes), device='cuda')
    epochs, records = [], []
    if resume_path.exists():
        resumed = torch.load(resume_path, map_location='cpu', weights_only=True, mmap=True)
        if resumed['provenance'] != provenance:
            raise ValueError(f'Resume provenance differs at {output}')
        visual.load_state_dict(resumed['visual'])
        shadow.optimizer.load_state_dict(resumed['optimizer'])
        bank.extend(v.cuda() for v in resumed['bank'])
        ends.extend(resumed['ends'])
        epochs, records = resumed['epochs'], resumed['records']
        del resumed
    packed = torch.stack(list(bank)) if len(bank) == HISTORY else None
    for epoch in range(len(epochs), EPOCHS):
        started = time.monotonic()
        data.epoch = epoch
        model.train()
        for i, (images, labels, _) in enumerate(loader(data, 8, BATCH_SIZE, seed + epoch, True)):
            step = epoch * steps + i + 1
            boundary = step % INTERVAL == 0 or i + 1 == steps
            schedule(step - 1)
            shadow.clear()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images.cuda(non_blocking=True))
                loss = F.cross_entropy(logits.float(), labels.cuda(non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(op.params, 1., error_if_nonfinite=True)
            decay_point, raw = shadow.propose()
            selected, row = choose(op, shadow, arm, packed, raw, gate_x, gate_y, fisher_x, boundary)
            shadow.apply(selected, decay_point, window)
            records.append(dict(step=step, epoch=epoch + 1, boundary=boundary, history_end_steps=list(ends),
                                training_ce=float(loss.detach()), lr=shadow.optimizer.param_groups[0]['lr'], **row))
            if boundary:
                bank.append(window)
                ends.append(step)
                window = torch.zeros_like(window)
                packed = torch.stack(list(bank)) if len(bank) == HISTORY else None
                print(f'seed {seed} {arm} step {step}/{EPOCHS * steps}, beta={row["beta"]}', flush=True)
        value = existing.evaluate(model, validation)
        epochs.append(dict(epoch=epoch + 1, seconds=time.monotonic() - started, **value))
        save(resume_path, dict(provenance=provenance, visual=visual.state_dict(), optimizer=shadow.optimizer.state_dict(),
                               bank=list(bank), ends=list(ends), epochs=epochs, records=records))
        existing.write(output / 'progress.json', dict(provenance=provenance, epochs=epochs,
                       gate_counts={str(beta): sum(r['beta'] == beta for r in records) for beta in BETAS}))
        print(f'seed {seed} {arm} epoch {epoch + 1}: val acc={value["accuracy"]:.6f}, CE={value["ce"]:.5f}', flush=True)
    exact = None
    if seed == 42 and arm == 'adamw':
        state = torch.load(reference, weights_only=True, mmap=True, map_location='cpu')['visual']
        exact = all(torch.equal(v.cpu(), state[k]) for k, v in visual.state_dict().items())
        if not exact:
            raise ValueError('Shadow AdamW did not reproduce the seed-42 historical endpoint')
    save(final, dict(provenance=provenance, visual=visual.state_dict(), validation=epochs[-1]))
    result = dict(provenance=provenance, epochs=epochs, records=records, exact_historical_baseline=exact,
                  checkpoint_sha256=existing.sha(final))
    existing.write(complete, result)
    resume_path.unlink()
    return result


def test_all(output, arms, expected):
    if not arms:
        return
    import pyarrow.parquet as parquet

    encoder, model, _, flat = build_model(existing, existing.ClassificationHead.load(str(HEADS / 'DTD.pt')).cuda())
    del flat
    test = ImageFolder(ROOT / 'datasets/dtd/val', transform=encoder.val_preprocess)
    fit, _ = splits('DTD')
    if len(test) != expected['n'] or test.class_to_idx != fit.dataset.class_to_idx:
        raise ValueError('Historical test layout or class mapping differs')
    materialized = {int(Path(path).stem.removeprefix('test_')): (Path(path), label) for path, label in test.samples}
    index = 0
    for source in expected['files']:
        file = parquet.ParquetFile(ROOT / 'datasets/_downloads/tanganke__dtd' / source['path'])
        for batch in file.iter_batches(batch_size=64, columns=['image', 'label']):
            for row in batch.to_pylist():
                path, label = materialized[index]
                if label != row['label'] or path.read_bytes() != row['image']['bytes']:
                    raise ValueError(f'Historical test image or label differs at row {index}')
                index += 1
    if index != len(test) or len(materialized) != len(test):
        raise ValueError('Historical test source does not cover the materialized split')
    iterator = loader(test, 8, BATCH_SIZE, 42)
    visual = encoder.model.visual
    for folder in arms:
        result_path = folder / 'historical_test.json'
        manifest = read(folder / 'complete.json')
        if result_path.exists():
            if read(result_path)['checkpoint_sha256'] != manifest['checkpoint_sha256']:
                raise ValueError('Historical test result belongs to a different checkpoint')
            continue
        state = torch.load(folder / 'epoch08.pt', weights_only=True, mmap=True, map_location='cpu')
        visual.load_state_dict(state['visual'])
        value = existing.evaluate(model, iterator)
        existing.write(result_path, dict(checkpoint_sha256=manifest['checkpoint_sha256'],
                       split='historical_dtd_test', provenance=expected, **value))
        print(f'{folder.parent.name}/{folder.name}: historical test {value["accuracy"]:.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with (args.output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base_hash = existing.sha(BASE)
        arms = []
        for seed in (42, 43, 44):
            results = {}
            for arm in ARMS:
                folder = args.output / f'seed{seed}' / arm
                results[arm] = train_arm(seed, arm, folder, base_hash, args.trace, args.reference)
                arms.append(folder)
                gc.collect()
                torch.cuda.empty_cache()
            counts = {arm: sum(r['beta'] < 1. for r in value['records']) for arm, value in results.items()}
            existing.write(args.output / f'seed{seed}' / 'gate_counts.json', counts)
            if seed == 42 and not counts['fisher']:
                print('Seed 42 Fisher gate never engaged; replication condition was not met', flush=True)
                break
        # Fixed checkpoints and replication decision exist before any test scoring.
        existing.write(args.output / 'training_complete.json', dict(arms=[str(p.relative_to(args.output)) for p in arms]))
        expected = read(ROOT / 'datasets/DTD_test_provenance.json')
        for row in expected['files']:
            path = ROOT / 'datasets/_downloads/tanganke__dtd' / row['path']
            if existing.sha(path) != row['sha256']:
                raise ValueError('Historical DTD test source hash mismatch')
        test_all(args.output, arms, expected)
        existing.write(args.output / 'complete.json', dict(arms=[str(p.relative_to(args.output)) for p in arms]))


if __name__ == '__main__':
    main()
