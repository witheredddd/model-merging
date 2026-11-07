"""SUN397 paired-protocol rerun with a damped parameter-metric step."""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import ImageFile

from metric_step import check_math, step_

ROOT = Path('/root/autodl-tmp/tsv')
REFERENCE = ROOT / 'results/SUN397_rank20_pair'
OUT = ROOT / 'results/SUN397_rank20_parameter_metric'
SCRATCH = Path('/dev/shm/tsv_sun397_rank20_parameter_metric')
sys.path.insert(0, str(ROOT / 'scripts/functional_geometry_sun397_v2_3'))
import train as previous

u = previous.u
RANK = 20
EPOCHS = 20
SEED = 42
RADIUS = .10
DAMPING = .05


def write(name, value):
    previous.write(OUT / name, value)


def status(stage, **details):
    write('status.json', dict(stage=stage, time=time.time(), pid=os.getpid(), **details))


def main(micro):
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(mode=0o700, exist_ok=True)
    assert SCRATCH.resolve() == SCRATCH
    status('loading')
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    u.ROOT = ROOT
    check_math()
    u.seed_all(SEED)
    head_path = ROOT / 'models/checkpoints/ViT-B-16/head_SUN397Val.pt'
    encoder, model, selected, base = previous.runtime.build_model(u, u.ClassificationHead.load(str(head_path)))
    assert len(base) == 84934656 and len(selected) == 48 and model.classification_head.out_features == 397
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in selected:
        parameter.requires_grad_(True)
    frozen = u.frozen_hash(model)
    examples, loader, validation, view = previous.data(encoder, SEED)
    split = dict(train=view.train_dataset.indices, validation=view.test_dataset.indices, classes=view.classnames, seed=0)
    reference = json.loads((REFERENCE / 'rank20_angle002/config.json').read_text())
    previous_protocol = reference['protocol']
    assert hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest() == previous_protocol['split_sha256']
    assert u.sha(head_path) == previous_protocol['head_sha256']
    assert u.sha(ROOT / 'models/checkpoints/ViT-B-16/pretrained.pt') == previous_protocol['base_sha256']
    assert [dict(name=n, shape=list(p.shape)) for n, p in selected] == previous_protocol['parameter_layout']
    protocol = dict(task='SUN397', rank=RANK, epochs=EPOCHS, seed=SEED, effective_batch=64,
                    init_id=reference['init_id'], reference_protocol=previous_protocol,
                    max_common_lr=.01, parameter_step_radius=RADIUS, coordinate_damping=DAMPING,
                    selection='minimum validation CE', test_policy='one locked model',
                    step_source_sha256=u.sha(Path(__file__).with_name('metric_step.py')))
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    config = OUT / 'config.json'
    if config.exists():
        assert json.loads(config.read_text())['identity'] == identity
    else:
        write('config.json', dict(identity=identity, protocol=protocol, micro=micro, source_sha256=u.sha(Path(__file__))))
    write('split.json', split)
    resume = SCRATCH / 'resume.pt'
    best = OUT / 'best.pt'
    basis_bytes = len(base) * RANK * 4
    if not best.exists():
        assert shutil.disk_usage(OUT).free > 2 * basis_bytes + 64 * 1024**2, 'Insufficient durable space for atomic best replacement'
    history = []
    step = updates = completed_epoch = 0
    best_ce = math.inf
    best_epoch = 0
    begin = time.time()
    if resume.exists() or best.exists():
        source = resume if resume.exists() else best
        state = torch.load(source, map_location='cpu', weights_only=False)
        previous.check_state(state, identity, len(base), RANK)
        basis, z = state['B'].cuda(), state['z'].cuda()
        step, updates, completed_epoch = state['step'], state['updates'], state['epoch']
        history, best_ce, best_epoch = state['history'], state['best_ce'], state['best_epoch']
        loader.generator.set_state(state['data_rng'])
        del state
        print('RESUME', source, completed_epoch, flush=True)
    else:
        state = torch.load(REFERENCE / 'shared/initial_basis.pt', map_location='cpu', weights_only=False)
        assert state['init_id'] == reference['init_id'] and tuple(state['B'].shape) == (len(base), RANK)
        basis = state['B'].cuda()
        del state
        z = torch.zeros(RANK, device='cuda')
        model.eval()
        u.materialize(selected, base, basis, z)
        baseline = u.evaluate(model, validation)
        history.append(dict(epoch=0, validation=baseline))
        write('history.json', history)

    for epoch in range(completed_epoch + 1, EPOCHS + 1):
        examples.epoch = epoch
        model.train()
        loss_sum = count = 0
        max_displacement = angle_sum = 0.
        backtracked = 0
        for images, targets, _ in loader:
            step += 1
            status('training', epoch=epoch, step=step, total_steps=EPOCHS * len(loader), basis_updates=updates)
            u.materialize(selected, base, basis, z)
            gradient, loss = previous.gradient(model, selected, images, targets, micro)
            warmup = len(loader)
            factor = step / warmup if step <= warmup else .01 + .99 * .5 * (1 + math.cos(math.pi * (step - warmup) / ((EPOCHS - 1) * warmup)))
            audit = step <= 3 or step % len(loader) == 0
            if audit:
                before = basis @ z
            info = step_(basis, z, gradient, .01 * factor, RADIUS, DAMPING)
            updates += int(info['basis_updated'])
            if audit:
                actual = float((basis @ z - before).norm())
                assert actual <= RADIUS + 2e-5 and abs(actual - info['parameter_step_norm']) < 2e-5
                u.materialize(selected, base, basis, z)
                with torch.no_grad():
                    after = torch.nn.functional.cross_entropy(model(images.cuda()), targets.cuda()).item()
                info.update(measured_parameter_step_norm=actual, same_batch_ce_after=after, same_batch_ce_change=after-loss)
            with (OUT / 'steps.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=step, epoch=epoch, ce=loss, **info)) + '\n')
            max_displacement = max(max_displacement, info['parameter_step_norm'])
            angle_sum += info['angle']
            backtracked += int(info['backtracks'] > 0)
            loss_sum += loss * len(targets)
            count += len(targets)
            del gradient
        orthogonality = float((u.gram(basis) - torch.eye(RANK, device='cuda')).abs().max())
        assert orthogonality < 2e-5
        status('validation', epoch=epoch, step=step)
        model.eval()
        u.materialize(selected, base, basis, z)
        metrics = u.evaluate(model, validation)
        improved = metrics['ce'] < best_ce
        if improved:
            best_ce, best_epoch = metrics['ce'], epoch
        row = dict(epoch=epoch, step=step, validation=metrics, train_ce=loss_sum/count,
                   basis_updates=updates, max_parameter_step_norm=max_displacement,
                   mean_angle=angle_sum/len(loader), backtracked_fraction=backtracked/len(loader),
                   orthogonality_error=orthogonality, peak_gpu_bytes=torch.cuda.max_memory_allocated(), seconds=time.time()-begin)
        history.append(row)
        status('checkpoint', epoch=epoch, step=step)
        state = dict(B=basis.cpu(), z=z.cpu(), identity=identity, epoch=epoch, step=step, updates=updates,
                     history=history, best_ce=best_ce, best_epoch=best_epoch, data_rng=loader.generator.get_state())
        pending = SCRATCH / 'pending.pt'
        torch.save(state, pending)
        del state
        if improved:
            assert shutil.disk_usage(OUT).free > pending.stat().st_size + 32 * 1024**2
            durable = OUT / 'best.tmp'
            shutil.copyfile(pending, durable)
            with durable.open('rb') as f:
                os.fsync(f.fileno())
            durable.replace(best)
        pending.replace(resume)
        write('history.json', history)
        write('validation_latest.json', dict(current=metrics, best_epoch=best_epoch, best_ce=best_ce, epoch=epoch))
        print(json.dumps(row), flush=True)

    assert u.frozen_hash(model) == frozen
    state = torch.load(best, map_location='cpu', weights_only=False)
    previous.check_state(state, identity, len(base), RANK)
    assert state['epoch'] == best_epoch
    basis.copy_(state['B'])
    z.copy_(state['z'])
    del state
    u.materialize(selected, base, basis, z)
    model.eval()
    chosen = u.evaluate(model, validation)
    assert abs(chosen['ce'] - best_ce) < 2e-6
    lock = dict(identity=identity, selected_epoch=best_epoch, validation=chosen, checkpoint_sha256=u.sha(best))
    write('selection_lock.json', lock)
    status('locked_test', epoch=best_epoch)
    cached = OUT / 'locked_test.json'
    if cached.exists():
        test = json.loads(cached.read_text())
        assert test['checkpoint_sha256'] == lock['checkpoint_sha256']
    else:
        dataset = u.get_dataset('SUN397', encoder.val_preprocess, str(ROOT/'datasets'), num_workers=0)
        test = dict(checkpoint_sha256=lock['checkpoint_sha256'], metrics=u.evaluate(model, u.loader(dataset.test_dataset)))
        write('locked_test.json', test)
    write('result.json', dict(**lock, test=test['metrics'], formal_steps=step, basis_updates=updates,
                              initial_validation=history[0]['validation'], rank=RANK, parameter_step_radius=RADIUS,
                              damping=DAMPING, trajectory_objective_enabled=False))
    resume.unlink(missing_ok=True)
    status('completed', epoch=best_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--micro', type=int, default=32)
    main(parser.parse_args().micro)
