"""Train the DTD ViT-B/16 expert with online, candidate-local Fisher geometry."""

import argparse
import fcntl
import json
import math
import sys
import time
from pathlib import Path

import torch


def propose(g, weights, basis, first, second, step, decay):
    """Lift the old moments, then incorporate the full current gradient."""
    direction = torch.empty_like(g)
    for offset in range(0, g.numel(), 1048576):
        section = slice(offset, offset + 1048576)
        grad = g[section]
        m = .1 * grad
        v = .001 * grad.square()
        if len(basis):
            frame = basis[:, section]
            m.add_(first @ frame, alpha=.9)
            v.add_(((second @ frame) * frame).sum(0).clamp_min_(0), alpha=.999)
        m.div_(1 - .9 ** step)
        v.div_(1 - .999 ** step).sqrt_().add_(1e-8)
        direction[section] = -m / v - decay * weights[section]
    if torch.dot(g, direction) >= 0:
        return -g * (direction.norm() / g.norm().clamp_min(1e-30)), True
    return direction, False


def candidate_frame(direction, basis, pending, history, budget):
    """Incremental two-pass modified Gram-Schmidt on whole-model directions."""
    frame = torch.empty((budget, direction.numel()), device=direction.device)
    rows = [direction]
    if pending is not None:
        rows.append(pending)
    rows.extend(basis.unbind())
    rows.extend(vector for _, _, vector in reversed(history))
    count = 0
    for vector in rows:
        previous = frame[:count]
        norm = vector.norm()
        residual = frame[count]
        residual.copy_(vector)
        if count:
            for _ in range(2):
                residual.sub_((previous @ residual) @ previous)
        length = residual.norm()
        if length > 1e-5 * norm:
            residual.div_(length)
            count += 1
            if count == budget:
                break
    return frame[:count]


def fisher(derivatives, probabilities):
    centered = derivatives - (derivatives * probabilities).sum(-1, keepdim=True)
    whitened = (centered * probabilities.sqrt()).flatten(1)
    return whitened @ whitened.T / probabilities.shape[0]


def fr_distance(old, new):
    # Chord length on the square-root probability sphere avoids acos at 1.
    chord = (old - new).norm(dim=-1)
    return float((4 * (chord / 2).clamp(max=1).asin()).square().mean().sqrt())


def select(matrix, proposals, grad, epsilon, rho, null_relative):
    eigenvalues, modes = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(0)
    horizontal = eigenvalues > max(1e-12, float(eigenvalues[-1]) * null_relative)
    activity = eigenvalues * (proposals @ modes).square().sum(0)
    available = activity.masked_fill(~horizontal, -1).argsort(descending=True)
    available = available[:int(horizontal.sum())]
    total = activity.sum()
    tail = (1 - activity[available].cumsum(0) / total.clamp_min(1e-30)).clamp_min(0)
    full_descent = -grad @ proposals[0]
    mode_descent = -(grad @ modes) * (proposals[0] @ modes)
    descent = mode_descent[available].cumsum(0)
    accepted = ((tail <= epsilon) & (descent >= rho * full_descent) & (descent > 0)).nonzero()
    indices = available[:int(accepted[0, 0]) + 1] if len(accepted) else available[:0]
    return modes[:, indices], {
        'eigenvalues': eigenvalues.tolist(),
        'horizontal_rank': len(available),
        'vertical_rank': len(eigenvalues) - len(available),
        'activity': activity.tolist(),
        'selected_mode_indices': indices.tolist(),
        'discarded_activity_fraction': float(tail[len(indices) - 1]) if len(indices) else 1.0,
        'full_descent': float(full_descent),
        'max_horizontal_prefix_descent': float(descent.max()) if len(descent) else 0.0,
    }


def align_and_measure(old, new):
    overlap = (new @ old.T).double()
    left, cosine, right = torch.linalg.svd(overlap, full_matrices=True)
    shared = len(cosine)
    # Match the shared coordinates; preserve a complete orthogonal rotation on growth.
    match = torch.eye(len(new), dtype=torch.float64, device=new.device)
    if len(new) >= len(old):
        match[:shared, :shared] = right
    # On shrinkage the old principal coordinates define the surviving axes.
    rotation = match.T @ left.T
    return rotation.float() @ new, rotation @ overlap, cosine.clamp(0, 1).acos().rad2deg()


def move_moments(overlap, first, second, grad):
    # Store the full small raw second moment, so transport is exact in coordinates.
    first = .9 * (overlap @ first) + .1 * grad
    second = .999 * (overlap @ second @ overlap.T) + .001 * torch.outer(grad, grad)
    return first, second


def geometry(model, found, weights, frame, images, check):
    names = ['image_encoder.' + name for name, _ in found]
    sizes = [p.numel() for _, p in found]
    shapes = [p.shape for _, p in found]

    def logits(flat):
        params = {name: part.view(shape)
                  for name, part, shape in zip(names, flat.split(sizes), shapes)}
        return torch.func.functional_call(model, params, (images,)).float()

    with torch.no_grad():
        current = logits(weights).double()
        responses = torch.stack([torch.func.jvp(logits, (weights,), (row,))[1]
                                 for row in frame]).double()
        audit = None
        if check:
            # Check the JVP/VJP adjoint identity using multiple fixed cotangents.
            _, pullback = torch.func.vjp(logits, weights)
            index = torch.arange(current.numel(), device=current.device).view_as(current)
            cotangents = [torch.sin(index * frequency).float() for frequency in (.7, 1.3, 1.9)]
            forward, reverse = [], []
            for cotangent in cotangents:
                forward.append((cotangent * responses[0]).sum())
                reverse.append(torch.dot(pullback(cotangent)[0], frame[0]).double())
            forward, reverse = torch.stack(forward), torch.stack(reverse)
            adjoint_error = float((forward - reverse).norm() / forward.norm().clamp_min(1e-12))
            if not adjoint_error < .002:
                raise RuntimeError(f'CLIP JVP/VJP adjoint mismatch: {adjoint_error:.4g}')
            errors = []
            for h in (.002, .005, .01, .02, .05):
                difference = (logits(weights + h * frame[0]).double()
                              - logits(weights - h * frame[0]).double()) / (2 * h)
                relative = float((difference - responses[0]).norm()
                                 / responses[0].norm().clamp_min(1e-12))
                errors.append(dict(fd_step=h, relative_jvp_error=relative))
            audit = {'jvp_adjoint_error': adjoint_error, 'jvp_finite_difference': errors}
    return fisher(responses, current.softmax(-1)), current, audit


def save_atomic(path, value):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def flat_weights(found):
    return torch.cat([p.detach().flatten() for _, p in found])


def run(args):
    sys.path[:0] = [str(args.root), str(args.root / 'scripts'), str(args.root / 'scripts/tfpml')]
    import dtd_grassmann4 as existing
    import runtime

    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    # Forward AD requires the math SDPA kernel on this CLIP runtime.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    existing.seed_all(args.seed)
    head_path = args.root / 'models/checkpoints/ViT-B-16/head_DTDVal.rebuilt.pt'
    head_info = json.loads(head_path.with_suffix('.json').read_text())
    head_hash = existing.sha(head_path)
    if head_info['sha256'] != head_hash:
        raise RuntimeError('DTD head checksum mismatch')
    existing.ROOT = args.root
    head = existing.ClassificationHead.load(str(head_path))
    encoder, model, found, weights = runtime.build_model(existing, head)
    assert weights.numel() == 84934656 and len(found) == 48
    model.requires_grad_(False).eval()
    for _, parameter in found:
        parameter.requires_grad_(True)
    fixed_hash = existing.frozen_hash(model)
    data, _, validation, splits = runtime.training_data(existing, encoder, args.seed)
    classes = [name.replace('_', ' ') for name in splits.train_dataset.dataset.classes]
    if classes != head_info['classnames']:
        raise RuntimeError('DTD head labels differ from ImageFolder classes')
    assert len(data) == 3384 and len(splits.test_dataset) == 376
    probe_rng = torch.Generator().manual_seed(args.seed + 900)
    probe_order = torch.randperm(len(data), generator=probe_rng).tolist()

    def images_at(ids):
        return torch.stack([splits.train_dataset[i][0] for i in ids]).cuda()

    anchor_ids = probe_order[:args.probe]
    anchor = images_at(anchor_ids)
    pool = probe_order[args.probe:]
    basis = weights.new_empty((0, weights.numel()))
    first, second = basis.new_empty(0).double(), basis.new_empty((0, 0)).double()
    history, pending = [], None
    budget, step, applied, first_epoch, first_batch = args.budget, 0, 0, 0, 0
    output = args.output
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'}
    checkpoint = output / 'latest.pt'
    if args.resume:
        if json.loads((output / 'config.json').read_text())['args'] != cfg:
            raise ValueError('Resume requires the original configuration')
        state = torch.load(checkpoint, weights_only=True, map_location='cuda')
        weights, basis = state['weights'], state['basis']
        first, second, history, pending = (state[k] for k in ('first', 'second', 'history', 'pending'))
        budget, step, applied = state['budget'], state['step'], state['applied']
        first_epoch, first_batch = state['next_epoch'], state['next_batch']
        validation_rows = state['validation']
        existing.assign_flat(found, weights)
        del state
    else:
        validation_rows = [dict(epoch=0, **existing.evaluate(model, validation))]
        existing.write(output / 'config.json', dict(args=cfg, head_sha256=head_hash,
                       head_kind=head_info['kind'], parameter_names=[n for n, _ in found],
                       source_sha256=existing.sha(Path(__file__)),
                       base_sha256=existing.sha(head_path.parent / 'pretrained.pt'),
                       anchor_train_indices=[splits.train_dataset.indices[i] for i in anchor_ids]))
        existing.write(output / 'validation.json', validation_rows)
        print('initial validation', validation_rows[0], flush=True)

    def snapshot(next_epoch, next_batch):
        save_atomic(checkpoint, dict(weights=weights.cpu(), basis=basis.cpu(),
                    first=first.cpu(), second=second.cpu(), step=step, budget=budget, applied=applied,
                    history=[(t, p.cpu(), d.cpu()) for t, p, d in history],
                    pending=pending.cpu() if pending is not None else None,
                    next_epoch=next_epoch, next_batch=next_batch, validation=validation_rows))

    steps_dir = output / 'steps'
    steps_dir.mkdir(exist_ok=True)
    for epoch in range(first_epoch, args.epochs):
        data.epoch = epoch
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        batches = [order[i:i + args.batch] for i in range(0, len(order), args.batch)]
        if args.steps_per_epoch:
            batches = batches[:args.steps_per_epoch]
        start = first_batch if epoch == first_epoch else 0
        loader = torch.utils.data.DataLoader(data, batch_sampler=batches[start:], num_workers=4,
                                            pin_memory=True)
        for batch, (x, y, _) in enumerate(loader, start):
            started = time.monotonic()
            step += 1
            grad, loss = existing.gradient(model, found, x, y)
            direction, reset_direction = propose(grad, weights, basis, first.float(), second.float(),
                                                  step, args.decay)
            with torch.no_grad():
                anchor_probability = model(anchor).double().softmax(-1).sqrt()
                distances = [fr_distance(old, anchor_probability) for _, old, _ in history]
                history = [entry for entry, distance in zip(history, distances) if distance <= args.fr_radius]
                if len(basis) >= budget and budget < args.max_budget:
                    budget = min(args.max_budget, budget * 2)
                frame = candidate_frame(direction, basis, pending, history, budget)
                pending = None
                offset = (step - 1) * args.probe
                ids = [pool[(offset + i) % len(pool)] for i in range(args.probe)]
                probe = images_at(ids)
                matrix, logits_before, audit = geometry(model, found, weights, frame, probe, step == 1)
                recent = [(step, direction), *((t, d) for t, _, d in reversed(history))]
                coords = torch.stack([(frame @ d).double() * args.beta ** ((step - t) / 2)
                                      for t, d in recent])
                grad_coords = (frame @ grad).double()
                selected, row = select(matrix, coords, grad_coords, args.epsilon, args.rho, args.null_relative)
                rank = selected.shape[1]
                if rank:
                    new_basis = selected.T.float() @ frame
                    new_basis, overlap, angles = align_and_measure(basis, new_basis)
                    grad_in_basis = (new_basis @ grad).double()
                    first, second = move_moments(overlap, first, second, grad_in_basis)
                    update_coords = new_basis @ direction
                    update = update_coords @ new_basis
                    basis = new_basis
                    actual_descent = float(-grad_in_basis @ update_coords.double())
                    assert actual_descent >= (args.rho - 1e-4) * row['full_descent'] > 0
                    weights.add_(update, alpha=args.lr)
                    existing.assign_flat(found, weights)
                    reference = logits_before.log_softmax(-1)
                    updated = model(probe).double().log_softmax(-1)
                    two_kl = max(0., float(2 * (reference.exp() * (reference - updated)).sum(-1).mean()))
                    residual = direction - update
                    residual_coords = coords[0] - selected @ (selected.T @ coords[0])
                    applied += 1
                else:
                    # Reject the step if no horizontal prefix satisfies both guards.
                    update_coords = weights.new_empty(0)
                    angles = weights.new_empty(0)
                    first, second = move_moments(torch.eye(len(basis), device=weights.device,
                                                   dtype=torch.float64), first, second, (basis @ grad).double())
                    actual_descent = two_kl = 0.0
                    residual, residual_coords = direction, coords[0]
                total = float(coords[0] @ matrix @ coords[0])
                removed = max(0., float(residual_coords @ matrix @ residual_coords))
                residual_descent = float(-grad_coords @ residual_coords)
                if total > 0 and removed / total > args.epsilon and residual_descent > 0:
                    pending = residual
                history = (history + [(step, anchor_probability, direction)])[-budget:]
                kept_coords = coords[0] - residual_coords
                row.update(step=step, epoch=epoch + 1, batch=batch + 1, loss=loss, applied=bool(rank),
                           rank=rank, carried_rank=len(basis), candidate_rank=len(frame), candidate_budget=budget,
                           history_count=len(history), history_distances=distances, descent=actual_descent,
                           descent_ratio=actual_descent / max(row['full_descent'], 1e-30),
                           proposal_gradient_fallback=reset_direction, residual_descent=residual_descent,
                           fisher_energy=total, residual_fisher_energy=removed, innovation=pending is not None,
                           measured_2kl=two_kl,
                           quadratic_2kl=float(kept_coords @ matrix @ kept_coords) * args.lr ** 2,
                           angles_deg=angles.tolist(), coordinates=update_coords.tolist(),
                           probe_train_indices=[splits.train_dataset.indices[i] for i in ids],
                           seconds=time.monotonic() - started)
                if audit is not None:
                    row.update(audit)
                existing.write(steps_dir / f'{step:06d}.json', row)
                print(f"step {step} loss={loss:.4f} rank={rank}/{len(frame)} "
                      f"descent={row['descent_ratio']:.3f} dt={row['seconds']:.2f}s", flush=True)
                if (batch + 1) % args.checkpoint_every == 0 and batch + 1 < len(batches):
                    snapshot(epoch, batch + 1)
        validation_rows.append(dict(epoch=epoch + 1, **existing.evaluate(model, validation)))
        existing.write(output / 'validation.json', validation_rows)
        save_atomic(output / f'basis_epoch_{epoch + 1:03d}.pt',
                    dict(basis=basis.cpu(), step=step, names=[n for n, _ in found]))
        snapshot(epoch + 1, 0)
        print('validation', validation_rows[-1], flush=True)
    if existing.frozen_hash(model) != fixed_hash:
        raise RuntimeError('Frozen model state changed during DTD training')
    if not applied:
        raise RuntimeError('No proposed update satisfied the Fisher and descent guards')
    save_atomic(output / 'encoder.pt', {k: v.cpu() for k, v in encoder.state_dict().items()})
    existing.write(output / 'complete.json', dict(steps=step, applied=applied,
                   validation=validation_rows, frozen_hash=fixed_hash))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--steps-per-epoch', type=int, default=0)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--probe', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--max-budget', type=int, default=16)
    parser.add_argument('--checkpoint-every', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--decay', type=float, default=.1)
    parser.add_argument('--epsilon', type=float, default=.05)
    parser.add_argument('--rho', type=float, default=.9)
    parser.add_argument('--beta', type=float, default=.9)
    parser.add_argument('--null-relative', type=float, default=1e-4)
    parser.add_argument('--fr-radius', type=float, default=.15)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.budget <= args.max_budget and 1 <= args.probe <= 64 and args.epochs > 0
            and args.batch > 0 and args.steps_per_epoch >= 0 and args.checkpoint_every > 0
            and args.lr > 0 and args.decay >= 0 and args.fr_radius > 0 and 0 < args.beta <= 1
            and 0 < args.rho <= 1 and 0 < args.epsilon < 1 and 0 < args.null_relative < 1):
        parser.error('Invalid run parameters')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.resume and (args.output / 'config.json').exists():
            parser.error('Run exists; use --resume or a new output directory')
        run(args)


if __name__ == '__main__':
    main()
