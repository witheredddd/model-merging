"""DTD training by Fisher–Rao task directions, horizontal lifts and transport."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from collect_dtd_updates import batches, flat, load_experiment, save
from study_dtd_quotient import predictor, response
from train_atlas import embedding
from vo_quotient import VOQuotient


def descent(q, labels):
    """Negative CE gradient on the product of radius-two Fisher spheres."""
    n = len(labels)
    p_half = q.reshape(n, -1) * math.sqrt(n) / 2
    one_hot = F.one_hot(labels, p_half.shape[1]).to(p_half)
    selected = (p_half * one_hot).sum(-1, keepdim=True).clamp_min(1e-12)
    return ((one_hot / selected - p_half) / math.sqrt(n)).flatten()


def transport(source, target, direction, count):
    """Shortest-geodesic Levi-Civita transport on each categorical sphere."""
    a, b, v = (value.reshape(count, -1) for value in (source, target, direction))
    divisor = 4 / count + (a * b).sum(-1, keepdim=True)
    moved = v - ((v * b).sum(-1, keepdim=True) / divisor.clamp_min(1e-15)) * (a + b)
    return moved.flatten()


def distance(a, b, count):
    chord = (a - b).reshape(count, -1).norm(dim=-1) * math.sqrt(count)
    return float((4 * (chord / 4).clamp(max=1).asin()).square().mean().sqrt())


def dual_cg(operator, target, iterations, damping):
    norm = target.norm()
    if not torch.isfinite(norm) or norm <= 1e-14:
        return torch.zeros_like(target), 0., 0.
    # Normalize only the right-hand side; the caller decides the step size.
    rhs = target / norm
    first = operator(rhs)
    ridge = damping * (rhs @ first).clamp_min(1e-12)
    solution = torch.zeros_like(rhs)
    residual, direction = rhs, rhs
    error = residual @ residual
    for i in range(iterations):
        image = (first if i == 0 else operator(direction)) + ridge * direction
        denominator = direction @ image
        if denominator <= 1e-20:
            break
        alpha = error / denominator
        solution = solution + alpha * direction
        residual = residual - alpha * image
        new_error = residual @ residual
        if new_error <= 1e-8:
            break
        direction = residual + new_error / error * direction
        error = new_error
    return solution, float(residual.norm()), float(ridge)


class HorizontalLift:
    def __init__(self, predict, theta, images):
        self.theta = theta
        self.function = lambda weights: embedding(predict(weights, images))
        self.point, self.pullback = torch.func.vjp(self.function, theta)

    def action(self, direction):
        return torch.func.jvp(self.function, (self.theta,), (direction,))[1]

    def solve(self, target, iterations, damping):
        def kernel(x):
            return self.action(self.pullback(x)[0])

        coefficient, residual, ridge = dual_cg(kernel, target, iterations, damping)
        direction = self.pullback(coefficient)[0]
        achieved = self.action(direction)
        alignment = float(F.cosine_similarity(target, achieved, dim=0))
        return direction, dict(cg_residual=residual, damping=ridge, target_cosine=alignment)


def losses(logits, labels, first):
    a = F.cross_entropy(logits[:first], labels[:first])
    b = F.cross_entropy(logits[first:], labels[first:])
    return torch.stack([a, b])


def bounded_step(predict, theta, direction, images, labels, before, first, radius, cap):
    scale = min(radius, cap / max(float(direction.norm()), 1e-20))
    for _ in range(4):
        update = scale * direction
        after = losses(predict(theta + update, images), labels, first)
        if bool((before - after >= 1e-5).all()):
            return update, scale, (before - after).tolist()
        scale *= .5
    return None, 0., None


@torch.no_grad()
def select(predict, theta, candidates, images, labels, first, radius, cap):
    if not candidates:
        return [], [], None, 0., None
    prediction = predict(theta, images)
    before = losses(prediction, labels, first)
    count, classes = prediction.shape
    scales = (math.sqrt(count / first), math.sqrt(count / (count - first)))
    target = (descent(embedding(prediction[:first]), labels[:first]),
              descent(embedding(prediction[first:]), labels[first:]))
    y = response(predict, theta, [d for _, d in candidates], images)
    norm = y.norm(dim=0)
    edge = first * classes
    weights, response_basis, coefficients, selected = [], [], [], []
    used = set()
    strongest = 0.
    while len(used) < len(candidates):
        z = torch.stack(response_basis, dim=1) if response_basis else y[:, :0]
        coordinates = z.T @ y
        outside = y - z @ coordinates
        size = outside.norm(dim=0)
        directions = outside / size.clamp_min(1e-20)
        parts = (directions[:edge] * scales[0], directions[edge:] * scales[1])
        slopes = torch.stack([v @ b for v, b in zip(target, parts)])
        alignment = slopes / torch.stack([v.norm() * b.norm(dim=0) for v, b in zip(target, parts)]).clamp_min(1e-20)
        sign = slopes.sum(0).sign()
        agreement = (alignment * sign).min(0).values
        value = (slopes * sign).min(0).values
        valid = (size > .10 * norm) & (size > 1e-10) & (agreement >= .015) & (value >= .15 * strongest)
        for i in used:
            valid[i] = False
        if not bool(valid.any()):
            break
        index = int(value.masked_fill(~valid, -torch.inf).argmax())
        used.add(index)
        label, original = candidates[index]
        direction = original.clone()
        for coefficient, basis in zip(coordinates[:, index], weights):
            direction.add_(basis, alpha=-float(coefficient))
        direction.mul_(float(sign[index] / size[index]))
        _, _, gain = bounded_step(predict, theta, direction, images, labels, before, first, radius, cap)
        if gain is None:
            continue
        strongest = max(strongest, float(value[index]))
        weights.append(direction)
        response_basis.append(sign[index] * directions[:, index])
        coefficients.append(float(sign[index] * slopes[:, index].mean()))
        selected.append(dict(source=label, independent_ce_gain=gain, minimum_alignment=float(agreement[index])))
    if not weights:
        return weights, selected, None, 0., None
    combined = torch.zeros_like(theta)
    scale = math.sqrt(sum(x * x for x in coefficients))
    for coefficient, direction in zip(coefficients, weights):
        combined.add_(direction, alpha=coefficient / scale)
    step, applied_scale, gain = bounded_step(predict, theta, combined, images, labels, before, first, radius, cap)
    return weights, selected, step, applied_scale, gain


def run(args):
    existing, model, found, base, dataset, validation, splits, ids, base_hash = load_experiment(args.root, args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    predict = predictor(model, found)
    theta = base.clone()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    def record(event, **values):
        row = dict(event=event, **values)
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(row, flush=True)

    def get(selected):
        samples = [splits.train_dataset[i] for i in selected]
        return torch.stack([x for x, _ in samples]).cuda(), torch.tensor([y for _, y in samples], device='cuda')

    @torch.no_grad()
    def assess():
        if not args.adamw:
            existing.assign_flat(found, theta)
        return existing.evaluate(model, validation)

    anchor, _ = get(ids[:args.anchors])
    audit, audit_labels = get(ids[48:64])
    used = ids[96:]
    record('config', seed=args.seed, epochs=args.epochs, method='adamw' if args.adamw else 'bundle',
           train=len(used), base_sha256=base_hash, radius=args.radius, cap=args.cap, anchors=args.anchors,
           iterations=args.iterations, damping=args.damping, transport=not args.no_transport and not args.adamw,
           source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    best = assess()
    record('validation', epoch=0, step=0, **best)
    best_epoch = 0
    best_ce = best['ce']
    save(output / 'best.pt', dict(weights=base.cpu(), epoch=0, score=best,
                                  names=[n for n, _ in found], base_sha256=base_hash))
    if args.adamw:
        for _, parameter in found:
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW([p for _, p in found], lr=1e-5, weight_decay=.1, fused=True)

    step = 0
    history = []
    previous_point = None
    for epoch in range(1, args.epochs + 1):
        gauge_checked = False
        for x, y, _ in batches(dataset, used, args.seed, epoch):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            step += 1
            started = time.monotonic()
            if args.adamw:
                optimizer.zero_grad(set_to_none=True)
                for k in range(0, len(y), 16):
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = F.cross_entropy(model(x[k:k + 16]).float(), y[k:k + 16], reduction='sum') / len(y)
                    loss.backward()
                optimizer.step()
                continue
            with torch.no_grad():
                point = embedding(predict(theta, anchor))
                before_audit = float(F.cross_entropy(predict(theta, audit), audit_labels))
                candidates, diagnostics = [], []
                if history and not args.no_transport:
                    lift = HorizontalLift(predict, theta, anchor)
                    for old in history:
                        transported = transport(previous_point, point, old, len(anchor))
                        direction, info = lift.solve(transported, args.iterations, args.damping)
                        info['transport_norm_error'] = abs(float(transported.norm() / old.norm().clamp_min(1e-20)) - 1)
                        diagnostics.append(dict(source='transport', **info))
                        if info['target_cosine'] > .5:
                            candidates.append(('transport', direction))
                    del lift
                induction = min(8, len(y) // 5)
                for start in range(0, 3 * induction, induction):
                    sl = slice(start, start + induction)
                    lift = HorizontalLift(predict, theta, x[sl])
                    direction, info = lift.solve(descent(lift.point, y[sl]), args.iterations, args.damping)
                    diagnostics.append(dict(source='task', **info))
                    if info['target_cosine'] > .5:
                        candidates.append(('task', direction))
                    del lift
                holdout_x, holdout_y = x[3 * induction:], y[3 * induction:]
                weights, selected, update, scale, gain = select(predict, theta, candidates, holdout_x, holdout_y,
                                                               len(holdout_y) // 2, args.radius, args.cap)
                previous_point = point
                history = ([column for column in response(predict, theta, weights, anchor).T if column.norm() > 1e-10]
                           if weights and not args.no_transport else [])
                gauge = None
                if update is not None:
                    if not gauge_checked:
                        gauge = VOQuotient(model, found, theta).check_projection(update)
                        gauge_checked = True
                    theta.add_(update)
                post_point = embedding(predict(theta, anchor))
                after_audit = float(F.cross_entropy(predict(theta, audit), audit_labels))
                record('step', step=step, epoch=epoch, rank=len(weights), applied=update is not None,
                       selected=selected, lift=diagnostics, ce_gain_independent=gain, audit_ce_gain=before_audit - after_audit,
                       fr_step=distance(point, post_point, len(anchor)), scale=scale, gauge=gauge,
                       seconds=time.monotonic() - started)
        score = assess()
        record('validation', epoch=epoch, step=step, **score)
        weights = flat(found) if args.adamw else theta
        state = dict(weights=weights.cpu(), epoch=epoch, score=score, names=[n for n, _ in found], base_sha256=base_hash)
        save(output / 'latest.pt', state)
        if score['ce'] < best_ce:
            best, best_ce, best_epoch = score, score['ce'], epoch
            save(output / 'best.pt', state)
    record('complete', steps=step, best_epoch=best_epoch, **best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--anchors', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--damping', type=float, default=.04)
    parser.add_argument('--radius', type=float, default=.14)
    parser.add_argument('--cap', type=float, default=.18)
    parser.add_argument('--adamw', action='store_true')
    parser.add_argument('--no-transport', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.anchors <= 48 and args.epochs >= 1 and args.iterations >= 1
            and min(args.radius, args.cap, args.damping) > 0):
        parser.error('Invalid training settings')
    run(args)
