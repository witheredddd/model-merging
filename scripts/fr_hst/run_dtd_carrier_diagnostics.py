"""Full-visual GGN probes and causal residual replay on existing DTD states."""

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from analyze_dtd_expert_geometry import load_expert
from measure_dtd_dynamic_mechanisms import load_trace
from train_eight_expert_queue import BASE, HEADS, ROOT, build_model, existing, read, splits


RANK = 3
ANCHORS = (0, 54, 108, 216)


def orthogonal(v, basis):
    for _ in range(2):
        for u in basis:
            v = v - (u @ v) * u
    return v / v.norm()


def row_space(rows, rank, normalize=True):
    stacked = torch.stack([r / r.norm() for r in rows] if normalize else rows)
    gram = stacked @ stacked.T
    values, vectors = torch.linalg.eigh(gram.double())
    if values[-rank] < 1e-7:
        raise ValueError('Insufficient independent gradient or update samples')
    weights = (vectors[:, -rank:] / values[-rank:].sqrt()).T.flip(0)
    return list(weights.float() @ stacked)


def angles(a, b):
    overlap = torch.stack(a) @ torch.stack(b).T
    return torch.linalg.svdvals(overlap).clamp(0, 1).acos().rad2deg().cpu().tolist()


class Operator:
    def __init__(self, model, visual):
        self.model, self.visual = model, visual
        keys, self.params = zip(*visual.named_parameters())
        self.keys = keys
        self.names = tuple('image_encoder.model.visual.' + key for key in keys)
        self.sizes = [p.numel() for p in self.params]
        self.shapes = [p.shape for p in self.params]
        self.generator = torch.Generator(device='cuda').manual_seed(20261109)

    def parts(self, vector):
        return tuple(x.view(shape) for x, shape in zip(vector.split(self.sizes), self.shapes))

    def delta(self, a, b):
        return torch.cat([(b['visual'][k] - a['visual'][k]).flatten().cuda() for k in self.keys])

    def random(self):
        return torch.randn(sum(self.sizes), device='cuda', generator=self.generator)

    def load(self, state):
        self.visual.load_state_dict(state['visual'])

    @torch.no_grad()
    def evaluate(self, x, delta=None):
        if delta is None:
            return self.model(x).float()
        overrides = {n: p + d for n, p, d in zip(self.names, self.params, self.parts(delta))}
        return torch.func.functional_call(self.model, overrides, (x,)).float()

    def gradient(self, x, y):
        loss = F.cross_entropy(self.model(x).float(), y)
        return torch.cat([p.detach().flatten() for p in torch.autograd.grad(loss, self.params)])

    @torch.no_grad()
    def logit_jvp(self, x, vector):
        def predict(*weights):
            return torch.func.functional_call(self.model, dict(zip(self.names, weights)), (x,)).float()

        logits, response = torch.func.jvp(predict, self.params, self.parts(vector))
        return logits, response

    def ggn(self, x, vector):
        z, jv = self.logit_jvp(x, vector)
        p = z.softmax(-1)
        response = p * (jv - (p * jv).sum(-1, keepdim=True)) / len(x)
        z = self.model(x).float()
        return torch.cat([g.detach().flatten() for g in torch.autograd.grad(z, self.params, response)])

    def quadratic(self, x, y, vector):
        z, jv = self.logit_jvp(x, vector)
        p = z.double().softmax(-1)
        linear = ((p - F.one_hot(y, p.shape[1])) * jv).sum(-1).mean()
        center = jv - (p * jv).sum(-1, keepdim=True)
        curvature = (p * center.square()).sum(-1).mean()
        return float(-linear - .5 * curvature), float(curvature)


def metrics(before, after, labels):
    p, q = (z.double().log_softmax(-1) for z in (before, after))
    return dict(ce_gain=float(F.nll_loss(p, labels) - F.nll_loss(q, labels)),
                kl_from_start=float((p.exp() * (p - q)).sum(-1).mean()),
                accuracy_gain=float((after.argmax(-1) == labels).double().mean()
                                    - (before.argmax(-1) == labels).double().mean()))


def spectrum(op, images, length=12, trace_samples=6):
    # Fully reorthogonalized Lanczos. A fresh product validates each retained Ritz pair.
    basis, diagonal, off_diagonal = [], [], []
    direction = orthogonal(op.random(), [])
    for i in range(length):
        basis.append(direction)
        image = op.ggn(images, direction)
        diagonal.append(float(direction @ image))
        if i + 1 < length:
            direction = orthogonal(image, basis)
            off_diagonal.append(float(direction @ image))
    tridiagonal = torch.diag(torch.tensor(diagonal, dtype=torch.double))
    off = torch.tensor(off_diagonal, dtype=torch.double)
    tridiagonal += torch.diag(off, 1) + torch.diag(off, -1)
    eig, vec = torch.linalg.eigh(tridiagonal)
    top = list(vec[:, -RANK:].T.flip(0).float().cuda() @ torch.stack(basis))
    ritz = eig[-RANK:].flip(0).tolist()
    error = [float((op.ggn(images, v) - e * v).norm() / abs(e)) for v, e in zip(top, ritz)]
    traces, squares = [], []
    for _ in range(trace_samples):
        noise = op.random().sign()
        image = op.ggn(images, noise)
        traces.append(float(noise @ image))
        squares.append(float(image @ image))
    rank = (sum(traces) / trace_samples) ** 2 / (sum(squares) / trace_samples)
    return top, dict(ritz_values=eig.flip(0).tolist(), top_relative_residual=error,
                     trace_samples=traces, squared_trace_samples=squares, participation_ratio=rank,
                     ggn_products=length + RANK + trace_samples)


def fit_bases(op, history, x, y):
    start = time.monotonic()
    op.load(history[-1])
    current = op.gradient(x, y)
    seconds = {'common_gradient': time.monotonic() - start}
    start = time.monotonic()
    gradients = [current]
    for previous in history[:-1]:
        op.load(previous)
        gradients.append(op.gradient(x, y))
    op.load(history[-1])
    supplementary = 7 - len(gradients)
    for _ in range(supplementary):
        index = torch.randint(len(x), (len(x),), device='cuda', generator=op.generator)
        gradients.append(op.gradient(x[index], y[index]))
    pca = row_space(gradients, RANK)
    del gradients
    torch.cuda.synchronize()
    seconds['gradient_pca'] = time.monotonic() - start
    start = time.monotonic()
    # Two products form K_3. The third evaluates the final basis direction's fit curvature.
    krylov = [orthogonal(current, [])]
    for _ in range(RANK - 1):
        krylov.append(orthogonal(op.ggn(x, krylov[-1]), krylov))
    last_ggn = op.ggn(x, krylov[-1])
    last_curvature = float(krylov[-1] @ last_ggn)
    del last_ggn
    seconds['krylov'] = time.monotonic() - start
    start = time.monotonic()
    sketch = []
    for _ in range(RANK):
        sketch.append(orthogonal(op.ggn(x, op.random()), sketch))
    torch.cuda.synchronize()
    seconds['ggn_sketch'] = time.monotonic() - start
    top, stats = spectrum(op, x)
    bases = {'gradient_pca': pca, 'krylov': krylov, 'ggn_sketch': sketch, 'ggn_ritz': top}
    directions = dict(pca_to_ggn_angles_deg=angles(pca, top), krylov_to_ggn_angles_deg=angles(krylov, top),
                      gradient_energy_in_ggn_top3=float((torch.stack(top) @ current).square().sum() / current.square().sum()))
    budget = dict(gradient_pca={'vjp': 7, 'jvp': 0}, krylov={'vjp': 4, 'jvp': 3}, ggn_sketch={'vjp': 4, 'jvp': 3},
                  past_checkpoints=len(history) - 1, bootstrap_gradients=supplementary,
                  basis_generation_seconds=seconds, last_krylov_fit_curvature=last_curvature)
    return bases, directions, budget, stats


def analyze_anchor(op, states, step, groups):
    current = states[step]
    history = [state for k, state in states.items() if k <= step][-7:]
    # Use a same-width future interval where one exists. The endpoint supplies only local diagnostics.
    future = states.get(step + 27) if step < 216 else None
    delta = op.delta(current, future) if future else None
    radius = float(delta.norm()) if future else float(op.delta(states[189], current).norm())
    results = []
    for group in (0, 1):
        x, y = groups[group]
        audit_x, audit_y = groups[1 - group]
        bases, directions, budget, stats = fit_bases(op, history, x, y)
        old = op.evaluate(audit_x)
        reference = metrics(old, op.evaluate(audit_x, delta), audit_y) if future else None
        audit = {}
        g = op.gradient(x, y)
        for name, rows in bases.items():
            basis = torch.stack(rows)
            fit_descent = -(basis @ g) @ basis
            trial = .25 * radius * fit_descent / fit_descent.norm()
            record = {}
            tests = [('equal_norm_fit_descent', trial)]
            if future:
                projection = (basis @ delta) @ basis
                record['future_parameter_energy'] = float(projection.square().sum() / delta.square().sum())
                tests.append(('future_projection', projection))
            for label, update in tests:
                actual = metrics(old, op.evaluate(audit_x, update), audit_y)
                predicted, curvature = op.quadratic(audit_x, audit_y, update)
                record[label] = dict(**actual, quadratic_ce_gain=predicted, quadratic_curvature=curvature,
                                     quadratic_error=predicted - actual['ce_gain'])
            if reference and reference['ce_gain'] > 0:
                record['future_ce_retention'] = record['future_projection']['ce_gain'] / reference['ce_gain']
            audit[name] = record
        results.append(dict(fit_group=group, audit_group=1 - group, future_step=future['step'] if future else None,
                            equal_norm_radius=.25 * radius, full_future=reference, geometry=directions,
                            spectrum=stats, budget=budget, methods=audit))
        print(f'GGN anchor {step}, fit group {group} complete', flush=True)
    return dict(step=step, groups=results)


def audit_five_sweeps(op, states, groups):
    """Match the two GGN products that actually generate K_3, and check the operator."""
    anchors = []
    for step in ANCHORS:
        history = [state for k, state in states.items() if k <= step][-5:]
        current = states[step]
        future = states.get(step + 27) if step < 216 else None
        delta = op.delta(current, future) if future else None
        radius = float(delta.norm()) if future else float(op.delta(states[189], current).norm())
        entries = []
        for group, (x, y) in enumerate(groups):
            op.generator.manual_seed(20261109 + step + group * 1000)
            audit_x, audit_y = groups[1 - group]
            gradients = []
            for state in history:
                op.load(state)
                gradients.append(op.gradient(x, y))
            for _ in range(5 - len(gradients)):
                index = torch.randint(len(x), (len(x),), device='cuda', generator=op.generator)
                gradients.append(op.gradient(x[index], y[index]))
            old = op.evaluate(audit_x)
            full = metrics(old, op.evaluate(audit_x, delta), audit_y) if future else None
            gradient = gradients[len(history) - 1]
            krylov = [orthogonal(gradient, [])]
            for _ in range(RANK - 1):
                krylov.append(orthogonal(op.ggn(x, krylov[-1]), krylov))
            frames = {'normalized': row_space(gradients, RANK),
                      'raw': row_space(gradients, RANK, normalize=False), 'krylov': krylov}
            results = {}
            for name, vectors in frames.items():
                basis = torch.stack(vectors)
                direction = -(basis @ gradient) @ basis
                direction *= radius / direction.norm()
                sweep = []
                for scale in (.01, .03, .1, .25):
                    trial = scale * direction
                    actual = metrics(old, op.evaluate(audit_x, trial), audit_y)
                    predicted, curvature = op.quadratic(audit_x, audit_y, trial)
                    sweep.append(dict(scale=scale, **actual, quadratic_ce_gain=predicted, quadratic_curvature=curvature))
                result = dict(equal_norm_fit_descent=sweep[-1], scale_sweep=sweep)
                if future:
                    update = (basis @ delta) @ basis
                    actual = metrics(old, op.evaluate(audit_x, update), audit_y)
                    result.update(future_projection=actual, future_parameter_energy=float(update.square().sum() / delta.square().sum()))
                    if full['ce_gain'] > 0:
                        result['future_ce_retention'] = actual['ce_gain'] / full['ce_gain']
                results[name] = result
            del gradients
            v, w = [orthogonal(op.random(), []) for _ in range(2)]
            gv, gw = op.ggn(x, v), op.ggn(x, w)
            curvature = float(v @ gv)
            _, output_curvature = op.quadratic(x, y, v)
            variance_error = abs(curvature - output_curvature) / output_curvature
            symmetry_error = abs(float(v @ gw - w @ gv)) / (curvature * float(w @ gw)) ** .5
            if curvature <= 0 or max(variance_error, symmetry_error) > 1e-4:
                raise ValueError(f'GGN variance or symmetry check failed at {step}/{group}')
            entries.append(dict(fit_group=group, audit_group=1 - group, past_checkpoints=len(history) - 1,
                                bootstrap_gradients=5 - len(history), pca_fit_vjp=5,
                                comparable_krylov_fit_sweeps={'vjp': 3, 'jvp': 2},
                                ggn_variance_relative_error=variance_error, ggn_bilinear_symmetry_error=symmetry_error,
                                methods=results))
        anchors.append(dict(step=step, groups=entries))
        print(f'Five-sweep PCA and GGN identity checks at {step} complete', flush=True)
    return anchors


def replay(op, states, groups, output):
    sequence = list(states.values())
    deltas = [op.delta(a, b) for a, b in zip(sequence[:4], sequence[1:5])]
    basis = torch.stack(row_space(deltas, RANK))
    del deltas
    residuals, records = [], []
    for start, end in zip(sequence[4:], sequence[5:]):
        op.load(start)
        delta = op.delta(start, end)
        inside = (basis @ delta) @ basis
        outside = delta - inside
        previous_cosines = [float((r @ outside) / outside.norm()) for r in residuals]
        residuals.append(outside / outside.norm())
        groups_out, full_outputs = [], []
        for g, (x, y) in enumerate(groups):
            before = op.evaluate(x)
            conditions = {'carrier': metrics(before, op.evaluate(x, inside), y),
                          'residual': metrics(before, op.evaluate(x, outside), y)}
            for amount in (.25, .5, .75, 1.):
                logits = op.evaluate(x, inside + amount * outside)
                conditions[f'carrier_plus_{amount:g}'] = metrics(before, logits, y)
            full_outputs.append(logits.double().softmax(-1).cpu())
            groups_out.append(dict(probe_group=g, conditions=conditions,
                                   ce_interaction=conditions['carrier_plus_1']['ce_gain'] - conditions['carrier']['ce_gain']
                                   - conditions['residual']['ce_gain']))
        op.load(end)
        for group, (x, _) in enumerate(groups):
            p = op.evaluate(x).double().softmax(-1).cpu()
            if (not torch.allclose(p, end['probability'][group], atol=2e-5, rtol=0)
                    or not torch.allclose(full_outputs[group], p, atol=2e-5, rtol=0)):
                raise ValueError(f'Replay endpoint mismatch at step {end["step"]}')
        records.append(dict(start=start['step'], end=end['step'], carrier_energy_fraction=float(inside.square().sum() / delta.square().sum()),
                            cosine_to_prior_residuals=previous_cosines, groups=groups_out))
        print(f'residual replay {start["step"]}–{end["step"]} complete', flush=True)
    frame = torch.stack(row_space(residuals[:6], RANK))
    recurrence = [float((frame @ r).square().sum()) for r in residuals[6:]]
    existing.write(output, dict(fit_steps=[state['step'] for state in sequence[:5]], rank=RANK, windows=records,
                                residual_recurrence_rank3_later_six=recurrence))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-five-sweeps', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, states = load_trace(args.trace)
    states = {s['step']: s for s in states}
    head = existing.ClassificationHead.load(str(HEADS / 'DTD.pt')).cuda().eval()
    encoder, model, found, base_flat = build_model(existing, head)
    del base_flat
    visual = encoder.model.visual
    model.requires_grad_(False).eval()
    visual.requires_grad_(True)
    base_hash = existing.sha(BASE)
    expert, expert_manifest = load_expert(args.queue, 'DTD', [name for name, _ in found], base_hash)
    for key, weight in visual.state_dict().items():
        if not torch.equal(weight.cpu(), states[0]['visual'][key]) or not torch.equal(expert[key], states[216]['visual'][key]):
            raise ValueError(f'Trajectory endpoint differs at {key}')
    data, _ = splits('DTD', encoder.val_preprocess, encoder.val_preprocess)
    lookup = {raw: i for i, raw in enumerate(data.indices)}
    groups = []
    for ids, labels in zip(manifest['raw_train_indices'], manifest['labels']):
        samples = [data[lookup[i]] for i in ids]
        if [y for _, y in samples] != labels:
            raise ValueError('Probe labels changed')
        groups.append((torch.stack([x for x, _ in samples]).cuda(), torch.tensor(labels, device='cuda')))
    provenance = dict(base_sha256=base_hash, expert_sha256=expert_manifest['checkpoint_sha256'],
                      head_sha256=expert_manifest['head_sha256'], trace_sha256=existing.sha(args.trace / 'trace.json'),
                      raw_train_indices=manifest['raw_train_indices'], rank=RANK)
    path = args.output / 'manifest.json'
    if path.exists() and read(path) != provenance:
        raise ValueError('Cached output belongs to a different trajectory')
    existing.write(path, provenance)
    op = Operator(model, visual)
    if args.audit_five_sweeps:
        existing.write(args.output / 'five_sweep_audit.json', dict(provenance=provenance,
                       anchors=audit_five_sweeps(op, states, groups)))
        return
    residual_path = args.output / 'residual_replay.json'
    if not residual_path.exists():
        replay(op, states, groups, residual_path)
    results = []
    for step in ANCHORS:
        path = args.output / f'ggn_{step:04d}.json'
        if not path.exists():
            op.generator.manual_seed(20261109 + step)
            existing.write(path, analyze_anchor(op, states, step, groups))
        results.append(read(path))
    existing.write(args.output / 'curvature.json', dict(provenance=provenance, anchors=results))


if __name__ == '__main__':
    main()
