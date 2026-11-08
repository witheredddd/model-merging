"""DTD functional atlas with trainable charts and a frozen composition interface."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class Mode(nn.Module):
    def __init__(self, weights, width):
        super().__init__()
        self.left = nn.ParameterList()
        self.right = nn.ParameterList()
        for group in weights:
            count, rows, cols = group.shape
            self.left.append(nn.Parameter(group.new_zeros(2, count, rows, width)))
            self.right.append(nn.Parameter(group.new_empty(2, count, width, cols).normal_(std=cols ** -.5)))


class Chart(nn.Module):
    def __init__(self, weights, width, positions, section, seam=None):
        super().__init__()
        self.modes = nn.ModuleList([Mode(weights, width)])
        self.code = weights[0].new_tensor([.5] if seam is None else [.5, 0.]).requires_grad_()
        self.seam = seam
        self.gains = nn.Parameter(weights[0].new_ones(sum(len(group) for group in weights)))
        self.positions = positions
        for name, value in zip(('section_v', 'section_o', 'section_basis'), section):
            self.register_buffer(name, value.to(weights[0]))

    def delta(self, code):
        # Zero-preserving and nested under growth. Each adjacent pair shares a nonlinear term.
        seam_code = None
        if self.seam is not None:
            seam_code, code = code[-1], code[:-1]
        quadratic = code.square()
        quadratic = quadratic + .5 * torch.cat([code.new_zeros(1), code[:-1] * code[1:]])
        coefficients = torch.stack([code, quadratic], dim=1)
        result = []
        for group, gates in enumerate(self.gains.split([p.shape[1] for p in self.modes[0].left])):
            left = torch.cat([(c[:, None, None, None] * m.left[group]).permute(1, 2, 0, 3).flatten(2)
                              for m, c in zip(self.modes, coefficients)], dim=2)
            right = torch.cat([m.right[group].permute(1, 0, 2, 3).flatten(1, 2)
                               for m in self.modes], dim=1)
            result.append(torch.bmm(left, right) * gates[:, None, None])
        if self.section_basis.shape[1]:
            (vg, vi), (og, oi) = self.positions
            h = self.section_v.shape[0]
            start = result[vg].shape[1] * 2 // 3
            value, out = result[vg][vi, start:start + h], result[og][oi, :, :h]
            overlap = torch.stack([(value * self.section_v).sum(), (out * self.section_o).sum()])
            discard = self.section_basis @ (self.section_basis.T @ overlap)
            corrected_v, corrected_o = result[vg].clone(), result[og].clone()
            corrected_v[vi, start:start + h] = value - discard[0] * self.section_v
            corrected_o[oi, :, :h] = out - discard[1] * self.section_o
            result[vg], result[og] = corrected_v, corrected_o
        if seam_code is not None:
            result = [r + seam_code * a + seam_code.square() * b for r, (a, b) in zip(result, self.seam)]
        return result

    def grow(self, weights, width):
        mode = Mode(weights, width)
        self.modes.append(mode)
        code, zero = self.code.detach(), self.code.new_zeros(1)
        pieces = (code, zero) if self.seam is None else (code[:-1], zero, code[-1:])
        self.code = torch.cat(pieces).requires_grad_()
        return mode


@torch.no_grad()
def inherited_seam(parent, code):
    """Extract the exact coefficients of the parent's quadratic curve along axis zero."""
    axis = torch.zeros_like(code)
    axis[0] = 1
    middle = parent.delta(code)
    upper, lower = parent.delta(code + axis), parent.delta(code - axis)
    return [((p - m) / 2, (p + m) / 2 - c) for p, m, c in zip(upper, lower, middle)]


def logits(model, groups, weights, images):
    params = {name: matrix for names, block in zip(groups, weights) for name, matrix in zip(names, block)}
    return torch.func.functional_call(model, params, (images,)).float()


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def embedding(z):
    # Concatenation embeds the empirical product of categorical Fisher spheres isometrically.
    return (2 / math.sqrt(len(z)) * z.double().log_softmax(-1).mul(.5).exp()).flatten()


def fr_distance(a, b):
    chords = (a.double().softmax(-1).sqrt() - b.double().softmax(-1).sqrt()).norm(dim=-1)
    return float((4 * (chords / 2).clamp(max=1).asin()).square().mean().sqrt())


def supported_fibers(fit, audit, relative=.03):
    """Find weak normalized response directions on fit, require replication on audit."""
    matrix = fit.T @ fit
    values, modes = torch.linalg.eigh(matrix)
    fit_scale = matrix.diag().median().clamp_min(1e-20)
    audit_scale = audit.square().sum(0).median().clamp_min(1e-20)
    ratios = (audit @ modes).square().sum(0) / audit_scale
    accepted = (values <= relative * fit_scale) & (ratios <= relative)
    return modes[:, accepted], dict(fit_relative=(values / fit_scale).tolist(),
                                   audit_relative=ratios.tolist(), accepted=int(accepted.sum()))


@torch.no_grad()
def local_geometry(predict, code, fit, audit):
    """Empirical first and second fundamental forms in square-root prediction space."""
    images = torch.cat([fit, audit])
    n = len(fit)
    z = predict(code, images)
    center = embedding(z[:n])
    step = .04
    axes = torch.eye(len(code), device=code.device)
    columns, check_columns = [], []
    hessian = center.new_empty((len(center), len(code), len(code)))
    for i, axis in enumerate(axes):
        plus = predict(code + step * axis, images)
        minus = predict(code - step * axis, images)
        upper, lower = embedding(plus[:n]), embedding(minus[:n])
        columns.append((upper - lower) / (2 * step))
        check_columns.append((embedding(plus[n:]) - embedding(minus[n:])) / (2 * step))
        hessian[:, i, i] = (upper + lower - 2 * center) / step ** 2
        for j in range(i):
            other = axes[j]
            mixed = sum(s * t * embedding(predict(code + step * (s * axis + t * other), fit))
                        for s in (-1, 1) for t in (-1, 1)) / (4 * step ** 2)
            hessian[:, i, j] = hessian[:, j, i] = mixed
    jacobian = torch.stack(columns, dim=1)
    check = torch.stack(check_columns, dim=1)
    fibers, fiber_row = supported_fibers(jacobian, check)
    horizontal = torch.eye(len(code), device=code.device, dtype=torch.double) - fibers @ fibers.T
    tangent = jacobian @ horizontal
    gram = tangent.T @ tangent
    flat = (horizontal @ hessian @ horizontal).flatten(1)
    normal = (flat - tangent @ (torch.linalg.pinv(tangent, rtol=1e-5) @ flat)).view_as(hessian)
    sectional = None
    if len(code) >= 2:
        area = gram[0, 0] * gram[1, 1] - gram[0, 1].square()
        if area > 1e-10 * gram.diag().max().square().clamp_min(1e-10):
            sectional = float((normal[:, 0, 0] @ normal[:, 1, 1] - normal[:, 0, 1].square().sum()) / area)
    return gram, horizontal, normal, dict(latent_fibers=fiber_row, curvature_norm=float(normal.norm()),
                                          sectional_01=sectional, audit_gram=(check.T @ check).tolist())


def coordinate_step(grad, geometry, lr, radius):
    gram, horizontal, normal, _ = geometry
    damping = .02 * gram.diag().mean().clamp_min(1e-4)
    step = -lr * (horizontal @ torch.linalg.solve(gram + damping * torch.eye(len(grad), device=grad.device),
                                                 horizontal @ grad.double()))
    scale = min(1., .04 / max(float(step.norm()), 1e-20),
                radius / max(float((step @ gram @ step).clamp_min(0).sqrt()), 1e-20))
    step = step * scale
    curvature = float((normal @ step @ step).norm()) / 2
    # Bound the second-order prediction error as well as the first-order Fisher length.
    curvature_scale = 1 / math.sqrt(1 + curvature / (radius * .06))
    return (curvature_scale * step).float(), curvature_scale


@torch.no_grad()
def block_geometry(model, groups, center, deltas, fit, audit):
    """Explore independent matrix perturbations, including combinations off the SGD path."""
    images = torch.cat([fit, audit])
    params = {name: matrix for names, block in zip(groups, center) for name, matrix in zip(names, block)}
    norms, columns = [], []
    for names, block in zip(groups, deltas):
        for name, matrix in zip(names, block):
            norm = matrix.norm().clamp_min(1e-7)
            norms.append(norm)

            def call(weight):
                changed = dict(params, **{name: weight})
                return torch.func.functional_call(model, changed, (images,)).float()

            pred, response = torch.func.jvp(call, (params[name],), (matrix / norm,))
            p = pred.double().softmax(-1)
            dz = response.double()
            columns.append((dz - (p * dz).sum(-1, keepdim=True)) * p.sqrt())
    responses = torch.stack(columns, dim=-1)
    fit_j = responses[:len(fit)].flatten(0, 1) / math.sqrt(len(fit))
    audit_j = responses[len(fit):].flatten(0, 1) / math.sqrt(len(audit))
    fibers, row = supported_fibers(fit_j, audit_j)
    # Compare off-diagonal cosine structures from independent images.
    def cosines(jac):
        directions = F.normalize(jac, dim=0)
        return (directions.T @ directions)[torch.triu_indices(jac.shape[1], jac.shape[1], 1, device=jac.device).unbind()]

    row['cross_batch_coupling_cosine'] = float(F.cosine_similarity(cosines(fit_j), cosines(audit_j), dim=0))
    row['gate_norms'] = torch.stack(norms).tolist()
    row['fiber_basis'] = fibers.T.tolist()
    return fibers, torch.stack(norms).double(), row


@torch.no_grad()
def attention_fiber(model, groups, weights, found, fit, audit):
    """Exact V/O gauge orbit on one attention head, with an off-orbit intervention."""
    images = torch.cat([fit, audit])
    params = {name: matrix for names, group in zip(groups, weights) for name, matrix in zip(names, group)}
    name = next(name for name, _ in found if name.endswith('attn.in_proj_weight'))
    prefix = 'image_encoder.' + name.rsplit('.', 1)[0]
    value = params[prefix + '.in_proj_weight'].clone()
    out = params[prefix + '.out_proj.weight'].clone()
    bias_name = prefix + '.in_proj_bias'
    bias = dict(model.named_parameters())[bias_name].detach().clone()
    start = 2 * out.shape[0]
    sl = slice(start, start + 64)
    scale = torch.linspace(.8, 1.2, 64, device=out.device)
    value[sl] *= scale[:, None]
    bias[sl] *= scale
    out[:, :64] /= scale
    base = torch.func.functional_call(model, params, (images,))
    params[prefix + '.in_proj_weight'] = value
    params[bias_name] = bias
    off_orbit = torch.func.functional_call(model, params, (images,))
    params[prefix + '.out_proj.weight'] = out
    same = torch.func.functional_call(model, params, (images,))
    row = dict(gauge_fit_fr=fr_distance(base[:len(fit)], same[:len(fit)]),
               gauge_audit_fr=fr_distance(base[len(fit):], same[len(fit):]),
               broken_gauge_fr=fr_distance(base, off_orbit))
    if row['gauge_audit_fr'] > 2e-5 or row['broken_gauge_fr'] < 1e-4:
        raise RuntimeError(f'Attention fiber control failed: {row}')
    return row


@torch.no_grad()
def learn_section(model, groups, weights, positions, fit, audit):
    """Fit and verify a bias-compatible attention fiber in two independent ambient axes."""
    (vg, vi), (og, oi) = positions
    value, out = weights[vg][vi], weights[og][oi]
    name = groups[vg][vi]
    start = value.shape[0] * 2 // 3
    bias = dict(model.named_parameters())[name.removesuffix('weight') + 'bias'][start:start + 64]
    index = torch.arange(64, device=value.device, dtype=value.dtype)
    u = F.normalize(index.sin(), dim=0)
    span = torch.stack([u, F.normalize(bias, dim=0)], dim=1)
    q = torch.linalg.qr(span).Q
    v = F.normalize((index * 1.7).cos() - q @ (q.T @ (index * 1.7).cos()), dim=0)
    h = torch.outer(u, v)
    v_axis = h @ value[start:start + 64]
    o_axis = out[:, :64] @ h
    v_axis = v_axis / v_axis.norm().clamp_min(1e-10)
    o_axis = o_axis / o_axis.norm().clamp_min(1e-10)
    dv, do = torch.zeros_like(value), torch.zeros_like(out)
    dv[start:start + 64] = v_axis
    do[:, :64] = o_axis
    params = {n: m for names, group in zip(groups, weights) for n, m in zip(names, group)}
    out_name = groups[og][oi]
    origin = value.new_zeros(2)

    def embedded(a, images):
        overrides = dict(params, **{name: value + a[0] * dv, out_name: out + a[1] * do})
        return embedding(torch.func.functional_call(model, overrides, (images,)))

    def jac(images):
        return torch.stack([torch.func.jvp(lambda a: embedded(a, images), (origin,), (e,))[1]
                            for e in torch.eye(2, device=value.device)], dim=1)

    fibers, row = supported_fibers(jac(fit), jac(audit), relative=.001)
    if fibers.shape[1] != 1:
        raise RuntimeError(f'Could not reproduce the attention quotient section: {row}')
    row['finite_audit_shift'] = float((embedded(.25 * fibers[:, 0].float(), audit)
                                      - embedded(origin, audit)).norm())
    row['learned_fiber'] = fibers.T.tolist()
    return (v_axis, o_axis, fibers), row


@torch.no_grad()
def chart_transition(old, new, old_center, anchor, model, groups, fit, audit):
    def derivatives(chart, code, base, x):
        cols = []
        for axis in torch.eye(len(code), device=code.device):
            _, response = torch.func.jvp(lambda a: embedding(logits(model, groups, add(base, chart.delta(a)), x)),
                                        (code,), (axis,))
            cols.append(response)
        return torch.stack(cols, dim=1)

    z = torch.zeros_like(new.code)
    old_fit = derivatives(old, old.code.detach(), old_center, fit)
    new_fit = derivatives(new, z, anchor, fit)
    conversion = torch.linalg.pinv(old_fit, rtol=1e-4) @ new_fit
    old_audit = derivatives(old, old.code.detach(), old_center, audit)
    new_audit = derivatives(new, z, anchor, audit)
    seam_errors = []
    for s in (-.1, -.05, .05, .1):
        parent_code, child_code = old.code.detach().clone(), z.clone()
        parent_code[0] += s
        child_code[-1] = s
        a, b = add(old_center, old.delta(parent_code)), add(anchor, new.delta(child_code))
        seam_errors.append(max(float((x - y).abs().max()) for x, y in zip(a, b)))
    if max(seam_errors) > 2e-6:
        raise RuntimeError(f'Shared coordinate transition failed: {seam_errors}')
    return dict(matrix=conversion.tolist(), seam_parameter_max_error=max(seam_errors),
                seam_parent_axis=0, seam_child_axis=len(z) - 1,
                fit_residual=float((old_fit @ conversion - new_fit).norm()
                / new_fit.norm().clamp_min(1e-12)), audit_residual=float((old_audit @ conversion - new_audit).norm()
                / new_audit.norm().clamp_min(1e-12)))


def compose_atlas(bundle, base, codes=None):
    """Reconstruct from the same base or evaluate replacement EC coordinates."""
    result = base
    entries = bundle['charts']
    codes = [entry['code'] for entry in entries] if codes is None else codes
    if len(codes) != len(entries):
        raise ValueError('One coordinate vector is required per chart')
    parent, parent_code = None, None
    for entry, code in zip(entries, codes):
        if (tuple(code.shape) != tuple(entry['code'].shape) or not torch.isfinite(code).all()
                or (code.abs() > 1.5).any() or (parent is not None and code[-1].abs() > .120001)):
            raise ValueError('Coordinate outside the chart domain')
        seam = None if parent is None else inherited_seam(parent, parent_code)
        chart = Chart(base, bundle['width'], bundle['positions'],
                      tuple(entry['map'][k] for k in ('section_v', 'section_o', 'section_basis')), seam=seam)
        for _ in range(1, entry['private_rank']):
            chart.grow(base, bundle['width'])
        chart.load_state_dict(entry['map'])
        result = add(result, chart.delta(code.to(base[0])))
        parent, parent_code = chart, entry['code'].to(base[0])
    return result


@torch.no_grad()
def audit_coupling(bundle, model, base, frozen, pools):
    groups = bundle['groups']
    depths = [[int(n.split('resblocks.')[1].split('.')[0]) // 3 for n in names] for names in groups]
    codes = [entry['code'] for entry in bundle['charts']]
    zero = [torch.zeros_like(code) for code in codes]

    def restrict(blocks, depth):
        return [block * block.new_tensor([i in depth for i in level])[:, None, None]
                for block, level in zip(blocks, depths)]

    for chart, code in enumerate(codes):
        sample = list(codes)
        direction = torch.zeros_like(code)
        direction[0] = -.025 if code[0] > 1.45 else .025
        sample[chart] = code + direction
        moved = compose_atlas(bundle, base, sample)
        tangent = [(a - b) / float(direction[0]) for a, b in zip(moved, frozen)]
        sample = list(zero)
        sample[chart] = code
        single = compose_atlas(bundle, base, sample)
        displacement = [a - b for a, b in zip(single, base)]
        rows = []
        for images, labels in pools:
            columns = []
            scalar = frozen[0].new_zeros(())
            baseline = F.cross_entropy(logits(model, groups, frozen, images), labels)

            def remove(depth):
                changed = [w - d for w, d in zip(frozen, restrict(displacement, depth))]
                return F.cross_entropy(logits(model, groups, changed, images), labels)

            effects = [remove({i}) - baseline for i in range(4)]
            for i in range(4):
                block = restrict(tangent, {i})

                def predict(t):
                    weights = [w + t * d for w, d in zip(frozen, block)]
                    return embedding(logits(model, groups, weights, images))

                columns.append(torch.func.jvp(predict, (scalar,), (torch.ones_like(scalar),))[1])
            matrix = F.normalize(torch.stack(columns), dim=1)
            gram = matrix @ matrix.T
            edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
            rows.append(dict(fisher_cosines=[float(gram[i, j]) for i, j in edges],
                             drop_ce=[float(v) for v in effects],
                             interaction_ce=[float(remove({i, j}) - baseline - effects[i] - effects[j])
                                             for i, j in edges]))
        a, b = (torch.tensor(row['fisher_cosines']) for row in rows)
        stable = (a.abs() >= .10) & (b.abs() >= .10) & (a.sign() == b.sign())
        yield dict(chart=chart, pools=rows, same_sign_edges=int(stable.sum()),
                   cross_pool_cosine=float(F.cosine_similarity(a, b, dim=0)))


def save(path, data):
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def train(args):
    sys.path[:0] = [str(args.root), str(args.root / 'scripts'), str(args.root / 'scripts/tfpml')]
    import dtd_grassmann4 as existing
    import runtime

    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    existing.seed_all(args.seed)
    existing.ROOT = args.root
    head_path = args.root / 'models/checkpoints/ViT-B-16/head_DTDVal.rebuilt.pt'
    info = json.loads(head_path.with_suffix('.json').read_text())
    if existing.sha(head_path) != info['sha256']:
        raise RuntimeError('DTD head checksum mismatch')
    head = existing.ClassificationHead.load(str(head_path))
    encoder, model, found, initial = runtime.build_model(existing, head)
    model.requires_grad_(False).eval()
    shapes = {}
    for name, parameter in found:
        shapes.setdefault(parameter.shape, []).append(('image_encoder.' + name, parameter.detach()))
    groups = [[n for n, _ in pairs] for pairs in shapes.values()]
    base = [torch.stack([p for _, p in pairs]) for pairs in shapes.values()]
    assert len(found) == 48 and initial.numel() == 84934656
    del initial
    trainset, _, validation, splits = runtime.training_data(existing, encoder, args.seed)
    assert [c.replace('_', ' ') for c in splits.train_dataset.dataset.classes] == info['classnames']
    order = torch.randperm(len(trainset), generator=torch.Generator().manual_seed(900)).tolist()
    fit_ids, audit_ids, sgd_ids = order[:48], order[48:96], order[96:]

    def probe(ids):
        examples = [splits.train_dataset[i] for i in ids]
        return torch.stack([x for x, _ in examples]).cuda(), torch.tensor([y for _, y in examples], device='cuda')

    fit, fit_y = probe(fit_ids[:args.probe])
    audit, audit_y = probe(audit_ids[:args.probe])
    value_name = next(n for names in groups for n in names if n.endswith('attn.in_proj_weight'))
    out_name = value_name.removesuffix('in_proj_weight') + 'out_proj.weight'
    locations = {n: (i, j) for i, names in enumerate(groups) for j, n in enumerate(names)}
    positions = [locations[n] for n in (value_name, out_name)]
    anchor = base
    completed = []
    age, step = 0, 0
    geometry, fibers, norms = None, None, None
    best = float('inf')
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    def record(event, **data):
        row = dict(event=event, step=step, **data)
        with (output / 'metrics.jsonl').open('a') as out:
            out.write(json.dumps(row, allow_nan=False) + '\n')
        print(row, flush=True)

    def predict(code, images):
        return logits(model, groups, add(anchor, chart.delta(code)), images)

    @torch.no_grad()
    def evaluate(weights):
        correct, loss, count = 0, 0., 0
        for x, y in validation:
            z = logits(model, groups, weights, x.cuda())
            y = y.cuda()
            loss += float(F.cross_entropy(z, y, reduction='sum'))
            correct += int((z.argmax(-1) == y).sum())
            count += len(y)
        return dict(ce=loss / count, accuracy=correct / count, n=count)

    def entry(c):
        return dict(code=c.code.detach().cpu(), private_rank=len(c.modes),
                    map={k: v.detach().cpu() for k, v in c.state_dict().items()})

    def bundle():
        return dict(schema=2, groups=groups, positions=positions, width=args.width,
                    base_sha256=base_sha, head_sha256=info['sha256'],
                    private_domain=[-1.5, 1.5], seam_domain=[-.12, .12],
                    charts=[entry(c) for c, _ in completed] + [entry(chart)])

    base_sha = existing.sha(head_path.parent / 'pretrained.pt')
    record('config', seed=args.seed, epochs=args.epochs, width=args.width, lr=args.lr,
           probe=args.probe, refresh=args.refresh, max_rank=args.max_rank, max_charts=args.max_charts,
           fr_region=args.fr_region, train=len(sgd_ids), fit=len(fit_ids), audit=len(audit_ids),
           base_sha256=base_sha, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    record('gauge_control', **attention_fiber(model, groups, base, found, fit, audit))
    record('validation', epoch=0, **evaluate(base))

    def make_chart(center, seam=None):
        section, row = learn_section(model, groups, center, positions, fit, audit)
        record('attention_section', chart=len(completed), **row)
        local_chart = Chart(base, args.width, positions, section, seam=seam)
        opt = torch.optim.AdamW(local_chart.modes.parameters(), lr=args.lr, weight_decay=.01, fused=True)
        return local_chart, opt

    chart, optimizer = make_chart(anchor)

    def try_growth():
        # A trial chart mode is fit on one calibration split and accepted on the other.
        old_code = chart.code
        mode = chart.grow(base, args.width)
        trial_code = chart.code.detach().clone()
        index = len(chart.modes) - 1
        trial_code[index] = .25
        before_fit = predict(chart.code.detach(), fit).detach()
        before_audit = predict(chart.code.detach(), audit).detach()
        trial = torch.optim.AdamW(mode.parameters(), lr=args.lr, foreach=False)
        for _ in range(3):
            trial.zero_grad(set_to_none=True)
            loss = F.cross_entropy(predict(trial_code, fit), fit_y)
            grads = torch.autograd.grad(loss, tuple(mode.parameters()))
            for p, g in zip(mode.parameters(), grads):
                p.grad = g
            trial.step()
        with torch.no_grad():
            after_fit, after_audit = predict(trial_code, fit), predict(trial_code, audit)
            gain_fit = float(F.cross_entropy(before_fit, fit_y) - F.cross_entropy(after_fit, fit_y))
            gain_audit = float(F.cross_entropy(before_audit, audit_y) - F.cross_entropy(after_audit, audit_y))
            old_geo = local_geometry(predict, chart.code.detach(), fit, audit)
            def novelty(gram):
                other = [i for i in range(len(gram)) if i != index]
                v = gram[index, other]
                residual = gram[index, index] - v @ torch.linalg.pinv(gram[other][:, other], rtol=1e-5) @ v
                return float((residual / gram[index, index].clamp_min(1e-15)).clamp(0, 1))

            fit_novelty = novelty(old_geo[0])
            audit_novelty = novelty(old_geo[0].new_tensor(old_geo[-1]['audit_gram']))
        accepted = gain_fit > .003 and gain_audit > .002 and min(fit_novelty, audit_novelty) > .15
        record('rank_trial', accepted=accepted, fit_gain=gain_fit, audit_gain=gain_audit,
               fit_novelty=fit_novelty, audit_novelty=audit_novelty)
        if accepted:
            optimizer.add_param_group({'params': mode.parameters()})
            return True
        del chart.modes[-1]
        chart.code = old_code
        return False

    @torch.no_grad()
    def center_logits():
        return predict(chart.code, fit), predict(chart.code, audit)

    chart_origin = center_logits()
    curvature_limited = 0
    for epoch in range(1, args.epochs + 1):
        trainset.epoch = epoch - 1
        shuffle = torch.randperm(len(sgd_ids), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        batches = [[sgd_ids[i] for i in shuffle[start:start + args.batch]]
                   for start in range(0, len(shuffle), args.batch)]
        loader = torch.utils.data.DataLoader(trainset, batch_sampler=batches, num_workers=4, pin_memory=True)
        for x, y, _ in loader:
            step += 1
            age += 1
            if age % 24 == 0 and len(chart.modes) < args.max_rank and try_growth():
                geometry = None
            if geometry is None or age % args.refresh == 1:
                geometry = local_geometry(predict, chart.code.detach(), fit, audit)
                record('local_geometry', chart=len(completed), rank=len(chart.code), **geometry[-1])
            if age % 48 == 12:
                with torch.no_grad():
                    displacement = chart.delta(chart.code)
                    raw = [d / gain[:, None, None] for d, gain in zip(displacement, chart.gains.split([len(b) for b in base]))]
                    fibers, norms, data = block_geometry(model, groups, add(anchor, displacement), raw, fit, audit)
                    record('block_fibers', chart=len(completed), **data)
            optimizer.zero_grad(set_to_none=True)
            chart.code.grad = None
            chart.gains.grad = None
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            # Nearby coordinates expose the map to task loss around its current operating point.
            for offset in range(0, len(y), 16):
                code = chart.code
                if step % 4 == 0:
                    axis = (step // 4) % len(code)
                    jitter = torch.zeros_like(code)
                    jitter[axis] = .06 if offset % 32 == 0 else -.06
                    code = code + jitter
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = predict(code, x[offset:offset + 16])
                    loss = F.cross_entropy(pred, y[offset:offset + 16], reduction='sum') / len(y)
                loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta, shrink = coordinate_step(chart.code.grad, geometry, .01, .035)
                curvature_limited += int(shrink < .99)
                chart.code.add_(delta).clamp_(-1.5, 1.5)
                if chart.seam is not None:
                    chart.code[-1].clamp_(-.12, .12)
                gain_step = -.01 * chart.gains.grad.double()
                if fibers is not None:
                    scaled = gain_step * norms
                    gain_step = (scaled - fibers @ (fibers.T @ scaled)) / norms
                gain_scale = min(1., .01 / max(float(gain_step.abs().max()), 1e-20))
                gain_margin = torch.where(gain_step >= 0, 1.5 - chart.gains, chart.gains - .5)
                gain_scale = min(gain_scale, float((gain_margin / gain_step.abs().clamp_min(1e-20)).min()))
                chart.gains.add_(gain_step, alpha=max(0., gain_scale))
            if age % args.refresh == 0:
                z_fit, z_audit = center_logits()
                distance = min(fr_distance(chart_origin[0], z_fit), fr_distance(chart_origin[1], z_audit))
                record('step', chart=len(completed), age=age, rank=len(chart.code),
                       curvature_scale=shrink, region_distance=distance, code=chart.code.tolist())
                if age >= 48 and distance > args.fr_region and len(completed) + 1 < args.max_charts:
                    # The new chart's zero coincides with the old chart's departing coordinate.
                    completed.append((chart, anchor))
                    with torch.no_grad():
                        anchor = add(anchor, chart.delta(chart.code))
                    seam = inherited_seam(chart, chart.code.detach())
                    chart, optimizer = make_chart(anchor, seam)
                    chart_origin = center_logits()
                    age, geometry, fibers, norms = 0, None, None, None
                    record('spawn', chart=len(completed), previous_region_distance=distance)
            if args.steps and step == args.steps:
                break
        with torch.no_grad():
            live = add(anchor, chart.delta(chart.code))
        score = evaluate(live)
        record('validation', epoch=epoch, **score)
        state = bundle()
        save(output / 'atlas_latest.pt', state)
        if score['ce'] < best:
            best = score['ce']
            best_epoch = epoch
            save(output / 'atlas_best.pt', state)
        if args.steps or (epoch >= args.min_epochs and score['accuracy'] >= .80):
            break

    selected = torch.load(output / 'atlas_best.pt', weights_only=True, map_location='cpu')
    with torch.no_grad():
        reconstructed = compose_atlas(state, base)
        error = max(float((a - b).abs().max()) for a, b in zip(live, reconstructed))
        if error > 2e-6:
            raise RuntimeError(f'Atlas round trip error: {error}')
        record('round_trip', parameter_max_error=error, best_epoch=best_epoch,
               ranks=[len(c['code']) for c in selected['charts']])
        for i, (old, old_anchor) in enumerate(completed):
            new, new_anchor = (completed + [(chart, anchor)])[i + 1]
            record('transition', source=i, target=i + 1,
                   **chart_transition(old, new, old_anchor, new_anchor, model, groups, fit, audit))
        selected_weights = compose_atlas(selected, base)
        record('selected_validation', epoch=best_epoch, **evaluate(selected_weights))
        pools = [probe(ids[args.probe:args.probe + 16]) for ids in (fit_ids, audit_ids)]
        for row in audit_coupling(selected, model, base, selected_weights, pools):
            record('frozen_coupling', **row)
        encoder_state = {k: v.cpu() for k, v in encoder.state_dict().items()}
        encoder_state.update({n.removeprefix('image_encoder.'): p.cpu()
                              for names, block in zip(groups, selected_weights) for n, p in zip(names, block)})
        save(output / 'encoder.pt', encoder_state)
    record('complete', curvature_limited_steps=curvature_limited)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--min-epochs', type=int, default=4)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=4)
    parser.add_argument('--lr', type=float, default=.0015)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--probe', type=int, default=8)
    parser.add_argument('--refresh', type=int, default=8)
    parser.add_argument('--max-rank', type=int, default=3)
    parser.add_argument('--max-charts', type=int, default=4)
    parser.add_argument('--fr-region', type=float, default=.75)
    args = parser.parse_args()
    if not (1 <= args.min_epochs <= args.epochs and 2 <= args.refresh and 2 <= args.probe <= 32
            and args.width > 0 and args.batch > 0 and args.steps >= 0 and args.lr > 0
            and args.max_rank > 0 and args.max_charts > 0 and args.fr_region > 0):
        parser.error('Invalid training budget')
    train(args)
