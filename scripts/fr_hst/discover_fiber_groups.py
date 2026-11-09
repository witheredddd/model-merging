"""Generalized Fisher modes of complete expert updates and finite interventions."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from build_dtd_fiber_atlas import distance, load_dtd_points
from fiber_atlas_experts import load_task, probe, score
from train_atlas import embedding


TASKS = ('DTD', 'SUN397')


def load_pair(root, dtd_path, sun_path):
    tasks = [load_task(root, name) for name in TASKS]
    dtd, sun = tasks
    names = [n for n, _ in dtd.found]
    if sun.base_sha256 != dtd.base_sha256 or [n for n, _ in sun.found] != names:
        raise RuntimeError('Experts have different base models or matrix order')
    info = json.loads((sun_path / 'complete.json').read_text())
    if (info['base_sha256'] != sun.base_sha256 or info['head_sha256'] != sun.head_sha256
            or info['task'] != 'SUN397' or info['names'] != names):
        raise RuntimeError('SUN397 expert provenance mismatch')
    selected = info['selected']['step']
    sun_weight = torch.load(sun_path / f'checkpoint_{selected:04d}.pt', map_location='cpu',
                            mmap=True, weights_only=True)['weights'].cuda()
    manifest = json.loads((dtd_path / 'real42/complete.json').read_text())
    step = manifest['epoch_2']['step']
    dtd_weight = load_dtd_points(dtd_path, dtd, {(42, step)})[42, step].cuda()
    return tasks, dtd.base, [dtd_weight, sun_weight], [manifest['epoch_2'], info['selected']]


def segments(task):
    start = 0
    result = []
    for name, p in task.found:
        result.append(('image_encoder.' + name, p.shape, slice(start, start + p.numel())))
        start += p.numel()
    return result


def decode(delta, coefficients, parts):
    result = torch.empty_like(delta)
    for c, (_, _, section) in zip(coefficients.tolist(), parts):
        result[section] = c * delta[section]
    return result


@torch.no_grad()
def fisher_gram(task, center, unit, parts, images):
    params = {name: center[section].view(shape) for name, shape, section in parts}
    columns = []
    for name, shape, section in parts:
        def forward(weight):
            changed = dict(params, **{name: weight})
            return embedding(torch.func.functional_call(task.model, changed, (images,)).float())

        columns.append(torch.func.jvp(forward, (params[name],), (unit[section].view(shape),))[1])
    y = torch.stack(columns, dim=1)
    return (y.T @ y).cpu()


def generalized(a, b):
    total = a + b
    ridge = .01 * total.diag().median().clamp_min(1e-12)
    e, q = torch.linalg.eigh(total + ridge * torch.eye(len(a), dtype=a.dtype))
    whiten = q * e.rsqrt()
    values, vectors = torch.linalg.eigh(whiten.T @ a @ whiten)
    return values, whiten @ vectors, float(ridge)


def classify(vector, matrices):
    length = vector.square().sum().clamp_min(1e-20)
    energy = [float((vector @ c @ vector).clamp_min(0) / length) for c in matrices]
    floors = [.02 * float(c.diag().median()) for c in matrices]
    if energy[0] <= floors[0] and energy[1] <= floors[1]:
        label = 'null'
    elif energy[0] >= 5 * energy[1] and energy[0] > floors[0]:
        label = 'DTD'
    elif energy[1] >= 5 * energy[0] and energy[1] > floors[1]:
        label = 'SUN397'
    else:
        label = 'shared'
    return label, energy


@torch.no_grad()
def intervene(tasks, center, delta, pools):
    before = [task.predict(center, x) for task, (x, _) in zip(tasks, pools)]
    rows = []
    for amount in (.25, .5, 1.):
        effects = []
        for task, (x, y), old in zip(tasks, pools, before):
            new = task.predict(center + amount * delta, x)
            effects.append(dict(ce_gain=float(F.cross_entropy(old, y) - F.cross_entropy(new, y)),
                                accuracy_gain=float(((new.argmax(1) == y).float() - (old.argmax(1) == y).float()).mean()),
                                fr=float(distance(old.double().softmax(-1).sqrt(), new.double().softmax(-1).sqrt()))))
        rows.append(dict(amount=amount, effects=effects))
    return rows


def choose(label, rows):
    if label == 'shared':
        passed = [r for r in rows if min(x['fr'] for x in r['effects']) >= .02
                  and max(x['ce_gain'] for x in r['effects']) >= .01]
        return max(passed, key=lambda r: sum(x['ce_gain'] for x in r['effects']), default=None)
    target = TASKS.index(label)
    passed = [r for r in rows if r['effects'][target]['ce_gain'] >= .02
              and r['effects'][1-target]['fr'] <= .10 and r['effects'][1-target]['ce_gain'] >= -.02
              and r['effects'][1-target]['accuracy_gain'] >= -.01]
    return max(passed, key=lambda r: r['effects'][target]['ce_gain'], default=None)


def fusion_splits(tasks, probability):
    excluded = set()
    if probability:
        excluded = {i for p in json.loads(probability.read_text())['probes'] for i in p['validation_local_indices']}
    confirmation, final = [], []
    for i, task in enumerate(tasks):
        ids = [j for j in range(len(task.validation)) if i != 0 or j not in excluded]
        order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(1701)).tolist()
        split = len(ids) // 3
        confirmation.append(torch.utils.data.Subset(task.validation, [ids[j] for j in order[:split]]))
        final.append(torch.utils.data.Subset(task.validation, [ids[j] for j in order[split:]]))
    return confirmation, final


@torch.no_grad()
def validation_shift(task, before, after, dataset):
    total = 0.
    for x, _ in torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4):
        x = x.cuda()
        a, b = [task.predict(t, x).double().softmax(-1).sqrt() for t in (before, after)]
        total += float(distance(a, b).square()) * len(x)
    return (total / len(dataset)) ** .5


@torch.no_grad()
def run(args):
    tasks, base, experts, summaries = load_pair(args.root, args.dtd, args.sun)
    parts = segments(tasks[0])
    fit = [probe(t.fit, t.fit_ids[:12])[0] for t in tasks]
    audit = [probe(t.fit, t.audit_ids[:16])[0] for t in tasks]
    finite_pools = [probe(t.fit, t.audit_ids[16:]) for t in tasks]
    references = [base, *experts]
    val, final_data = fusion_splits(tasks, args.probability)
    scores = [[score(task, center, data) for task, data in zip(tasks, val)] for center in references]
    records, spec, interventions = [], [], []
    for source, expert in enumerate(experts):
        tau = expert - base
        norms = torch.tensor([float(tau[section].norm()) for _, _, section in parts], dtype=torch.double)
        if (norms <= 1e-9).any():
            raise RuntimeError('A complete expert matrix update is missing')
        unit = decode(tau, norms.reciprocal(), parts)
        fitted = [fisher_gram(t, base, unit, parts, x) for t, x in zip(tasks, fit)]
        audits = []
        for index, reference in enumerate(references):
            audits.append([fisher_gram(t, reference, unit, parts, x) for t, x in zip(tasks, audit)])
            print(f'{TASKS[source]} restricted Fisher reference {index} done', flush=True)
        values, directions, ridge = generalized(*fitted)
        coordinates = torch.linalg.solve(directions, norms)
        labels, diagnostics = [], []
        for v in directions.T:
            label, energy = classify(v, fitted)
            replicated = [classify(v, matrices) for matrices in audits]
            votes = [x[0] for x in replicated]
            # The intended intervention is at the other task's expert.
            accepted = votes.count(label) >= 2 and votes[2-source] == label
            labels.append(label if accepted else 'unresolved')
            diagnostics.append(dict(fit_label=label, fit_energy=energy, audit=replicated, replicated=accepted))
        kept = []
        for label in (TASKS[source], 'shared'):
            members = [i for i, l in enumerate(labels) if l == label]
            if not members:
                continue
            vector = directions[:, members] @ coordinates[members]
            coefficient = vector / norms
            audit_labels = [classify(vector, c)[0] for c in audits]
            row = dict(source=TASKS[source], label=label, members=members, audit_labels=audit_labels,
                       relative_parameter_norm=float(vector.norm() / norms.norm()))
            if audit_labels.count(label) >= 2 and audit_labels[2-source] == label:
                delta = decode(tau, coefficient, parts)
                trials = intervene(tasks, experts[1-source], delta, finite_pools)
                best = choose(label, trials)
                row.update(trials=trials, calibration=best)
                if best:
                    theta = experts[1-source] + best['amount'] * delta
                    final = [score(t, theta, ds) for t, ds in zip(tasks, val)]
                    old = scores[2-source]
                    check = [dict(ce_gain=a['ce'] - b['ce'], accuracy_gain=b['accuracy'] - a['accuracy'])
                             for a, b in zip(old, final)]
                    for effect, t, ds in zip(check, tasks, val):
                        effect['fr'] = validation_shift(t, experts[1-source], theta, ds)
                    row.update(validation=final, validation_effects=check)
                    if label == 'shared':
                        keep = max(x['ce_gain'] for x in check) > 0
                    else:
                        t = TASKS.index(label)
                        keep = (check[t]['ce_gain'] > 0 and check[t]['accuracy_gain'] >= 0
                                and check[1-t]['accuracy_gain'] >= -.01 and check[1-t]['ce_gain'] >= -.02
                                and check[1-t]['fr'] <= .10)
                    if keep:
                        kept.append(dict(label=label, coefficients=coefficient.tolist()))
                        row['accepted'] = True
            interventions.append(row)
        residual = torch.ones_like(norms)
        for g in kept:
            residual -= torch.tensor(g['coefficients'], dtype=torch.double)
        kept.append(dict(label='residual', coefficients=residual.tolist()))
        reconstruction = torch.stack([torch.tensor(g['coefficients'], dtype=torch.double) for g in kept]).sum(0)
        if (reconstruction - 1).abs().max() > 1e-8:
            raise RuntimeError('Group coordinates did not recover the expert update')
        spec.append(kept)
        whole = intervene(tasks, experts[1-source], tau, finite_pools)
        records.append(dict(source=TASKS[source], eigenvalues=values.tolist(), damping=ridge,
                            modes=diagnostics, whole_task_audit=whole,
                            residual_relative_norm=float((residual * norms).norm() / norms.norm())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tasks[0].existing.write(args.output, dict(base_sha256=tasks[0].base_sha256, experts=summaries,
        names=[n for n, _ in tasks[0].found], group_coefficients=spec, records=records,
        interventions=interventions, reference_validation=scores,
        confirmation_validation_indices=[d.indices for d in val], final_validation_indices=[d.indices for d in final_data]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--dtd', type=Path, required=True)
    parser.add_argument('--sun', type=Path, required=True)
    parser.add_argument('--probability', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
