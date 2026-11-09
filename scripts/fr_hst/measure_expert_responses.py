"""Shared eight-task Fisher response bank and expert-point checks."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFile

from analyze_dtd_expert_geometry import load_expert
from train_atlas import embedding
from train_eight_expert_queue import BASE, HEADS, ROOT, TASKS, build_model, existing, splits


def save_arrays(path, **arrays):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def probes(encoder, task):
    ImageFile.LOAD_TRUNCATED_IMAGES = task == 'SUN397'
    fit, _ = splits(task, encoder.val_preprocess, encoder.val_preprocess)
    indices = torch.randperm(len(fit), generator=torch.Generator().manual_seed(20261109))[:64].tolist()
    raw = [fit.indices[i] if hasattr(fit, 'indices') else i for i in indices]
    paths = [str(fit.samples[i][0]) for i in indices] if task == 'EuroSAT' else None
    return torch.stack([fit[i][0] for i in indices]).cuda(), dict(raw_indices=raw, paths=paths)


def product_grams(outputs):
    # One 64-image embedding was computed. Undo its factor of 1/sqrt(64)
    # relative to the per-group factor of 1/sqrt(32).
    a, b = outputs.chunk(2, dim=0)
    return torch.stack([2 * (g.flatten(0, 1).T @ g.flatten(0, 1)) for g in (a, b)]).cpu()


@torch.no_grad()
def all_atoms(model, images, names, params, atoms):
    columns = []
    for name, weight, block in zip(names, params, atoms):
        def forward(w):
            return embedding(torch.func.functional_call(model, {name: w}, (images,)).float())

        def tangent(v):
            return torch.func.jvp(forward, (weight,), (v,))[1]

        result = torch.vmap(tangent, chunk_size=4)(block)
        columns.append(result.T)
    # Columns use task-major order: each expert contributes its 48 consecutive atoms.
    return product_grams(torch.stack(columns, dim=2).reshape(len(images), -1, len(TASKS) * len(names)))


@torch.no_grad()
def candidate_grams(model, images, names, params, tangents):
    def forward(*weights):
        overrides = dict(zip(names, weights))
        return embedding(torch.func.functional_call(model, overrides, (images,)).float())

    result = torch.stack([torch.func.jvp(forward, params, update)[1] for update in tangents], dim=1)
    return product_grams(result.reshape(len(images), -1, len(tangents)))


def whiten(metric):
    eigenvalues, vectors = torch.linalg.eigh(metric)
    keep = eigenvalues > eigenvalues.max() * 1e-8
    return vectors[:, keep] * eigenvalues[keep].rsqrt()


def candidates(grams, metric, norm, task_names):
    w = whiten(metric)
    transformed = w.T @ grams[:, 0] @ w
    total = transformed.sum(0)
    ridge = .01 * total.diag().median()
    e, q = torch.linalg.eigh(total + ridge * torch.eye(len(total), dtype=torch.double))
    transform = q * e.rsqrt()
    vectors, metadata = [], []
    for index, task in enumerate(task_names):
        full = torch.zeros_like(norm)
        full[index] = norm[index]
        vectors.append(full.flatten())
        metadata.append(dict(task=task, kind='whole_matrix_update'))
        scores, modes = torch.linalg.eigh(transform.T @ transformed[index] @ transform)
        for j in (-1, -2):
            v = w @ transform @ modes[:, j]
            v /= (v @ metric @ v).sqrt()
            vectors.append(v)
            metadata.append(dict(task=task, kind=f'fisher_mode_{-j}', fitted_share_regularized=float(scores[j])))
    return torch.stack(vectors), metadata, float(ridge)


def energies(matrix, vectors):
    return torch.einsum('ci,...ij,cj->...c', vectors, matrix, vectors).clamp_min(0)


def profiles(value):
    # The axis before image group and direction indexes observing tasks.
    return value / value.sum(0, keepdim=True).clamp_min(1e-30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    task_names = list(TASKS)
    base_hash = existing.sha(BASE)
    heads = {k: existing.ClassificationHead.load(str(HEADS / f'{k}.pt')).cuda().eval() for k in task_names}
    encoder, model, found, base_flat = build_model(existing, heads['DTD'])
    del base_flat
    model.requires_grad_(False).eval()
    for head in heads.values():
        head.requires_grad_(False)
    visual = encoder.model.visual
    names, params = zip(*found)
    state_keys = [name.removeprefix('model.visual.') for name in names]
    call_names = tuple('image_encoder.' + name for name in names)
    block_count = len(names)
    metric = torch.zeros((len(task_names) * block_count,) * 2, dtype=torch.double)
    checkpoint_states, checkpoints = {}, {}
    for task in task_names:
        checkpoint_states[task], checkpoints[task] = load_expert(args.queue, task, list(names), base_hash)
    atoms, norm = [], []
    for block, (key, p) in enumerate(zip(state_keys, params)):
        changes = torch.stack([checkpoint_states[k][key].cuda() - p for k in task_names])
        lengths = changes.flatten(1).norm(dim=1)
        if (lengths <= 0).any():
            raise ValueError(f'Empty atom in {key}')
        unit = changes / lengths.reshape(-1, *([1] * p.ndim))
        atoms.append(unit)
        norm.append(lengths.cpu().double())
        index = torch.arange(len(task_names)) * block_count + block
        rows = unit.flatten(1).double()
        metric[index[:, None], index] = (rows @ rows.T).cpu()
    norms = torch.stack(norm, dim=1)
    images, image_ids = {}, {}
    for task in task_names:
        images[task], image_ids[task] = probes(encoder, task)
    provenance = dict(task_order=task_names, matrix_order=names, base_sha256=base_hash,
                      checkpoints={k: m['checkpoint_sha256'] for k, m in checkpoints.items()}, image_ids=image_ids,
                      shape=[len(task_names), 2, metric.shape[0], metric.shape[1]], basis_order='task_then_matrix',
                      image_groups='fit_first_32_audit_last_32_train_only')
    manifest = out / 'manifest.json'
    if manifest.exists() and existing.json.loads(existing.json.dumps(provenance)) != existing.json.loads(manifest.read_text()):
        raise ValueError('Response bank provenance differs from existing files')
    existing.write(manifest, provenance)
    banks = []
    for task in task_names:
        path = out / f'base_{task}.npz'
        if path.exists():
            with np.load(path) as result:
                bank = torch.from_numpy(result['gram'])
        else:
            model.classification_head = heads[task]
            bank = all_atoms(model, images[task], call_names, params, atoms)
            if not bank.isfinite().all():
                raise ValueError(f'Nonfinite base response for {task}')
            save_arrays(path, gram=bank.numpy())
        banks.append(bank)
        print(f'base {task}: both {metric.shape[0]}x{metric.shape[1]} matrices ready', flush=True)
    grams = torch.stack(banks)
    coeffs, rows, ridge = candidates(grams, metric, norms, task_names)
    base_energy = energies(grams, coeffs)
    base_share = profiles(base_energy)
    save_arrays(out / 'eight_task_base.npz', gram=grams.numpy(), atom_metric=metric.numpy(),
                atom_norms=norms.numpy(), candidate_coefficients=coeffs.numpy())
    chosen = {k: [] for k in task_names}
    for i, row in enumerate(rows):
        owner = row['task']
        # Rank references by fitted response only. Keep the second group for audit.
        other = max((j for j, k in enumerate(task_names) if k != owner), key=lambda j: float(base_energy[j, 0, i]))
        refs = (owner, task_names[other])
        for ref in refs:
            chosen[ref].append(i)
        row.update(references=refs, base_energy=base_energy[:, :, i].tolist(), base_share=base_share[:, :, i].tolist())

    for ref, indices in chosen.items():
        path = out / f'at_{ref}.npz'
        if path.exists():
            with np.load(path) as result:
                checks = torch.from_numpy(result['gram'])
                if result['candidate_indices'].tolist() != indices:
                    raise ValueError(f'Candidate ordering changed at {ref}')
        else:
            visual.load_state_dict(checkpoint_states[ref])
            tangents = []
            for i in indices:
                coeff = coeffs[i].reshape(len(task_names), block_count).float().cuda()
                tangents.append(tuple((coeff[:, b] @ atom.flatten(1)).reshape(atom.shape[1:])
                                      for b, atom in enumerate(atoms)))
            checks = []
            for observer in task_names:
                model.classification_head = heads[observer]
                checks.append(candidate_grams(model, images[observer], call_names, params, tangents))
            checks = torch.stack(checks)
            if not checks.isfinite().all():
                raise ValueError(f'Nonfinite candidate response at {ref}')
            save_arrays(path, gram=checks.numpy(), candidate_indices=np.array(indices))
            del tangents
        energy = checks.diagonal(dim1=-2, dim2=-1).clamp_min(0)
        shares = profiles(energy)
        for i, candidate in enumerate(indices):
            rows[candidate].setdefault('expert_checks', {})[ref] = dict(
                energy=energy[:, :, i].tolist(), share=shares[:, :, i].tolist())
        print(f'{ref} expert: checked {len(indices)} directions across all observing tasks', flush=True)
    existing.write(out / 'summary.json', dict(provenance=provenance, eligibility=checkpoints,
                   candidate_ridge=ridge, candidate_order=rows, basis_rank=whiten(metric).shape[1]))


if __name__ == '__main__':
    main()
