"""Measure DTD parameter, Fisher, and loss-gradient dynamics on saved checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from analyze_dtd_expert_geometry import load_expert
from measure_expert_responses import product_grams, save_arrays
from train_atlas import embedding
from train_eight_expert_queue import BASE, HEADS, ROOT, build_model, existing, read, splits


def load_trace(folder):
    manifest = read(folder / 'trace.json')
    states = []
    for row in manifest['rows']:
        path = folder / row['checkpoint']
        if existing.sha(path) != row['sha256']:
            raise ValueError(f'Checkpoint hash mismatch at step {row["step"]}')
        state = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
        if (state['epoch'], state['step']) != (row['epoch'], row['step']):
            raise ValueError(f'Checkpoint index mismatch at {path.name}')
        states.append(state)
    return manifest, states


def parameter_geometry(states, keys, atoms):
    windows = [state['delta_visual'] for state in states[1:]]
    gram = torch.zeros((len(windows), len(windows)), dtype=torch.double, device='cuda')
    coeff = []
    for key in states[0]['visual']:
        rows = torch.stack([window[key].flatten() for window in windows]).cuda().double()
        gram += rows @ rows.T
        if key in atoms:
            coeff.append((rows @ atoms[key].flatten().double()).cpu())
    # Use the declared order, independently of the ordering in state_dict().
    encountered = [key for key in states[0]['visual'] if key in atoms]
    columns = dict(zip(encountered, coeff))
    return gram.cpu(), torch.stack([columns[key] for key in keys], dim=1)


@torch.no_grad()
def fixed_response(model, images, call_names, params, tangents):
    columns = []
    for name, parameter, tangent in zip(call_names, params, tangents):
        def response(weight):
            logits = torch.func.functional_call(model, {name: weight}, (images,)).float()
            return embedding(logits)

        columns.append(torch.func.jvp(response, (parameter,), (tangent,))[1])
    stacked = torch.stack(columns, dim=1).reshape(len(images), -1, len(columns))
    return stacked.cpu(), product_grams(stacked)


@torch.no_grad()
def linear_response(model, images, all_names, params, delta):
    def predict(*weights):
        logits = torch.func.functional_call(model, dict(zip(all_names, weights)), (images,)).float()
        return embedding(logits)

    response = torch.func.jvp(predict, params, delta)[1]
    return response.reshape(2, -1).mul(2 ** .5).cpu()


def endpoint_gradients(model, params, images, r0, r1):
    logits = model(images).float()
    rows = []
    for i, residual in enumerate((r0, r1)):
        grads = torch.autograd.grad(logits, params, grad_outputs=residual.to(logits) / len(images), retain_graph=i == 0)
        rows.append(tuple(g.detach() for g in grads))
    return rows


def gradient_mechanism(model, visual, start, end, images, label_groups, params):
    groups = []
    for group, x in enumerate(images.chunk(2)):
        labels = F.one_hot(label_groups[group], num_classes=47)
        r0 = start['probability'][group] - labels
        r1 = end['probability'][group] - labels
        visual.load_state_dict(start['visual'])
        g00, g01 = endpoint_gradients(model, params, x, r0, r1)
        visual.load_state_dict(end['visual'])
        g10, g11 = endpoint_gradients(model, params, x, r0, r1)
        g = torch.zeros((4, 4), dtype=torch.double, device='cuda')
        for a, b, c, d in zip(g00, g01, g10, g11):
            a, b, c, d = (x.flatten().double() for x in (a, b, c, d))
            terms = torch.stack((b - a, c - a, d - c - b + a, d - a))
            g += terms @ terms.T
        groups.append(g.cpu())
    return torch.stack(groups)


def intervals(rows):
    # Each 27-step epoch has its own two recorded pieces. Never compare windows
    # from different training regions as though their widths alone differ.
    fine = [(i, i + 1) for i in range(len(rows) - 1)]
    epoch_ends = [i for i, row in enumerate(rows) if row['epoch_end']]
    coarse = list(zip(epoch_ends, epoch_ends[1:]))
    return fine, coarse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing.ROOT = ROOT
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest, states = load_trace(args.trace)
    head = existing.ClassificationHead.load(str(HEADS / 'DTD.pt')).cuda().eval()
    encoder, model, found, base_flat = build_model(existing, head)
    del base_flat
    visual = encoder.model.visual
    model.requires_grad_(False).eval()
    matrix_names, matrix_params = zip(*found)
    base_hash = existing.sha(BASE)
    expert, expert_manifest = load_expert(args.queue, 'DTD', list(matrix_names), base_hash)
    for key, value in visual.state_dict().items():
        if not torch.equal(value.cpu(), states[0]['visual'][key]) or not torch.equal(expert[key], states[-1]['visual'][key]):
            raise ValueError(f'Trajectory endpoints do not match pretrained and expert weights at {key}')
    keys = tuple(name.removeprefix('model.visual.') for name in matrix_names)
    atom_changes = [(expert[k] - states[0]['visual'][k]).cuda() for k in keys]
    norms = torch.stack([x.norm() for x in atom_changes])
    tangents = tuple(delta / norm for delta, norm in zip(atom_changes, norms))
    atoms = dict(zip(keys, tangents))
    del atom_changes
    data, _ = splits('DTD', encoder.val_preprocess, encoder.val_preprocess)
    lookup = {raw: index for index, raw in enumerate(data.indices)}
    batches = [[data[lookup[i]] for i in group] for group in manifest['raw_train_indices']]
    images = torch.stack([x for batch in batches for x, _ in batch]).cuda()
    labels = [torch.tensor([y for _, y in batch]) for batch in batches]
    if [y.tolist() for y in labels] != manifest['labels']:
        raise ValueError('Probe labels changed')
    call_names = tuple('image_encoder.' + name for name in matrix_names)
    full_keys, params = zip(*visual.named_parameters())
    full_names = tuple('image_encoder.model.visual.' + k for k in full_keys)
    metadata = dict(base_sha256=base_hash, expert_sha256=expert_manifest['checkpoint_sha256'],
                    head_sha256=expert_manifest['head_sha256'], trace_sha256=existing.sha(args.trace / 'trace.json'),
                    steps=[s['step'] for s in states], epoch_ends=[s['step'] for s in manifest['rows'] if s['epoch_end']],
                    matrix_names=list(matrix_names), raw_train_indices=manifest['raw_train_indices'],
                    atom_norms=norms.cpu().tolist())
    metadata_path = output / 'manifest.json'
    if metadata_path.exists() and read(metadata_path) != metadata:
        raise ValueError('Existing output provenance differs')
    existing.write(metadata_path, metadata)
    parameter_path = output / 'parameter.npz'
    if not parameter_path.exists():
        param_gram, coefficients = parameter_geometry(states, keys, atoms)
        save_arrays(parameter_path, gram=param_gram.numpy(), coefficients=coefficients.numpy())
    for state in states:
        path = output / f'anchor_{state["step"]:04d}.npz'
        if not path.exists():
            visual.load_state_dict(state['visual'])
            with torch.no_grad():
                actual = model(images).double().softmax(-1).cpu().reshape(2, 32, 47)
            if not torch.allclose(actual, torch.stack(state['probability']), atol=2e-5, rtol=0):
                raise ValueError(f'Probe probabilities failed to replay at step {state["step"]}')
            response, gram = fixed_response(model, images, call_names, matrix_params, tangents)
            save_arrays(path, response=response.numpy(), gram=gram.numpy(), probability=actual.numpy())
        print(f'anchor {state["step"]}: Fisher response ready', flush=True)

    fine, coarse = intervals(manifest['rows'])
    for i, j in [*fine, *coarse]:
        start, end = states[i], states[j]
        path = output / f'interval_{start["step"]:04d}_{end["step"]:04d}.npz'
        if not path.exists():
            visual.load_state_dict(start['visual'])
            delta = tuple((end['visual'][k] - start['visual'][k]).cuda() for k in full_keys)
            derivative = linear_response(model, images, full_names, params, delta)
            del delta
            visual.requires_grad_(True)
            gradient_gram = gradient_mechanism(model, visual, start, end, images, labels, params)
            visual.requires_grad_(False)
            save_arrays(path, derivative=derivative.numpy(), gradient_gram=gradient_gram.numpy())
        print(f'interval {start["step"]}–{end["step"]}: full update JVP and gradient terms ready', flush=True)
    existing.write(output / 'complete.json', dict(fine=fine, coarse=coarse))


if __name__ == '__main__':
    main()
