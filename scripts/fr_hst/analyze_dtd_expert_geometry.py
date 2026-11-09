"""Measure the new DTD expert update on fixed probes and historical validation."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import ImageFile
from torch.nn import functional as F

from build_dtd_fiber_atlas import distance
from discover_fiber_groups import fisher_gram, segments
from train_eight_expert_queue import BASE, HEADS, ROOT, build_model, existing, loader, read, splits


def load_expert(queue, task, names, base_hash):
    folder = queue / task
    manifest = read(folder / 'complete.json')
    path = folder / 'best.pt'
    if existing.sha(path) != manifest['checkpoint_sha256']:
        raise ValueError(f'{task}: checkpoint hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    meta = saved['provenance']
    if (meta['task'], meta['base_sha256'], meta['matrix_names']) != (task, base_hash, names):
        raise ValueError(f'{task}: incompatible expert checkpoint')
    if meta['head_sha256'] != existing.sha(HEADS / f'{task}.pt'):
        raise ValueError(f'{task}: head hash mismatch')
    return saved['visual'], manifest


def flatten(visual, keys):
    return torch.cat([visual[key].reshape(-1) for key in keys])


def spectra(c):
    eig, vec = torch.linalg.eigh(c)
    e, v = eig.flip(0).clamp_min(0), vec.flip(1)
    return e, v


def fraction(c, v):
    return float((v.T @ c @ v).trace() / c.trace())


def cosine(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def analyze_grams(grams, coefficients):
    fit = grams['base']['DTD'][0]
    eigenvalues, vectors = spectra(fit)
    top = vectors[:, :4]
    bottom = vectors[:, -4:]
    result = {}
    for anchor, tasks in grams.items():
        result[anchor] = {}
        for task, groups in tasks.items():
            a, b = groups
            rows = []
            for c in groups:
                e, _ = spectra(c)
                rows.append(dict(trace=float(c.trace()), top4=float(e[:4].sum() / e.sum()),
                                 top8=float(e[:8].sum() / e.sum()), rank_90=int((e.cumsum(0) < .9 * e.sum()).sum()) + 1,
                                 participation_ratio=float(e.sum().square() / e.square().sum()),
                                 full_matrix_update_rms=float((coefficients @ c @ coefficients).sqrt()),
                                 fitted_base_dtd_top4_fraction=fraction(c, top),
                                 fitted_base_dtd_bottom4_fraction=fraction(c, bottom),
                                 block_isolated_energy=float((coefficients.square() @ c.diag()))))
            result[anchor][task] = dict(group_cosine=cosine(a, b), groups=rows)
    for task in result['base']:
        for anchor in ('dtd_expert', 'sun_expert'):
            if task in result[anchor]:
                result[anchor][task]['cosine_with_base_by_group'] = [
                    cosine(a, b) for a, b in zip(grams['base'][task], grams[anchor][task])]
    return result, eigenvalues


@torch.inference_mode()
def logits(model, batches):
    return torch.cat([model(x).float().cpu() for x, _ in batches])


def report(before, after, labels):
    return dict(accuracy=float((after.argmax(1) == labels).double().mean()),
                ce=float(F.cross_entropy(after, labels)),
                fr=float(distance(before.double().softmax(-1).sqrt(), after.double().softmax(-1).sqrt())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # SUN397 decoder audit

    existing.ROOT = ROOT
    base_hash = existing.sha(BASE)
    tasks = ('DTD', 'SUN397')
    heads = {task: existing.ClassificationHead.load(str(HEADS / f'{task}.pt')).cuda().eval() for task in tasks}
    encoder, model, found, flat_base = build_model(existing, heads['DTD'])
    visual = encoder.model.visual
    model.requires_grad_(False).eval()
    for head in heads.values():
        head.requires_grad_(False)
    names = [name for name, _ in found]
    keys = [name.removeprefix('model.visual.') for name in names]
    parts = segments(SimpleNamespace(found=found))
    base = {name: value.detach().clone() for name, value in visual.state_dict().items()}
    dtd, dtd_manifest = load_expert(args.queue, 'DTD', names, base_hash)
    sun, sun_manifest = load_expert(args.queue, 'SUN397', names, base_hash)
    dtd, sun = ({k: v.cuda() for k, v in state.items()} for state in (dtd, sun))
    anchors = {'base': base, 'dtd_expert': dtd, 'sun_expert': sun}
    delta = {name: dtd[name] - value for name, value in base.items()}
    selected = set(keys)
    matrix = flatten(delta, keys)
    coefficients = torch.stack([matrix[s].norm() for _, _, s in parts]).double().cpu()
    if (coefficients <= 0).any():
        raise ValueError('DTD update has empty matrix atoms')
    unit = torch.cat([matrix[s] / c for (_, _, s), c in zip(parts, coefficients.tolist())])
    remainder_norm = sum(float(v.square().sum()) for k, v in delta.items() if k not in selected) ** .5
    probes, probe_ids, batches = {}, {}, {}
    for task in tasks:
        fit, validation = splits(task, encoder.val_preprocess, encoder.val_preprocess)
        ids = torch.randperm(len(fit), generator=torch.Generator().manual_seed(20261109))[:64]
        probe_ids[task] = [[fit.indices[i] for i in g] for g in ids.split(32)]
        probes[task] = [torch.stack([fit[i][0] for i in g]).cuda() for g in ids.split(32)]
        # Keep validation batches on the host and use the same image order for every weight state.
        batches[task] = [(x, y) for x, y in loader(validation, 8, 128, 42)]

    geometry = {}
    for anchor, state in anchors.items():
        visual.load_state_dict(state)
        flat = flat_base if anchor == 'base' else flatten(state, keys)
        geometry[anchor] = {}
        for task in tasks:
            model.classification_head = heads[task]
            geometry[anchor][task] = [fisher_gram(SimpleNamespace(model=model), flat, unit, parts, images)
                                      for images in probes[task]]
            print(f'{anchor} / {task}: two Fisher matrices measured', flush=True)
    summaries, eigenvalues = analyze_grams(geometry, coefficients)

    def predictions():
        result = {}
        for task in tasks:
            model.classification_head = heads[task]
            result[task] = logits(model, ((x.cuda(non_blocking=True), y) for x, y in batches[task]))
        return result

    labels = {task: torch.cat([y for _, y in batches[task]]) for task in tasks}
    finite = {}
    for anchor in ('base', 'sun_expert'):
        state = anchors[anchor]
        visual.load_state_dict(state)
        baseline = predictions()
        rows = [dict(direction='full', alpha=0., tasks={k: report(z, z, labels[k]) for k, z in baseline.items()})]
        for mode, amounts in [('full', (.25, .5, 1.)), ('matrix48', (.25, 1.)), ('remainder', (.25, 1.))]:
            for amount in amounts:
                if anchor == 'base' and mode == 'full' and amount == 1.:
                    visual.load_state_dict(dtd)
                else:
                    visual.load_state_dict({k: w + amount * delta[k]
                                            if mode == 'full' or (k in selected) == (mode == 'matrix48') else w
                                            for k, w in state.items()})
                pred = predictions()
                rows.append(dict(direction=mode, alpha=amount,
                                 tasks={k: report(baseline[k], z, labels[k]) for k, z in pred.items()}))
        finite[anchor] = rows
        print(f'{anchor}: validation interventions measured', flush=True)
    for task, row, manifest in [('DTD', finite['base'][3], dtd_manifest),
                                ('SUN397', finite['sun_expert'][0], sun_manifest)]:
        if abs(row['tasks'][task]['accuracy'] - manifest['best']['accuracy']) > 1e-6:
            raise ValueError(f'{task} endpoint accuracy failed to reproduce')
    existing.write(args.output, dict(base_sha256=base_hash, checkpoints={'DTD': dtd_manifest, 'SUN397': sun_manifest},
                                     matrix_names=names, matrix_norms=coefficients.tolist(),
                                     matrix_norm=float(coefficients.norm()), remainder_norm=remainder_norm,
                                     probe_raw_train_indices=probe_ids, geometry=summaries,
                                     grams={a: {k: [m.tolist() for m in ms] for k, ms in ts.items()}
                                            for a, ts in geometry.items()},
                                     base_dtd_fit_eigenvalues=eigenvalues.tolist(), historical_validation=finite))


if __name__ == '__main__':
    main()
