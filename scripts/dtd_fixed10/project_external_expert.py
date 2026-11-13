"""Frozen-space transfer projection of the archived high-performing DTD expert.

Inference only: no optimizer, gradient, coordinate fitting, or basis update.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        raise FileExistsError('Completed experiment is immutable; choose a new output')
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.30)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path('/home/shangxueyuan/merging/tsv')
    sys.path.insert(0, str(root/'scripts/fisher_rao_online_v5_98f36a1'))
    from clip_runtime import ClipTask
    raw = ClipTask(root, out, 128, 32, task='DTD')
    from task_data import test_set
    reference = root/'results/EXPERTS8_FD_REVIEWED_20261110/DTD/adaptive_dynamics'
    manifest = json.loads((reference/'complete.json').read_text())
    historical = json.loads((reference/'historical_test.json').read_text())
    checkpoint = reference/'epoch08.pt'
    basis_path = Path('/media/shangxueyuan/data1/DTD_fixed10_20261113/fixed_basis.pt')
    write(out/'status.json', dict(stage='checking_inputs', training=False))
    digest = sha(checkpoint)
    assert digest == manifest['checkpoint_sha256'] == historical['checkpoint_sha256']
    basis_digest = sha(basis_path)
    provenance = manifest['provenance']
    assert provenance['base_sha256'] == raw.rt.input['base_sha256']
    assert provenance['head_sha256'] == raw.rt.input['head_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    assert saved['provenance'] == provenance
    visual = saved['visual']
    basis = torch.load(basis_path, map_location='cpu', weights_only=True, mmap=True)
    assert basis['identity']['base'] == provenance['base_sha256']
    assert basis['identity']['head'] == provenance['head_sha256']
    B = basis['B_rows']
    assert tuple(B.shape) == (10, raw.theta0.numel())
    gram_error = float((B@B.T-torch.eye(10, dtype=torch.double)).norm())
    assert gram_error < 1e-8
    current_visual = raw.rt.encoder.model.visual.state_dict()
    assert set(visual) == set(current_visual)
    assert all(visual[k].shape == current_visual[k].shape for k in visual)
    active_names = [n.removeprefix('model.visual.') for n, _ in raw.rt.found]
    assert len(active_names) == 48 and all(n in visual for n in active_names)
    expert_flat = torch.cat([visual[n].flatten() for n in active_names])
    base = raw.theta0.cpu().double()
    tau = expert_flat.double()-base
    z = B@tau
    projected = B.T@z
    residual = tau-projected
    assert float((B@residual).norm()) < 1e-8
    outside = []
    for name, tensor in visual.items():
        if name not in active_names:
            delta = tensor.double()-current_visual[name].cpu().double()
            norm = float(delta.norm())
            if norm:
                outside.append(dict(name=name, numel=tensor.numel(), update_norm=norm))
    active_sq = float(tau.square().sum())
    full_sq = active_sq+sum(row['update_norm']**2 for row in outside)
    theta_recon = (base+projected).float().cuda()
    active_expert = expert_flat.cuda()
    expert_map = {'image_encoder.model.visual.'+n: tensor.cuda() for n, tensor in visual.items()}
    assert set(expert_map).issubset(raw.rt.model.state_dict())
    dataset = test_set('DTD', raw.rt.encoder.val_preprocess)
    assert len(dataset) == 1880
    assert dataset.class_to_idx == raw.rt.raw.class_to_idx
    train_ids = {Path(p).relative_to(root/'datasets/dtd/train').as_posix() for p,_ in raw.rt.raw.samples}
    test_ids = [Path(p).relative_to(root/'datasets/dtd/val').as_posix() for p,_ in dataset.samples]
    assert not train_ids & set(test_ids)
    identity = dict(source_sha256=sha(__file__), expert_checkpoint_sha256=digest,
                    fixed_basis_sha256=basis_digest, base_sha256=provenance['base_sha256'],
                    head_sha256=provenance['head_sha256'], prior_basis_rule=basis['identity']['eigen_rule'],
                    expert_path=str(checkpoint), fixed_basis_path=str(basis_path),
                    expert_steps=216, basis_source='Old v5 128-step trajectory; unchanged from fixed10 experiment',
                    test_split='dtd/val', test_n=1880,
                    official_fold_assignment_independently_verified=False,
                    training=False, coordinate_optimization=False, basis_optimization=False,
                    expert_used_to_construct_basis=False,
                    active_scope=active_names, active_D=raw.theta0.numel(),
                    sample_ids=test_ids, class_to_idx=dataset.class_to_idx)
    write(out/'config.json', identity)
    local_state = dict(active_parameter=theta_recon.cpu(), coordinates=z, identity=identity)
    torch.save(local_state, out/'reconstructed_state.pt')
    del local_state
    storage = {arm: dict(ce=[], correct=[], labels_predicted=[], kl_from_full=[],
                         absolute_ce_delta_from_full=[]) for arm in ('base','full_expert','active48_only','fisher10')}
    iterator = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True, shuffle=False)
    for x, y in iterator:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        logits = dict(base=raw.rt.predict(raw.theta0, x).double(),
                      full_expert=torch.func.functional_call(raw.rt.model, expert_map, (x,)).double(),
                      active48_only=raw.rt.predict(active_expert, x).double(),
                      fisher10=raw.rt.predict(theta_recon, x).double())
        old = logits['full_expert']
        old_ce = F.cross_entropy(old, y, reduction='none')
        for arm, value in logits.items():
            ce = F.cross_entropy(value, y, reduction='none')
            row = storage[arm]
            row['ce'].extend(ce.cpu().tolist())
            row['correct'].extend((value.argmax(1)==y).cpu().tolist())
            row['labels_predicted'].extend(value.argmax(1).cpu().tolist())
            row['kl_from_full'].extend((old.softmax(1)*(old.log_softmax(1)-value.log_softmax(1))).sum(1).cpu().tolist())
            row['absolute_ce_delta_from_full'].extend((ce-old_ce).abs().cpu().tolist())
        write(out/'status.json', dict(stage='paired_evaluation', samples=len(storage['base']['ce']),
                                     target=len(dataset), training=False))
    scores = {}
    for arm, row in storage.items():
        scores[arm] = dict(n=len(dataset), correct=sum(row['correct']),
                          accuracy=sum(row['correct'])/len(dataset), CE=sum(row['ce'])/len(dataset),
                          prediction_agreement_with_full=sum(a==b for a,b in zip(
                              row['labels_predicted'],storage['full_expert']['labels_predicted']))/len(dataset),
                          mean_KL_full_to_model=sum(row['kl_from_full'])/len(dataset),
                          maximum_absolute_sample_CE_difference_from_full=max(row['absolute_ce_delta_from_full']))
    report = dict(identity=identity, historical_reference=historical, scores=scores,
                  geometry=dict(basis_orthogonality_error=gram_error, active_update_norm=active_sq**.5,
                                full_visual_update_norm=full_sq**.5,
                                projected_update_norm=float(projected.norm()),
                                residual_update_norm=float(residual.norm()),
                                retained_active_squared_norm=float(projected.square().sum())/active_sq,
                                retained_full_visual_squared_norm=float(projected.square().sum())/full_sq,
                                residual_coordinate_norm=float((B@residual).norm()),
                                changed_outside_active48=outside),
                  caveats=['Historical test mirror dtd/val; no independently verified official fold.',
                           'No training or selection on these test measurements.',
                           'Development sets from the old protocol participated in original expert training.',
                           'Active48-only is a diagnostic control for changes outside the basis scope.'])
    write(out/'report.json', report)
    write(out/'status.json', dict(stage='complete', training=False, n=1880))
    print(json.dumps(dict(scores=scores, geometry=report['geometry'])), flush=True)


if __name__ == '__main__':
    main()
