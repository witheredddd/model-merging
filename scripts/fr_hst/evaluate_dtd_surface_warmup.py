"""Compare the exact warmup anchor and a selected DTD surface checkpoint."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from dtd_surface_learning import canonical_total
from run_dtd_carrier_diagnostics import Operator
from train_eight_expert_queue import BASE, HEADS, ROOT, build_model, existing, loader, splits


@torch.no_grad()
def evaluate(checkpoint):
    state = torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
    provenance = state['provenance']
    best = state['best']
    surface = state['surface']
    if provenance['task']!='DTD' or provenance['algorithm']!='persistent_ce_surfaces_v1':
        raise ValueError('Expected a DTD surface checkpoint')
    head_path = HEADS/'DTD.pt'
    if existing.sha(BASE)!=provenance['base_sha256'] or existing.sha(head_path)!=provenance['head_sha256']:
        raise ValueError('Checkpoint initialization provenance mismatch')
    existing.ROOT = ROOT
    head = existing.ClassificationHead.load(str(head_path)).cuda()
    encoder,model,_,flat = build_model(existing,head)
    del flat
    visual = encoder.model.visual
    op = Operator(model,visual)
    _,validation = splits('DTD',val_transform=encoder.val_preprocess)
    if len(validation)!=provenance['validation_n']:
        raise ValueError('Unexpected holdout size')
    batches = loader(validation,2,32,provenance['seed'])
    anchor = surface['warmup_anchor']
    for p,w in zip(op.params,op.parts(anchor)):
        p.copy_(w)
    _,warmup,y = existing.evaluate(model,batches,retain=True)
    total = canonical_total(surface).float()
    if any(not torch.equal(a,state['visual'][key]) for a,key in zip(op.parts(total),op.keys)):
        raise ValueError('Saved group updates do not reconstruct the selected checkpoint')
    visual.load_state_dict(state['visual'])
    _,selected,selected_y = existing.evaluate(model,batches,retain=True)
    if not torch.equal(y,selected_y):
        raise ValueError('Paired validation order mismatch')
    before,after = warmup.argmax(-1),selected.argmax(-1)
    hit_before,hit_after = before==y,after==y
    ce_before,ce_after = [F.cross_entropy(z.double(),y,reduction='none') for z in (warmup,selected)]
    if float(hit_after.double().mean())!=best['validation']['accuracy'] or abs(float(ce_after.mean())-best['validation']['ce'])>1e-4:
        raise ValueError('Evaluation does not match the selected checkpoint validation')
    return dict(provenance=provenance,selection='lowest_validation_ce_at_snapshot',
                warmup=dict(step=provenance['warmup'],correct=int(hit_before.sum()),ce=float(ce_before.mean())),
                selected=dict(epoch=best['epoch'],step=best['step'],correct=int(hit_after.sum()),ce=float(ce_after.mean())),
                n=len(y),newly_correct=int((~hit_before & hit_after).sum()),
                newly_incorrect=int((hit_before & ~hit_after).sum()),
                rows=[dict(id=int(idx),label=int(label),warmup_prediction=int(p),selected_prediction=int(q),
                           warmup_ce=float(l),selected_ce=float(m))
                      for idx,label,p,q,l,m in zip(validation.indices,y,before,after,ce_before,ce_after)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    existing.write(args.output,evaluate(args.checkpoint))


if __name__=='__main__':
    main()
