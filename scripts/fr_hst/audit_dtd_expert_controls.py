"""Measure finite coordinate controls about a trained DTD expert, without training."""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import ImageFolder

from run_dtd_carrier_diagnostics import Operator, metrics
from train_dtd_functional_dynamics import tangent, tensor
from train_dtd_online_carrier import decoder, response
from train_dtd_rolling_innovation import probe_mode
from train_eight_expert_queue import BATCH_SIZE, HEADS, ROOT, build_model, existing, loader, read


RADII = (.005, .02)
RANK_THRESHOLD = .01


def differences(z, q, adjustment):
    return np.concatenate([decoder(v+dv, a)[0]-decoder(v, a)[0] for v,a,dv in zip(z,q,adjustment)])


def dose(direction, z, q, scale, parameter_gram, target):
    direction = direction.reshape(z.shape)
    local = tangent(z,q) @ direction.flatten()
    high = target / max(scale * np.sqrt(local @ parameter_gram @ local), 1e-12)
    low = 0.

    def size(t):
        w = differences(z,q,t*direction)
        return scale*np.sqrt(max(0.,w@parameter_gram@w))

    for _ in range(15):
        if size(high)>=target:
            break
        high *= 2
    else:
        raise ValueError('Could not bound requested parameter dose')
    for _ in range(36):
        middle=(low+high)/2
        if size(middle)<target:
            low=middle
        else:
            high=middle
    return high*direction


def score(old, new, labels):
    result=metrics(old,new,labels)
    a,b=(z.double().softmax(-1).sqrt() for z in (old,new))
    chord=(a-b).norm(dim=-1).div(2).clamp(0,1)
    result.update(ce=float(F.cross_entropy(new.double(),labels)),
                  accuracy=float((new.argmax(-1)==labels).double().mean()),
                  fisher_rao_rms=float(((4*chord.asin()).square().mean()).sqrt()))
    return result


@torch.no_grad()
def logits(op, batches, update=None):
    if update is None:
        return torch.cat([op.evaluate(x) for x,_ in batches])
    override=dict(zip(op.names,(p+d for p,d in zip(op.params,op.parts(update)))))
    return torch.cat([torch.func.functional_call(op.model,override,(x,)).float() for x,_ in batches])


def information(op, axes, z, q, scale, probes):
    d=tangent(z,q)
    u=scale*(tensor(d.T,axes)@axes)
    info=[]
    with probe_mode(op.model):
        for x,y in probes:
            row=np.stack([response(op,x,v) for v in u])
            g=row@row.T
            b=(u@op.gradient(x,y)).double().cpu().numpy()
            eig=np.linalg.eigvalsh(g)
            cross=g.copy()
            for k in range(len(z)):
                cross[2*k:2*k+2,2*k:2*k+2]=0
            info.append(dict(fisher=g.tolist(),fisher_spectrum=eig.tolist(),
                             effective_rank=int((eig>RANK_THRESHOLD*eig.max()).sum()),
                             gradient=b.tolist(),cross_block_fraction=float(np.linalg.norm(cross)/np.linalg.norm(g))))
    return info


def directions(fit, components):
    gram,b=np.asarray(fit['fisher']),np.asarray(fit['gradient'])
    ridge=max(1e-10,.001*np.trace(gram)/len(gram))
    joint=-np.linalg.solve(gram+ridge*np.eye(len(gram)),b)
    axes=np.eye(len(gram))
    for j,row in enumerate(axes):
        for sign in (-1,1):
            yield f'axis_{j+1}_{sign:+d}',sign*row,True
    yield 'joint_descent',joint,True
    yield 'joint_ascent',-joint,True
    local=[]
    for i in range(components):
        sl=slice(2*i,2*i+2)
        v=np.zeros(len(gram))
        v[sl]=-np.linalg.solve(gram[sl,sl]+ridge*np.eye(2),b[sl])
        v/=max(np.linalg.norm(v),1e-20)
        local.append(v)
        yield f'component_{i+1}_descent',v,True
    for i,j in combinations(range(components),2):
        yield f'pair_{i+1}_{j+1}',local[i]+local[j],False
    yield 'all_components_sum',np.sum(local,axis=0),True


def evaluate_expert(folder, op, probes, test, pretrained):
    manifest=read(folder/'complete.json')
    path=folder/'epoch08.pt'
    assert existing.sha(path)==manifest['checkpoint_sha256']
    checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    op.load(checkpoint)
    axes=checkpoint['frames'].flatten(0,1).cuda()
    z,q=np.array(checkpoint['z']),np.array(checkpoint['q'])
    scale=checkpoint['scale']
    original_norm=float(op.delta(pretrained,checkpoint).norm())
    parameter_gram=(axes@axes.T).double().cpu().numpy()
    info=information(op,axes,z,q,scale,probes)
    before=[op.evaluate(x) for x,_ in probes]
    baseline=logits(op,test)
    labels=torch.cat([y for _,y in test])
    zero=tensor(differences(z,q,np.zeros_like(z)),axes)@axes
    assert not bool(torch.count_nonzero(zero))
    # Check the full zero-dose logits, including the functional-call path.
    torch.testing.assert_close(logits(op,test,zero),baseline,atol=1e-6,rtol=0)
    historical=read(folder/'historical_test.json')
    base_score=score(baseline,baseline,labels)
    assert base_score['accuracy']==historical['accuracy']
    perturbations=[]
    for name,direction,use_test in directions(info[0],len(z)):
        if np.linalg.norm(direction)<1e-12:
            continue
        for fraction in RADII:
            norm=original_norm*fraction
            a=dose(direction,z,q,scale,parameter_gram,norm)
            delta=scale*(tensor(differences(z,q,a),axes)@axes)
            torch.testing.assert_close(delta.norm(),delta.new_tensor(norm),rtol=2e-4,atol=1e-9)
            row=dict(name=name,relative_radius=fraction,coordinate_delta=a.tolist(),parameter_norm=float(delta.norm()))
            row['probes']=[dict(**score(b,op.evaluate(x,delta),y),
                                predicted_fisher_energy=float(a.ravel()@np.array(s['fisher'])@a.ravel()))
                           for b,(x,y),s in zip(before,probes,info)]
            if use_test and fraction==RADII[-1]:
                row['test']=score(baseline,logits(op,test,delta),labels)
            perturbations.append(row)
        print(f'{folder.parent.name}: {name} evaluated',flush=True)
    g0,g1=[np.array(g['fisher']) for g in info]
    return dict(checkpoint_sha256=manifest['checkpoint_sha256'],provenance=checkpoint['provenance'],
                z=z.tolist(),q=q.tolist(),scale=scale,components=len(z),dimensions=z.size,
                zero_exact=True,baseline_test=base_score,full_expert_displacement_norm=original_norm,
                probe_geometry=info,cross_probe_gram_cosine=float((g0*g1).sum()/(np.linalg.norm(g0)*np.linalg.norm(g1))),
                perturbations=perturbations)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--early',type=Path,required=True)
    p.add_argument('--pretrained',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    existing.ROOT=ROOT
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    encoder,model,_,flat=build_model(existing,existing.ClassificationHead.load(str(HEADS/'DTD.pt')).cuda())
    del flat
    model.requires_grad_(False).eval()
    model.image_encoder.model.visual.requires_grad_(True)
    op=Operator(model,encoder.model.visual)
    train=ImageFolder(ROOT/'datasets/dtd/train',transform=encoder.val_preprocess)
    test=ImageFolder(ROOT/'datasets/dtd/val',transform=encoder.val_preprocess)
    assert len(train)==3760 and len(test)==1880 and train.class_to_idx==test.class_to_idx
    ids=torch.randperm(len(train),generator=torch.Generator().manual_seed(35036))[:64].tolist()
    probes=[]
    for group in (ids[:32],ids[32:]):
        samples=[train[i] for i in group]
        probes.append((torch.stack([x for x,_ in samples]).cuda(),torch.tensor([y for _,y in samples],device='cuda')))
    batches=[(x.cuda(),y.cuda()) for x,y in loader(test,8,BATCH_SIZE,42)]
    pretrained=torch.load(args.pretrained,map_location='cpu',mmap=True,weights_only=True)
    args.output.mkdir(parents=True,exist_ok=True)
    result={}
    for name,folder in [('original',args.original),('early_penalty',args.early)]:
        path=args.output/f'{name}.json'
        if not path.exists():
            existing.write(path,evaluate_expert(folder,op,probes,batches,pretrained))
        result[name]=read(path)
    existing.write(args.output/'controls.json',dict(probe_indices=[ids[:32],ids[32:]],radii=RADII,arms=result))


if __name__=='__main__':
    main()
