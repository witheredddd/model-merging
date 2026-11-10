"""Fixed-expert interventions for the DTD learned-group audit."""

from itertools import combinations, product

import numpy as np
import torch

from diagnose_dtd_fiber_paths import features, whitening
from diagnose_dtd_learned_groups import (Bank, Data, SEED, group_rows, linear, logits, loss,
                                         metrics, nonlinear, optimize)
from diagnose_dtd_residual_fiber import interval
from run_dtd_carrier_diagnostics import Operator
from train_dtd_rolling_innovation import probe_mode
from train_eight_expert_queue import HEADS, build_model, existing
import train_fd_task as trainer


def response_gram(j):
    gram = j.T @ j
    norm = np.sqrt(np.diag(gram))
    return dict(gram=gram.tolist(),rho=(gram/np.maximum(np.outer(norm,norm),1e-30)).tolist())


def matched_coordinate(bank, chart, i, sign, radius):
    direction = np.eye(2)[i]*sign
    lo,hi = 0.,1.
    while bank.norm(chart.decode(hi*direction))<radius:
        hi *= 2
    for _ in range(35):
        mid = (hi+lo)/2
        if bank.norm(chart.decode(mid*direction))<radius:
            lo = mid
        else:
            hi = mid
    return chart.decode(lo*direction), (lo*direction).tolist()


def perturbations(op, groups, baseline, bank, z, q, scale, displacements, j, radius):
    rows,single = [],{}
    for m,d in enumerate(displacements):
        chart = nonlinear(bank,z,q,scale,[m])
        controls = [(f'scale_{a:g}',a*d,None) for a in (-.2,-.1,0.,.1,.2)]
        for sign in (-1,1):
            controls.append((f'scale_matched_{sign:+}',sign*radius*d/bank.norm(d),None))
            for axis in range(2):
                delta,coordinate = matched_coordinate(bank,chart,axis,sign,radius)
                controls.append((f'coordinate_{axis+1}_{sign:+}',delta,coordinate))
        for name,coef,coordinate in controls:
            delta = bank.vector(coef)
            predictions = [logits(op,x,delta) for x,_ in groups]
            if name in ('scale_-0.1','scale_0.1'):
                single[m,float(name.removeprefix('scale_'))] = loss(predictions[1],groups[1][1])
            rows.append(dict(group=m+1,control=name,coordinate=coordinate,norm=float(delta.norm()),
                             scores=[dict(**metrics(old,new,y,SEED),predicted_fisher=float(np.linalg.norm(a@coef)**2))
                                     for old,new,(_,y),a in zip(baseline,predictions,groups,j)]))
        print(f'Endpoint controls for group {m+1} saved in memory',flush=True)
    return rows,single


def interactions(op, group, baseline, bank, displacements, single):
    x,y = group
    start = loss(baseline,y)
    rows = []
    for m,n in combinations(range(5),2):
        for a,b in product((-.1,.1),repeat=2):
            coefficient = a*displacements[m]+b*displacements[n]
            score = loss(logits(op,x,bank.vector(coefficient)),y)
            paired = score-single[m,a]-single[n,b]+start
            rows.append(dict(groups=[m+1,n+1],alpha=a,beta=b,norm=bank.norm(coefficient),
                             ce_interaction=interval(paired,SEED),split_means=[float(v.mean()) for v in np.array_split(paired,2)],
                             per_image=paired.tolist()))
    return rows


def compensations(op, groups, baseline, bank, displacements, z,q,scale):
    rows = []
    for m,d in enumerate(displacements):
        chart = nonlinear(bank,z,q,scale,[i for i in range(5) if i!=m])
        for fraction in (.1,1.):
            origin = -fraction*d
            correction,fit = optimize(op,groups[0],chart,bank.norm(origin),baseline[0],40,origin=origin)
            x,y = groups[1]
            removed = logits(op,x,bank.vector(origin))
            restored = logits(op,x,bank.vector(origin+correction))
            rows.append(dict(group=m+1,removed_fraction=fraction,removal_norm=bank.norm(origin),fit=fit,
                             removed=metrics(baseline[1],removed,y,SEED),restored=metrics(baseline[1],restored,y,SEED),
                             ce_recovery=interval(loss(removed,y)-loss(restored,y),SEED)))
        print(f'Endpoint compensation for group {m+1} saved in memory',flush=True)
    return rows


def regroup(op, groups, baseline, bank, tangent, j, radius):
    rng = np.random.default_rng(SEED)
    P = bank.dictionary.gram
    white,_ = whitening(tangent.T@P@tangent)
    span = tangent@white
    if span.shape[1]!=10:
        raise ValueError('The endpoint tangent failed the ten-dimensional regrouping check')
    plans = [('original_linear',tangent)]
    for i in range(3):
        while True:
            order = rng.permutation(10)
            if all(order[k]//2 != order[k+1]//2 for k in range(0,10,2)):
                break
        plans.append((f'repaired_{i+1}',tangent[:,order]))
    for i in range(3):
        rotation,_ = np.linalg.qr(rng.normal(size=(10,10)))
        plans.append((f'rotated_{i+1}',span@rotation))
    output = []
    for name,matrix in plans:
        rows,geometries = [],[]
        charts = [linear(bank,matrix[:,2*i:2*i+2]) for i in range(5)]
        unit = np.concatenate([c.tangent@c.white for c in charts],axis=1)
        for response in j:
            function = response@unit
            gram = function.T@function
            own = [np.linalg.eigvalsh(gram[2*i:2*i+2,2*i:2*i+2]).tolist() for i in range(5)]
            cross = gram.copy()
            for i in range(5):
                cross[2*i:2*i+2,2*i:2*i+2] = 0
            geometries.append(dict(gram=gram.tolist(),within_spectra=own,
                                   cross_fraction=float(np.linalg.norm(cross)/np.linalg.norm(gram))))
        for i,chart in enumerate(charts):
            coefficient,fit = optimize(op,groups[0],chart,radius,baseline[0],14)
            x,y = groups[1]
            new = logits(op,x,bank.vector(coefficient))
            rows.append(dict(group=i+1,fit=fit,validation=metrics(baseline[1],new,y,SEED),
                             validation_ce_per_image=loss(new,y).tolist()))
        chosen = min(rows,key=lambda row:row['fit']['best'])['group']
        output.append(dict(name=name,tangent_coefficients=matrix.tolist(),geometry=geometries,
                           groups=rows,fit_selected_group=chosen))
        print(f'Endpoint regrouping {name} saved in memory',flush=True)
    return output


def run(output, reference, source):
    path = reference/'epoch08.pt'
    if existing.sha(path)!=source['checkpoint_sha256']:
        raise ValueError('Reference checkpoint SHA256 mismatch')
    if existing.sha(HEADS/'DTD.pt')!=source['provenance']['head_sha256']:
        raise ValueError('Reference DTD head SHA256 mismatch')
    state = torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    encoder,model,_,flat = build_model(existing,existing.ClassificationHead.load(str(HEADS/'DTD.pt')).cuda())
    del flat
    model.requires_grad_(False).eval()
    encoder.model.visual.load_state_dict(state['visual'])
    op = Operator(model,encoder.model.visual)
    groups,ids,_ = Data(op).groups()
    frames,z,q,scale = (state[k] for k in ('frames','z','q','scale'))
    if len(z)!=5 or len(groups[1][1])!=376:
        raise ValueError('Unexpected endpoint or audit split size')
    with probe_mode(model):
        baseline = [logits(op,x) for x,_ in groups]
        bank = Bank(op,group_rows(frames),groups[0][0])
        j_val = np.stack([features(op,groups[1][0],row.to(bank.device)) for row in bank.dictionary.rows],axis=1)
        j = [bank.j,j_val]
        directions = []
        for i,(v,b) in enumerate(zip(z,q)):
            coefficient = np.zeros(15)
            coefficient[3*i:3*i+3] = scale*trainer.decoder(np.array(v),np.array(b))[0]*np.array(bank.dictionary.norms[3*i:3*i+3])
            directions.append(coefficient)
        radius = .1*float(np.median([bank.norm(d) for d in directions]))
        controls,single = perturbations(op,groups,baseline,bank,z,q,scale,directions,j,radius)
        # Persist intermediate stages for recovery and inspection.
        result = dict(checkpoint_sha256=source['checkpoint_sha256'],ids=ids,z=z,q=q,scale=scale,radius=radius,
                      baseline=[metrics(old,old,y,SEED) for old,(_,y) in zip(baseline,groups)],
                      displacements=[bank.norm(d) for d in directions],
                      geometry=[response_gram(a@np.stack(directions,axis=1)) for a in j],controls=controls)
        existing.write(output/'endpoint_controls.json',result)
        result['interactions'] = interactions(op,groups[1],baseline[1],bank,directions,single)
        result['compensations'] = compensations(op,groups,baseline,bank,directions,z,q,scale)
        existing.write(output/'endpoint_interactions.json',dict(interactions=result['interactions'],compensations=result['compensations']))
        tangent = nonlinear(bank,z,q,scale,list(range(5))).tangent
        result['regroupings'] = regroup(op,groups,baseline,bank,tangent,j,radius)
    existing.write(output/'endpoint.json',result)
