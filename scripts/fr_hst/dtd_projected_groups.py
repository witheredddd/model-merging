"""Windowed DTD direction learning and coordinate-only parameter updates."""

import numpy as np
import torch
from torch.nn import functional as F

from train_dtd_online_carrier import decoder, response
from train_dtd_rolling_innovation import probe_mode


WINDOW = 12
REPLAY = 6
REPLAY_IMAGES = 16
AXIS_CAP = .15


def loss(op, x, y, delta=None):
    return float(F.cross_entropy(op.evaluate(x,delta).double(),y))


def combine(bank, coefficients):
    out = torch.zeros_like(bank[0])
    for axis,value in zip(bank,coefficients):
        if value:
            out.add_(axis,alpha=float(value))
    return out


def chart_coefficients(z, q, a, scale):
    return scale*np.concatenate([decoder(v+d,b)[0]-decoder(v,b)[0] for v,b,d in zip(z,q,a.reshape(-1,2))])


def chart_tangent(z, q):
    out = np.zeros((3*len(z),2*len(z)))
    for i,(v,b) in enumerate(zip(z,q)):
        out[3*i:3*i+3,2*i:2*i+2] = decoder(v,b)[1]
    return out


def solve(gram, parameter, rhs):
    damping = max(1e-8,.001*np.trace(gram)/max(np.trace(parameter),1e-12))
    return np.linalg.pinv(gram+damping*parameter,rcond=1e-8,hermitian=True)@rhs


def candidates(frames, z, q, scale, radius, directions):
    flat = frames.flatten(0,1)
    parameter = (flat@flat.T).double().cpu().numpy()
    tangent = chart_tangent(z,q)
    for name,a in directions:
        v = scale*tangent@a
        a = a*min(1.,.98*radius/max(float(np.sqrt(max(0.,v@parameter@v))),1e-12))
        for size in (1.,.5,.25,.125):
            dz = size*a
            w = chart_coefficients(z,q,dz,scale)
            if w@parameter@w <= radius**2*(1+1e-7):
                yield name,dz,w


def choose_step(op, x, y, monitor, frames, z, q, scale, radius, directions):
    flat = frames.flatten(0,1)
    start = loss(op,x,y)
    by_direction = {}
    for name,a,w in candidates(frames,z,q,scale,radius,directions):
        score = loss(op,x,y,torch.as_tensor(w,device=flat.device,dtype=flat.dtype)@flat)
        if score < by_direction.get(name,(start,None,None))[0]:
            by_direction[name] = score,a,w
    if monitor is None:
        return min(by_direction.values(),default=(start,None,None),key=lambda row:row[0])
    mx,my = monitor
    best = loss(op,mx,my),None,None
    for _,a,w in by_direction.values():
        score = loss(op,mx,my,torch.as_tensor(w,device=flat.device,dtype=flat.dtype)@flat)
        if score < best[0]:
            best = score,a,w
    return best


def bank_coordinates(bank, vectors):
    return np.array([[float(u@v) for v in vectors] for u in bank])


def orthogonal_bank(vectors):
    bank = []
    for v in vectors:
        residual = v.clone()
        for _ in range(2):
            for u in bank:
                residual.add_(u,alpha=-float(u@residual))
        size = float(residual.norm())
        if size > 1e-5*float(v.norm()):
            bank.append(residual.div_(size))
    return bank


def q_score(u, metrics, gradients):
    if u.shape[1] == 0:
        return 0.
    values = []
    for g,b in zip(metrics,gradients):
        projected = u.T@b
        values.append(projected@np.linalg.pinv(u.T@g@u,rcond=1e-8,hermitian=True)@projected)
    return float(np.mean(values))


def fit_axes(initial, incumbent, metrics, gradients):
    """Optimize the shared Q objective in a small, orthonormal parameter bank."""
    old = torch.from_numpy(incumbent)
    count = old.shape[1]
    u = torch.from_numpy(initial.copy()).requires_grad_(True)
    metric,b = torch.from_numpy(metrics),torch.from_numpy(gradients)
    optim = torch.optim.Adam([u],lr=.025)
    eye = torch.eye(u.shape[1],dtype=u.dtype)
    best = initial.copy()
    with torch.enable_grad():
        objective = -float('inf')
        for i in range(17):
            optim.zero_grad()
            proj = b@u
            k = u.T@metric@u+1e-8*eye
            score = (proj*torch.linalg.solve(k,proj.unsqueeze(-1)).squeeze(-1)).sum(dim=-1).mean()
            if float(score.detach()) > objective:
                objective,best = float(score.detach()),u.detach().numpy().copy()
            if i == 16:
                break
            (-score).backward()
            optim.step()
            with torch.no_grad():
                for j in range(0,count,2):
                    d = u[:,j:j+2]-old[:,j:j+2]
                    limit = AXIS_CAP*old[:,j:j+2].norm()
                    u[:,j:j+2].copy_(old[:,j:j+2]+d*(limit/d.norm().clamp_min(limit)))
                if u.shape[1] > count:
                    u[:,count:].copy_(torch.linalg.qr(u[:,count:]).Q)
    return best


def birth_axes(old, metrics, gradients):
    # Gradients of the unexplained quadratic task gain, evaluated at one anchor.
    gaps = []
    for metric,b in zip(metrics,gradients):
        a = np.linalg.pinv(old.T@metric@old,hermitian=True)@(old.T@b)
        gaps.append(b-metric@old@a)
    _,s,vt = np.linalg.svd(np.stack(gaps),full_matrices=False)
    return vt[:2].T if len(s)>1 and s[1] > 1e-4*max(s[0],1e-12) else None


def materialize(bank, u, frames, q):
    groups = u.shape[1]//2
    new = frames.new_empty((groups,*frames.shape[1:]))
    bend = np.zeros((groups,3))
    bend[:len(q)] = q
    for i in range(groups):
        for j in range(2):
            new[i,j].copy_(combine(bank,u[:,2*i+j]))
        new[i,2].copy_(frames[min(i,len(frames)-1),2])
    return new,bend,np.zeros((groups,2))


def refresh(op, state, replay, recent, radius, max_components, margin):
    """Compare geometry at fixed weights. The selected chart is re-anchored here."""
    frames,z,q,scale = (state[k] for k in ('frames','z','q','scale'))
    probes = [(x.to(frames.device),y.to(frames.device)) for x,y in replay]
    old_axes = []
    for frame,v,b in zip(frames,z,q):
        jac = decoder(v,b)[1]
        old_axes.extend(frame[j]+float(jac[2,j])*frame[2] for j in range(2))
    with torch.enable_grad():
        full_grad = [op.gradient(x,y) for x,y in probes]
    bank = orthogonal_bank([*old_axes,*full_grad[:-2]])
    old = bank_coordinates(bank,old_axes)
    vectors = bank_coordinates(bank,full_grad).T
    del full_grad,old_axes
    grams = []
    for x,_ in probes:
        r = np.stack([response(op,x,u) for u in bank])
        grams.append(r@r.T)
    grams = np.stack(grams)
    mu = np.maximum(1e-8,.001*np.trace(grams,axis1=1,axis2=2)/len(bank))
    metrics = grams+mu[:,None,None]*np.eye(len(bank))
    gf,bf = metrics[:-2],vectors[:-2]
    adapted = fit_axes(old,old,gf,bf)

    def confirm(u):
        f,b,p = materialize(bank,u,frames,q)
        scores = []
        for (x,y),g,gradient in zip(probes[-2:],metrics[-2:],vectors[-2:]):
            a = -np.linalg.pinv(u.T@g@u,rcond=1e-8,hermitian=True)@(u.T@gradient)/scale
            value,_,_ = choose_step(op,x,y,None,f,p,b,scale,radius,[('task',a)])
            scores.append(value)
        return np.array(scores)

    before,after = confirm(old),confirm(adapted)
    improved = q_score(adapted,gf,bf)>q_score(old,gf,bf) and np.all(after<=before) and np.any(after<before)
    chosen,held = (adapted,after) if improved else (old,before)
    policy = 'adapted' if improved else 'frozen'
    gaps = sum(r['ce_adamw_gain']>r['ce_projected_gain']+margin for r in recent)
    trial = None
    if gaps>=3 and len(frames)<max_components:
        extra = birth_axes(chosen,gf,bf)
        if extra is not None:
            # Equal Q optimization budget and exact CE search for old and old+new.
            refined = fit_axes(chosen,old,gf,bf)
            expanded = fit_axes(np.column_stack((chosen,extra)),old,gf,bf)
            ref_ce,new_ce = confirm(refined),confirm(expanded)
            if np.all(ref_ce<=held):
                chosen,held,policy = refined,ref_ce,'adapted'
            delta = held-new_ce
            pass_q = q_score(expanded,gf,bf)>q_score(chosen,gf,bf)
            trial = dict(ce_marginal=delta.tolist(),q_improves=pass_q,
                         added=bool(pass_q and np.all(delta>margin)))
            if trial['added']:
                chosen,held,policy = expanded,new_ce,'birth'
    if policy!='frozen':
        state['frames'],state['q'],state['z'] = materialize(bank,chosen,frames,q)
        if policy=='birth':
            state['cumulative'].append(torch.zeros_like(state['cumulative'][0]))
    total = q_score(chosen,gf,bf)
    necessary = [total-q_score(np.delete(chosen,[2*i,2*i+1],axis=1),gf,bf)
                 for i in range(chosen.shape[1]//2)]
    return dict(policy=policy,fit_replay_batches=len(probes)-2,confirm_replay_batches=2,
                gaps=gaps,q_before=q_score(old,gf,bf),q_after=total,conditional_q=necessary,
                ce_before=before.tolist(),ce_after=held.tolist(),birth=trial)


@torch.no_grad()
def step(op, shadow, images, labels, state, replay, recent, since_warmup, max_components, margin):
    # Include the decay part in the reference. The reference itself never writes.
    raw = torch.cat([(p-w).flatten() for p,w in zip(shadow.shadows,op.params)])
    radius = float(raw.norm())
    refresh_info = None
    with probe_mode(op.model):
        if (since_warmup-1)%WINDOW == 0 and len(replay)==REPLAY and radius>0:
            refresh_info = refresh(op,state,replay,recent,radius,max_components,margin)
        frames,z,q,scale = (state[k] for k in ('frames','z','q','scale'))
        flat = frames.flatten(0,1)
        t = chart_tangent(z,q)
        x,y = images[:32],labels[:32]
        monitor = images[32:64],labels[32:64]
        with torch.enable_grad():
            gradient = op.gradient(x,y)
        r = np.stack([response(op,x,d) for d in flat])
        rv = response(op,x,raw)
        ru = t.T@r
        gram = ru@ru.T
        p = t.T@(flat@flat.T).double().cpu().numpy()@t
        b = t.T@(flat@gradient).double().cpu().numpy()
        projection = solve(gram,p,ru@rv)/scale
        natural = -solve(gram,p,b)/scale
        before = loss(op,*monitor)
        raw_ce = loss(op,*monitor,raw)
        value,a,w = choose_step(op,x,y,monitor,frames,z,q,scale,radius,
                                [('projected',projection),('task',natural)])
        cosine,ratio,applied_norm = None,0.,0.
        if w is not None:
            delta = torch.as_tensor(w,device=flat.device,dtype=flat.dtype)@flat
            applied_norm = float(delta.norm())
            rstep = w@r
            denom = float(rv@rv)
            ratio = float(rstep@rstep/denom) if denom>0 else None
            normalizer = np.linalg.norm(rstep)*np.linalg.norm(rv)
            cosine = float(rstep@rv/normalizer) if normalizer>0 else None
            for i,(frame,acc) in enumerate(zip(frames,state['cumulative'])):
                coef = torch.as_tensor(w[3*i:3*i+3],device=flat.device,dtype=flat.dtype)
                acc.add_((coef@frame).cpu())
            for param,change in zip(op.params,op.parts(delta)):
                param.add_(change)
            state['z'] = z+a.reshape(-1,2)
    return dict(geometry=refresh_info,window_position=(since_warmup-1)%WINDOW+1,components=len(frames),
                z=state['z'].tolist(),ce_projected_gain=before-value,ce_adamw_gain=before-raw_ce,
                functional_ratio=ratio,response_cosine=cosine,projected_norm=applied_norm,
                adamw_reference_norm=radius,pullback_fisher=gram.tolist(),unstructured_write_norm=0.)
