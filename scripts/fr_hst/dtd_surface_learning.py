"""Persistent quadratic groups trained by alternating exact task CE updates."""

import numpy as np
import torch
from torch.nn import functional as F

from dtd_projected_groups import chart_coefficients, chart_tangent, loss
from train_dtd_online_carrier import decoder, orthonormal, response
from train_dtd_rolling_innovation import probe_mode


WINDOW = 12
FIT_BATCHES = 6
CONFIRM_BATCHES = 3
REPLAY = FIT_BATCHES + CONFIRM_BATCHES
REPLAY_IMAGES = 24
COORDINATE_ROUNDS = 4
SURFACE_ROUNDS = 2
AXIS_CAP = .15
BEND_CAP = .08
BIRTH_MARGIN = .0002


def gradient_at(op, x, y, update):
    with torch.enable_grad():
        if update is None:
            weights = op.params
        else:
            weights = tuple((p.detach()+d).requires_grad_() for p,d in zip(op.params,op.parts(update)))
        logits = torch.func.functional_call(op.model,dict(zip(op.names,weights)),(x,)).float()
        value = F.cross_entropy(logits.double(),y)
        gradient = torch.cat([g.detach().flatten() for g in torch.autograd.grad(value,weights)])
    return float(value.detach()),gradient


class Chart:
    def __init__(self, op, frames, z, q, scale, metric_images):
        self.frames,self.z,self.q,self.scale = frames,z,q,scale
        self.flat = frames.flatten(0,1)
        self.parameter = (self.flat@self.flat.T).double().cpu().numpy()
        self.tangent = chart_tangent(z,q)
        self.response = np.stack([response(op,metric_images,self.vector(w)) for w in self.tangent.T])
        self.fisher = self.response@self.response.T
        p = self.tangent.T@self.parameter@self.tangent
        mu = max(1e-8,.001*np.trace(self.fisher)/max(np.trace(p),1e-12))
        self.inverse = np.linalg.pinv(self.fisher+mu*p,rcond=1e-8,hermitian=True)

    def vector(self, coefficients):
        return torch.as_tensor(coefficients,device=self.flat.device,dtype=self.flat.dtype)@self.flat

    def coefficients(self, a):
        return chart_coefficients(self.z,self.q,a,self.scale)

    def bounded(self, a, radius):
        w = self.coefficients(a)
        if w@self.parameter@w <= radius**2:
            return a,w
        lo,hi = 0.,1.
        for _ in range(18):
            mid = (lo+hi)/2
            w = self.coefficients(mid*a)
            if w@self.parameter@w <= radius**2:
                lo = mid
            else:
                hi = mid
        return lo*a,self.coefficients(lo*a)


def optimize_coordinates(op, x, y, chart, radius, terminal_gradient=False):
    """Four actual CE gradients with a fixed Fisher preconditioner and exact bounds."""
    a = np.zeros(2*len(chart.z))
    w = chart.coefficients(a)
    value,gradient = gradient_at(op,x,y,None)
    baseline = value
    accepted = 0
    for iteration in range(COORDINATE_ROUNDS):
        t = chart_tangent(chart.z+a.reshape(-1,2),chart.q)
        gradient_small = t.T@(chart.flat@gradient).double().cpu().numpy()
        descent = -chart.inverse@gradient_small/chart.scale
        # Normalize the proposal in the local parameter metric before backtracking.
        d = chart.scale*t@descent
        descent *= min(1.,radius/max(float(np.sqrt(max(0.,d@chart.parameter@d))),1e-12))
        for fraction in (1.,.5,.25):
            trial,trial_w = chart.bounded(a+fraction*descent,radius)
            delta = chart.vector(trial_w)
            if float(delta.norm())>radius*(1+2e-6):
                continue
            score = loss(op,x,y,delta)
            if score<value-1e-10:
                a,w,value = trial,trial_w,score
                accepted += 1
                if iteration+1<COORDINATE_ROUNDS or terminal_gradient:
                    value,gradient = gradient_at(op,x,y,delta)
                break
        else:
            break
    return a,w,value,baseline,accepted,gradient.cpu() if terminal_gradient else None


def replay_solutions(op, frames, z, q, scale, batches, radius, gradients=False):
    device = op.params[0].device
    chart = Chart(op,frames.to(device),z,q,scale,batches[0][0][:12])
    return [optimize_coordinates(op,x,y,chart,radius,terminal_gradient=gradients) for x,y in batches]


def surface_proposal(frames, q, reference_frames, reference_q, z, scale, solutions, device):
    """Envelope gradients at the optimized virtual CE endpoints, with fixed coordinates."""
    vectors = torch.stack([row[-1] for row in solutions]).to(device)
    weights = np.stack([row[1] for row in solutions])
    proposed = torch.empty_like(frames)
    next_q = q.copy()
    for i,axis in enumerate(frames):
        axis = axis.to(device)
        reference = reference_frames[i].to(device)
        coefficients = torch.as_tensor(weights[:,3*i:3*i+3].T.copy(),device=device,dtype=axis.dtype)
        grad_a = coefficients@vectors/len(solutions)
        gq = (vectors@axis[2]).double().cpu().numpy()
        z1,z2 = z[i]
        moves = np.stack([row[0].reshape(-1,2)[i] for row in solutions])
        a,b = moves.T
        quadratic = np.stack((a*(2*z1+a),z1*b+z2*a+a*b,b*(2*z2+b)),axis=1)
        grad_q = scale*(gq@quadratic)/len(solutions)
        cross = grad_a@axis.T
        grad_a.sub_(.5*(cross+cross.T)@axis)
        rate = min(50.,.075*float(axis.norm())/max(float(grad_a.norm()),1e-12))
        proposed[i].copy_(axis)
        for fraction in (1.,.5,.25):
            moved = orthonormal(axis-fraction*rate*grad_a)
            if float((moved-reference).norm()) <= AXIS_CAP*float(reference.norm()):
                proposed[i].copy_(moved)
                change = q[i]-fraction*rate*grad_q-reference_q[i]
                change *= min(1.,BEND_CAP/max(np.linalg.norm(change),1e-12))
                next_q[i] = np.clip(reference_q[i]+change,-.5,.5)
                break
    return proposed,next_q


def fit_surface(op, frames, z, q, scale, batches, radius, reference=None):
    reference_frames,reference_q = (frames,q) if reference is None else reference
    trace = []
    solutions = replay_solutions(op,frames,z,q,scale,batches,radius,True)
    trace.append(float(np.mean([row[2] for row in solutions])))
    for _ in range(SURFACE_ROUNDS):
        proposal,bend = surface_proposal(frames,q,reference_frames,reference_q,z,scale,solutions,op.params[0].device)
        moved = replay_solutions(op,proposal,z,bend,scale,batches,radius,True)
        objective = float(np.mean([row[2] for row in moved]))
        trace.append(objective)
        if objective < float(np.mean([row[2] for row in solutions])):
            frames,q,solutions = proposal,bend,moved
    return frames,q,solutions,trace


def offset_coefficients(state):
    # The full offset is d_m + A_m @ offset_coefficients[m]. Store it implicitly.
    return [-state['scale']*decoder(v,q)[0] for v,q in zip(state['z'],state['q'])]


def initialize_birth(solutions, device):
    residuals = torch.stack([row[-1] for row in solutions])
    norms = residuals.norm(dim=1).clamp_min(1e-12)
    residuals.div_(norms[:,None])
    eig,rot = torch.linalg.eigh((residuals@residuals.T).double())
    if float(eig[-3]) <= 1e-6*float(eig[-1]):
        return None
    axes = rot[:,-3:].T.flip(0).float()@residuals
    return orthonormal(axes.to(device)).cpu()


def birth_response(op, frames, z, q, seed, previous, x):
    chart = Chart(op,frames.to(x.device),z,q,1.,x)
    old = chart.response.T
    pinv = np.linalg.pinv(old,rcond=1e-6)
    current = np.stack([response(op,x,axis.to(x.device)) for axis in seed[:2]],axis=1)
    residual = current-old@(pinv@current)
    novelty = float(np.linalg.norm(residual)/max(np.linalg.norm(current),1e-12))
    overlap = None
    if previous is not None:
        prior = response(op,x,previous.to(x.device))
        prior = prior-old@(pinv@prior)
        if np.linalg.norm(prior)>1e-12:
            overlap = float(np.linalg.norm(residual@(np.linalg.pinv(residual,rcond=1e-6)@prior))/np.linalg.norm(prior))
    return novelty,overlap


def refresh(op, state, replay, recent, radius, max_components, correction=None):
    frames = state['frames'].cpu()
    state['frames'] = frames  # Stage only one chart on the GPU at a time during replay.
    q,z,scale = state['q'],state['z'],state['scale']
    window_frames,window_q = frames,q
    device = op.params[0].device
    fit_rows,confirm_rows = replay[:FIT_BATCHES],replay[FIT_BATCHES:]
    seen = set().union(*(ids.tolist() for _,_,ids in fit_rows))
    fit = [(x.to(device),y.to(device)) for x,y,_ in fit_rows]
    confirm = []
    for x,y,ids in confirm_rows:
        keep = [i for i,idx in enumerate(ids.tolist()) if idx not in seen]
        if keep:
            confirm.append((x[keep].to(device),y[keep].to(device)))
    if len(confirm)!=CONFIRM_BATCHES:
        state['frames'] = frames.to(device)
        state['pending_surface'] = None
        return dict(policy='frozen',reason='insufficient_disjoint_confirmation_batches',birth=None)

    def validation(f,p,v):
        return np.array([row[2] for row in replay_solutions(op,f,v,p,scale,confirm,radius)])

    start = validation(frames,q,z)
    adjusted,bend,solutions,trace = fit_surface(op,frames,z,q,scale,fit,radius)
    changed = validation(adjusted,bend,z)
    policy = 'frozen'
    if changed.mean()<start.mean() and np.all(changed<=start+BIRTH_MARGIN):
        frames,q,policy = adjusted,bend,'adapted'
    else:
        del solutions
        solutions = replay_solutions(op,frames,z,q,scale,fit,radius,True)
    del adjusted,bend
    held = changed if policy=='adapted' else start
    correction_info = None
    if correction is not None:
        frames,held,correction_info = correction(frames,q,z,window_frames,held,validation)
        if correction_info['chosen_angle']:
            policy = 'task_adapted'
            del solutions
            solutions = replay_solutions(op,frames,z,q,scale,fit,radius,True)
    gaps = sum(r['reference_ce_gain']>r['coordinate_ce_gain']+BIRTH_MARGIN for r in recent)
    trial = None
    previous = state.get('pending_surface')
    state['pending_surface'] = None
    if gaps>=3 and len(frames)<max_components:
        seed = initialize_birth(solutions,device)
        if seed is not None:
            novelty,overlap = birth_response(op,frames,z,q,seed,previous,confirm[0][0][:12])
            trial = dict(novelty=novelty,recurrence=overlap,added=False)
            if novelty>.10:
                state['pending_surface'] = seed[0].clone()
            if novelty>.10 and overlap is not None and overlap>.35:
                # Both alternatives have the same CE coordinate and surface iteration budgets.
                del solutions
                old,old_q,old_solutions,old_trace = fit_surface(op,frames,z,q,scale,fit,radius,(window_frames,window_q))
                del old_solutions
                old_ce = validation(old,old_q,z)
                expanded = torch.cat((frames,seed[None]),dim=0)
                new_z = np.vstack((z,np.zeros(2)))
                new_q = np.vstack((q,[.03,0.,.03]))
                extended_reference = (*window_frames,seed),np.vstack((window_q,[.03,0.,.03]))
                new,new_q,new_solutions,new_trace = fit_surface(op,expanded,new_z,new_q,scale,fit,radius,extended_reference)
                del expanded,new_solutions
                new_ce = validation(new,new_q,new_z)
                gain = np.minimum(old_ce,held)-new_ce
                trial.update(ce_marginal=gain.tolist(),old_trace=old_trace,new_trace=new_trace,
                             added=bool(np.all(gain>BIRTH_MARGIN)))
                if trial['added']:
                    frames,q,z,held,policy = new,new_q,new_z,new_ce,'birth'
                    state['cumulative'].append(torch.zeros_like(state['cumulative'][0]))
                    state['pending_surface'] = None
                elif old_ce.mean()<held.mean() and np.all(old_ce<=held+BIRTH_MARGIN):
                    frames,q,held,policy = old,old_q,old_ce,'adapted'
    state.update(frames=frames.to(device),q=q,z=z)
    state['offset_coefficients'] = offset_coefficients(state)
    return dict(policy=policy,fit_ce_trace=trace,confirm_before=start.tolist(),confirm_after=held.tolist(),
                task_correction=correction_info,
                reference_gaps=gaps,birth=trial,q=q.tolist(),fit_images=[len(x) for x,_ in fit],
                confirmation_images=[len(x) for x,_ in confirm])


def canonical_total(state):
    value = state['warmup_anchor'].double()
    for d in state['cumulative']:
        value.add_(d)
    return value


@torch.no_grad()
def step(op, shadow, images, labels, state, replay, recent, since_warmup, max_components,
         first_refresh=1, correction=None):
    raw = torch.cat([(p-w).flatten() for p,w in zip(shadow.shadows,op.params)])
    radius = float(raw.norm())
    boundary = None
    with probe_mode(op.model):
        if since_warmup>=first_refresh and (since_warmup-first_refresh)%WINDOW==0 and len(replay)==REPLAY and radius>0:
            boundary = refresh(op,state,replay,recent,radius,max_components,correction)
        x,y = images[:32],labels[:32]
        mx,my = images[32:64],labels[32:64]
        chart = Chart(op,state['frames'],state['z'],state['q'],state['scale'],x)
        a,w,fit_after,fit_before,iterations,_ = optimize_coordinates(op,x,y,chart,radius)
        before = loss(op,mx,my)
        reference = loss(op,mx,my,raw)
        value = before
        norm,ratio,cosine = 0.,0.,None
        if iterations:
            delta = chart.vector(w)
            trial = []
            for i,(frame,d) in enumerate(zip(state['frames'],state['cumulative'])):
                c = torch.as_tensor(w[3*i:3*i+3],device=frame.device,dtype=frame.dtype)
                trial.append(d+(c@frame).cpu())
            # Use the exact checkpoint reconstruction order even for the CE gate.
            total = canonical_total(dict(state,cumulative=trial))
            weights = op.parts(total.float().to(raw.device))
            logits = torch.func.functional_call(op.model,dict(zip(op.names,weights)),(mx,)).float()
            candidate = float(F.cross_entropy(logits.double(),my))
            if candidate<before:
                rw,rv = response(op,x,delta),response(op,x,raw)
                denom = float(rv@rv)
                ratio = float(rw@rw/denom) if denom>0 else None
                size = np.linalg.norm(rw)*np.linalg.norm(rv)
                cosine = float(rw@rv/size) if size>0 else None
                state['cumulative'] = trial
                state['z'] = state['z']+a.reshape(-1,2)
                state['offset_coefficients'] = offset_coefficients(state)
                for param,weight in zip(op.params,weights):
                    param.copy_(weight)
                norm = float(delta.norm())
                value = candidate
    return dict(window=boundary,components=len(state['z']),z=state['z'].tolist(),q=state['q'].tolist(),
                coordinate_iterations=iterations,fit_ce_gain=fit_before-fit_after,
                coordinate_ce_gain=before-value,reference_ce_gain=before-reference,coordinate_norm=norm,
                reference_norm=radius,fisher_ratio=ratio,response_cosine=cosine,
                pullback_fisher=chart.fisher.tolist(),unstructured_write_norm=0.)
