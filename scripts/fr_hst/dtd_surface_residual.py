"""Bounded task-gradient correction of one existing CE Surfaces tangent."""

import numpy as np
import torch

from dtd_projected_groups import chart_tangent
from dtd_surface_learning import AXIS_CAP, BIRTH_MARGIN


ANGLES = (5,10,15)


def explained(gram, target):
    if len(target)==0:
        return 0.
    return float(target@np.linalg.pinv(gram,rcond=1e-8,hermitian=True)@target)


def geometry(frames, z, q, gradient):
    flat = frames.flatten(0,1)
    coefficients = chart_tangent(z,q)
    gram = coefficients.T@(flat@flat.T).double().numpy()@coefficients
    target = coefficients.T@(flat@gradient).double().numpy()
    return gram,target,coefficients


def rotate(frame, tangent, residual, degrees):
    angle = np.deg2rad(degrees)
    a,b = frame@tangent,frame@residual
    out = frame.clone()
    out.addr_((np.cos(angle)-1)*a-np.sin(angle)*b,tangent)
    out.addr_((np.cos(angle)-1)*b+np.sin(angle)*a,residual)
    return out


def correct(frames, q, z, window_frames, held, validation, gradient, next_gradient):
    gram,target,coefficients = geometry(frames,z,q,gradient)
    total = float(gradient.norm())**2
    base = explained(gram,target)
    joint = frames.flatten(0,1)
    projection = coefficients@np.linalg.pinv(gram,rcond=1e-8,hermitian=True)@target
    residual = gradient-torch.from_numpy(projection).float()@joint
    size = float(residual.norm())
    info = dict(chosen_angle=0,residual_fraction=size/max(total**.5,1e-12),trials=[])
    if size<=max(total**.5*1e-6,1e-12):
        return frames,held,info
    contributions = []
    for j in range(len(target)):
        keep = np.arange(len(target))!=j
        contributions.append(max(0.,base-explained(gram[np.ix_(keep,keep)],target[keep])))
    j = int(np.argmin(contributions))
    group = j//2
    tangent = torch.from_numpy(coefficients[:,j].copy()).float()@joint
    tangent.div_(tangent.norm().clamp_min(1e-12))
    residual.add_(tangent,alpha=-float(residual@tangent))
    residual.div_(residual.norm().clamp_min(1e-12))
    if float(tangent@gradient)<0:
        residual.neg_()
    info.update(group=group+1,coordinate=j%2+1,conditional_gradient_energy=contributions)

    def apply(degrees):
        change = rotate(frames[group],tangent,residual,degrees)
        if float((change-window_frames[group]).norm())>AXIS_CAP*float(window_frames[group].norm()):
            return None
        candidate = frames.clone()
        candidate[group].copy_(change)
        return candidate

    chosen,best = 0,held
    for angle in ANGLES:
        candidate = apply(angle)
        if candidate is None:
            info['trials'].append(dict(angle=angle,within_axis_cap=False))
            continue
        scores = validation(candidate,q,z)
        del candidate
        gain = held-scores
        admissible = bool(np.all(gain>=0) and gain.mean()>BIRTH_MARGIN)
        info['trials'].append(dict(angle=angle,within_axis_cap=True,ce_gain=gain.tolist(),accepted=admissible))
        if admissible and scores.mean()<best.mean():
            chosen,best = angle,scores

    final = apply(chosen) if chosen else frames
    future_norm = max(float(next_gradient.norm())**2,1e-24)
    g0,b0,_ = geometry(frames,z,q,next_gradient)
    g1,b1,_ = geometry(final,z,q,next_gradient) if chosen else (g0,b0,None)
    info.update(chosen_angle=chosen,confirmation_ce_gain=(held-best).tolist(),
                next_batch_gradient_coverage=[explained(g0,b0)/future_norm,explained(g1,b1)/future_norm])
    return final,best,info
