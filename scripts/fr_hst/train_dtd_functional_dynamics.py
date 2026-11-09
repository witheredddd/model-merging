"""Full-split DTD with persistent Fisher–Rao coordinates and adaptive components."""

import argparse
import fcntl
import gc
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import ImageFolder

from run_dtd_carrier_diagnostics import Operator
from train_dtd_online_carrier import (
    DELAY_MAX, DELAY_MIN, commit, decoder, features, functional_pullback, initialize, learn, orthonormal, response,
)
from train_dtd_rolling_innovation import INTERVAL, LEARNING_RATE, ShadowAdamW, probe_mode, test_all
from train_eight_expert_queue import BASE, BATCH_SIZE, HEADS, ROOT, TASKS, build_model, cosine_lr, existing, loader, read, save


STEPS = 216
WARMUP = 75
SEED = 42
GEOMETRY_RATE = .002
GEOMETRY_CAP = .25
ARMS = ('previous_nonlinear', 'single_dynamics', 'adaptive_dynamics')


def tensor(value, like):
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def decode(frames, z, q):
    weights = np.concatenate([decoder(x, a)[0] for x, a in zip(z, q)])
    return tensor(weights, frames) @ frames.flatten(0, 1)


def tangent(z, q):
    out = np.zeros((3 * len(z), 2 * len(z)))
    for i, (x, a) in enumerate(zip(z, q)):
        out[3*i:3*i+3, 2*i:2*i+2] = decoder(x, a)[1]
    return out


def geometry(frames, z, q, responses):
    matrix = tangent(z, q)
    u = tensor(matrix.T, frames) @ frames.flatten(0, 1)
    ru = matrix.T @ responses
    p = (u @ u.T).double().cpu().numpy()
    g = ru @ ru.T
    damping = max(1e-8, .001 * np.trace(g) / np.trace(p))
    return u, ru, g, g + damping * p, float(damping)


def evolve(frames, q, z, physical_dz, residual, rate):
    norm = residual.norm().clamp_min(1e-12)
    axes, bends = [], []
    for a, b, v, dz in zip(frames, q, z, physical_dz):
        jac = decoder(v, b)[1]
        coeff = tensor(jac @ dz, a)
        dfeature = np.array([2*v[0]*dz[0], v[1]*dz[0]+v[0]*dz[1], 2*v[1]*dz[1]])
        outside = residual - (a @ residual) @ a
        axes.append(orthonormal(a - rate * (coeff / coeff.norm().clamp_min(1e-12))[:, None] * outside / norm))
        signed = float((a[2] @ residual) / norm)
        bends.append(np.clip(b-rate*signed*dfeature/max(np.linalg.norm(dz),1e-12), -.5, .5))
    return torch.stack(axes), np.stack(bends)


def ce(op, x, y, update):
    return float(F.cross_entropy(op.evaluate(x, update).double(), y))


def complement(u, ru, metric, damping, direction, image):
    cross = ru @ image + damping * (u @ direction).double().cpu().numpy()
    coefficient = np.linalg.solve(metric, cross)
    return direction - tensor(coefficient, u) @ u


def birth_frame(first, second, axes):
    second = second - (first @ second) * first
    if float(second.norm()) < .15:
        return None
    second = second / second.norm()
    projected = axes - (axes @ first)[:, None] * first - (axes @ second)[:, None] * second
    third = projected[projected.norm(dim=1).argmax()]
    return orthonormal(torch.stack((first,second,third/third.norm())))


def dynamics(op, x, y, raw, batch_gradient, state, step, adaptive, max_components, birth_penalty):
    frames, z, q, scale = (state[k] for k in ('frames','z','q','scale'))
    with probe_mode(op.model):
        gradient = op.gradient(x, y)
        responses = np.stack([response(op, x, axis) for axis in frames.flatten(0, 1)])
        u, ru, g, metric, damping = geometry(frames, z, q, responses)
        projected = (u @ gradient).double().cpu().numpy()
        physical = -np.linalg.solve(metric, projected)
        c = tensor(physical, u) @ u
        eta = min(1., float(.6*raw.norm() / c.norm().clamp_min(1e-12)))
        physical *= eta
        c = tensor(physical, u) @ u
        jvp = physical @ ru
        residual = gradient + (functional_pullback(op,x,jvp) + damping*c) / eta
        dz = physical.reshape(-1,2) / scale
        new_z = z + dz
        old_phi = decode(frames,z,q)
        moved = decode(frames,new_z,q)
        next_frames, next_q = frames,q
        geometry_update = torch.zeros_like(c)
        accepted_rate = 0.
        fisher_geo = 0.
        # Bound the exact decoder displacement in parameter and sampled function space.
        for rate in (GEOMETRY_RATE, GEOMETRY_RATE/4, GEOMETRY_RATE/16):
            candidate, bend = evolve(frames,q,z,physical.reshape(-1,2),residual,rate)
            difference = scale * (decode(candidate,new_z,bend) - moved)
            if float(difference.norm()) > GEOMETRY_CAP * float(c.norm()):
                continue
            displacement = float(np.linalg.norm(response(op,x,difference)))
            if displacement <= GEOMETRY_CAP * np.linalg.norm(jvp) + 1e-12:
                next_frames,next_q,geometry_update = candidate,bend,difference
                accepted_rate,fisher_geo = rate,displacement
                break
        structured = scale * (decode(next_frames,new_z,next_q)-old_phi)
        birth = None
        # Test consecutive batches in two-step blocks, with cooldown after birth.
        if adaptive and len(frames)<max_components and step-state['last_birth']>=16 and step%8 in (0,1):
            direction = -residual / residual.norm().clamp_min(1e-12)
            w = complement(u,ru,metric,damping,direction,response(op,x,direction))
            if float(w.norm()) > 1e-8:
                w = w / w.norm()
                length = structured.norm()
                alternative = structured + .3 * length * w
                alternative *= length / alternative.norm().clamp_min(1e-12)
                original_ce = ce(op,x,y,torch.zeros_like(raw))
                current_ce = ce(op,x,y,structured)
                alternative_ce = ce(op,x,y,alternative)
                base_margin = max(1e-5,.02*abs(original_ce-current_ce))
                margin = base_margin + birth_penalty
                passed = alternative_ce < min(original_ce,current_ce-margin)
                birth = dict(marginal_ce_gain=current_ce-alternative_ce,
                             penalized_ce_gain=current_ce-alternative_ce-birth_penalty,
                             base_margin=base_margin, birth_penalty=birth_penalty,
                             margin=margin, passed=passed, added=False)
                if passed and state['pending_step']==step-1:
                    new_axes = birth_frame(state['pending'],w,next_frames.flatten(0,1))
                    if new_axes is not None:
                        next_frames = torch.cat((next_frames,new_axes.unsqueeze(0)))
                        next_q = np.vstack((next_q,np.zeros(3)))
                        new_z = np.vstack((new_z,np.zeros(2)))
                        state.update(last_birth=step,pending=None,pending_step=-1)
                        birth['added'] = True
                if not birth['added']:
                    state.update(pending=w if passed else None,pending_step=step if passed else -1)
        trial = None
        if step%INTERVAL==0 or step==STEPS:
            before = ce(op,x,y,torch.zeros_like(raw))
            trial = dict(carrier_ce_gain=before-ce(op,x,y,structured),
                         raw_ce_gain=before-ce(op,x,y,raw))
    state.update(frames=next_frames,z=new_z,q=next_q)
    info = dict(z=new_z.tolist(),q=next_q.tolist(),components=len(next_frames),effective_eta=eta,
                damping=damping,fit_fisher_spectrum=np.linalg.eigvalsh(g).tolist(),
                tangent_norm=float(c.norm()),exact_decoder_update_norm=float(structured.norm()),
                geometry_norm=float(geometry_update.norm()),geometry_fisher_norm=fisher_geo,
                geometry_rate=accepted_rate,optimality_residual_norm=float(residual.norm()),
                birth=birth,local_ce=trial)
    return structured,info


def to_disk(state):
    return dict(**{k:v for k,v in state.items() if k not in ('z','q')},z=state['z'].tolist(),q=state['q'].tolist())


def train(arm, output, max_components, warmup=WARMUP, birth_penalty=0.):
    output.mkdir(parents=True,exist_ok=True)
    done = output/'complete.json'
    final = output/'epoch08.pt'
    if done.exists():
        result = read(done)
        assert existing.sha(final)==result['checkpoint_sha256']
        return result
    existing.seed_all(SEED)
    head = existing.ClassificationHead.load(str(HEADS/'DTD.pt')).cuda()
    encoder,model,_,flat = build_model(existing,head)
    del flat
    visual = encoder.model.visual
    model.requires_grad_(False)
    visual.requires_grad_(True)
    op = Operator(model,visual)
    shadow = ShadowAdamW(op)
    pretrained = [p.detach().clone() for p in op.params]
    full = ImageFolder(ROOT/'datasets/dtd/train',transform=encoder.train_preprocess)
    assert len(full)==3760
    data = existing.EpochData(full,SEED)
    batches = (len(data)+BATCH_SIZE-1)//BATCH_SIZE
    if warmup == WARMUP:
        windows = {step for step in range(1,warmup+1) if step%INTERVAL==0 or step%batches==0}
    else:
        windows = {round(warmup*i/5) for i in range(1,6)}
    assert len(windows)==5 and max(windows)==warmup
    schedule = cosine_lr(shadow.optimizer,LEARNING_RATE,500,TASKS['DTD'][0]*batches)
    history,records = [],[]
    window = torch.zeros(sum(op.sizes),device='cuda')
    backlog = torch.zeros_like(window)
    state = None
    resume = output/'resume.pt'
    provenance = dict(arm=arm,seed=SEED,steps=STEPS,train_n=len(full),batch_size=BATCH_SIZE,
                      geometry_batch_prefix=32,warmup=warmup,warmup_window_ends=sorted(windows),
                      max_components=max_components,birth_penalty=birth_penalty,
                      base_sha256=existing.sha(BASE),head_sha256=existing.sha(HEADS/'DTD.pt'))
    if resume.exists():
        saved = torch.load(resume,weights_only=True,map_location='cpu',mmap=True)
        saved_provenance = {'birth_penalty': 0., 'warmup_window_ends': sorted(windows), **saved['provenance']}
        assert saved_provenance==provenance
        visual.load_state_dict(saved['visual'])
        shadow.optimizer.load_state_dict(saved['optimizer'])
        history = [a.cuda() for a in saved['history']]
        if 'window' in saved:
            window = saved['window'].cuda()
        backlog,records = saved['backlog'].cuda(),saved['records']
        state = saved['dynamics']
        if state is not None:
            state.update(frames=state['frames'].cuda(),z=np.array(state['z']),q=np.array(state['q']),
                         pending=state['pending'].cuda() if state['pending'] is not None else None)
        del saved
    for epoch in range(len(records)//batches,(STEPS+batches-1)//batches):
        if len(records)==STEPS:
            break
        data.epoch=epoch
        model.train()
        for i,(images,labels,_) in enumerate(loader(data,8,BATCH_SIZE,SEED+epoch,True)):
            step=epoch*batches+i+1
            if step>STEPS:
                break
            boundary=step%INTERVAL==0 or i+1==batches
            schedule(step-1)
            shadow.clear()
            images,labels=images.cuda(non_blocking=True),labels.cuda(non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=F.cross_entropy(model(images).float(),labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(op.params,1.,error_if_nonfinite=True)
            decay,raw=shadow.propose()
            info=dict(step=step,training_ce=float(loss.detach()))
            if step<=warmup:
                shadow.apply(shadow.shadows,decay,window)
                if step in windows:
                    history.append(window)
                    window=torch.zeros_like(window)
            else:
                if state is None:
                    assert len(history)==5
                    state=dict(frames=initialize(history).unsqueeze(0),z=np.zeros((1,2)),q=np.array([[.08,0.,.08]]),
                               scale=16*float(raw.norm()),pending=None,pending_step=-1,last_birth=warmup)
                    history.clear()
                g=torch.cat([p.grad.detach().flatten() for p in op.params])
                x,y=images[:32],labels[:32]
                proposal=raw+backlog
                if arm=='previous_nonlinear':
                    innovation,frame,bend,old=learn(op,x,state['frames'][0],state['q'][0],proposal,g,backlog,boundary)
                    state.update(frames=frame.unsqueeze(0),q=bend[None],z=np.array(old['latent'])[None])
                    info.update(old)
                    delay=old['delay']
                else:
                    structured,details=dynamics(op,x,y,raw,g,state,step,arm=='adaptive_dynamics',
                                                max_components,birth_penalty)
                    innovation=proposal-structured
                    c_gain=(-g@structured).clamp_min(0)
                    r_gain=(-g@innovation).clamp_min(0)
                    delay=DELAY_MIN+(DELAY_MAX-DELAY_MIN)*float(c_gain/(c_gain+r_gain+1e-12))
                    info.update(details)
                applied=0. if step==STEPS else delay
                backlog,balance=commit(op,shadow,decay,raw,backlog,innovation,applied)
                info.update(**balance,applied_delay=applied)
            records.append(info)
            if step%10==0 or step==STEPS:
                print(f'{arm} step {step}/{STEPS} CE={info["training_ce"]:.5f} M={0 if state is None else len(state["frames"])}',flush=True)
        save(resume,dict(provenance=provenance,visual=visual.state_dict(),optimizer=shadow.optimizer.state_dict(),
                         history=history,window=window,backlog=backlog,records=records,
                         dynamics=to_disk(state) if state is not None else None))
        existing.write(output/'progress.json',dict(step=len(records),last=records[-1]))
    assert len(records)==STEPS
    # Use a training minibatch to report final rank. No test-dependent fitting.
    score_set=ImageFolder(ROOT/'datasets/dtd/train',transform=encoder.val_preprocess)
    indices=torch.randperm(len(score_set),generator=torch.Generator().manual_seed(SEED))[:32]
    x=torch.stack([score_set[i][0] for i in indices]).cuda()
    frames,z,q=(state[k] for k in ('frames','z','q'))
    with probe_mode(model):
        res=np.stack([response(op,x,v) for v in frames.flatten(0,1)])
        u,ru,g,_,_=geometry(frames,z,q,res)
    eig=np.linalg.eigvalsh(g)
    actual=torch.cat([(p.detach()-w).flatten() for p,w in zip(op.params,pretrained)])
    pure=None
    if arm!='previous_nonlinear':
        pure=state['scale']*decode(frames,z,q)
    gram=(frames.flatten(0,1)@frames.flatten(0,1).T).cpu().tolist()
    learned=dict(components=len(frames),coordinate_dimension=2*len(frames),
                 functional_rank_1pct=int((eig>.01*eig.max()).sum()),fisher_eigenvalues=eig.tolist(),
                 pullback_fisher=g.tolist(),parameter_axes_gram=gram,z=z.tolist(),q=q.tolist(),
                 persistent_coordinates=arm!='previous_nonlinear',
                 scale=state['scale'] if pure is not None else None,full_update_norm=float(actual.norm()),
                 direct_decoder_norm=float(pure.norm()) if pure is not None else None,
                 decoder_relative_endpoint_error=float((actual-pure).norm()/actual.norm()) if pure is not None else None)
    if pure is not None:
        learned['component_decoder_norms']=[float((state['scale']*(tensor(decoder(zi,qi)[0],a)@a)).norm())
                                           for a,zi,qi in zip(frames,z,q)]
    save(final,dict(provenance=provenance,visual=visual.state_dict(),frames=frames,z=z.tolist(),q=q.tolist(),
                    scale=learned['scale'],backlog=backlog))
    result=dict(provenance=provenance,records=records,learned=learned,checkpoint_sha256=existing.sha(final))
    if pure is not None:
        frozen=output/'frozen_decoder'
        frozen.mkdir(exist_ok=True)
        # No coefficient fitting after training. Use learned z and Psi as-is.
        with torch.no_grad():
            for p,w,d in zip(op.params,pretrained,op.parts(pure)):
                p.copy_(w+d)
        save(frozen/'epoch08.pt',dict(visual=visual.state_dict(),provenance=provenance))
        existing.write(frozen/'complete.json',dict(checkpoint_sha256=existing.sha(frozen/'epoch08.pt')))
    existing.write(done,result)
    resume.unlink()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-components',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=WARMUP)
    parser.add_argument('--birth-penalty',type=float,default=0.)
    parser.add_argument('--arm',choices=ARMS,action='append',help='Limit training to selected arm(s)')
    args=parser.parse_args()
    assert args.max_components>=2 and 5<=args.warmup<STEPS and 0<=args.birth_penalty<.01
    selected=tuple(dict.fromkeys(args.arm)) if args.arm else ARMS
    args.output.mkdir(parents=True,exist_ok=True)
    existing.ROOT=ROOT
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with (args.output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        arms,results=[],{}
        for arm in selected:
            folder=args.output/arm
            results[arm]=train(arm,folder,args.max_components,args.warmup,args.birth_penalty)
            arms.append(folder)
            if arm!='previous_nonlinear':
                arms.append(folder/'frozen_decoder')
            gc.collect()
            torch.cuda.empty_cache()
        existing.write(args.output/'training_complete.json',dict(arms=list(selected)))
        expected=read(ROOT/'datasets/DTD_test_provenance.json')
        for row in expected['files']:
            assert existing.sha(ROOT/'datasets/_downloads/tanganke__dtd'/row['path'])==row['sha256']
        test_all(args.output,arms,expected)
        for arm,value in results.items():
            folder=args.output/arm
            value['test']=read(folder/'historical_test.json')
            if arm!='previous_nonlinear':
                value['frozen_decoder_test']=read(folder/'frozen_decoder/historical_test.json')
        existing.write(args.output/'results.json',dict(arms=results))


if __name__=='__main__':
    main()
