"""Read-only fixed-state six-arm CE Surfaces counterfactuals, frozen 6c78fee."""
import argparse,copy,hashlib,json,sys,time,math,gc
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from algebra import orthogonal_coefficients,trust,refusal

def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,float) and not math.isfinite(v):return None
    return v

def write(path,v):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(clean(v),indent=2,allow_nan=False));temp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--frozen',type=Path,required=True);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    if (args.output/'results.json').exists():raise ValueError('Use new output; do not overwrite completed diagnostics')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    sys.path[:0]=[str(args.frozen/'scripts/fr_hst'),str(args.root),str(args.frozen/'scripts'),str(args.frozen/'scripts/tfpml')]
    import train_eight_expert_queue as assets
    import dtd_surface_learning as surface
    from run_dtd_carrier_diagnostics import Operator
    from train_dtd_rolling_innovation import probe_mode
    assets.ROOT=args.root;assets.BASE=args.root/'models/checkpoints/ViT-B-16/pretrained.pt'
    assets.HEADS=args.root/'results/functional_geometry_v2_no_expert/assets/heads';assets.existing.ROOT=args.root
    core=np._core if hasattr(np,'_core') else np.core
    with torch.serialization.safe_globals([core.multiarray.scalar,np.dtype,type(np.dtype('float64'))]):
        saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    provenance=saved['provenance']
    if assets.existing.sha(assets.BASE)!=provenance['base_sha256'] or assets.existing.sha(assets.HEADS/'DTD.pt')!=provenance['head_sha256']:
        raise ValueError('Checkpoint/base/head identity mismatch')
    encoder,model,_,unused=assets.build_model(assets.existing,assets.existing.ClassificationHead.load(str(assets.HEADS/'DTD.pt')))
    del unused
    model.requires_grad_(False);visual=encoder.model.visual;visual.requires_grad_(True)
    visual.load_state_dict(saved['visual']);op=Operator(model,visual)
    state=dict(saved['surface']);state['z']=np.array(state['z']);state['q']=np.array(state['q'])
    canonical=surface.canonical_total(state).float();actual=torch.cat([p.detach().flatten().cpu() for p in op.params])
    if not torch.equal(canonical,actual):raise ValueError('Archive visual/canonical reconstruction mismatch')
    base_hash=hashlib.sha256(actual.numpy().data).hexdigest();del actual,canonical
    frames=state['frames'].cuda();replay=saved['replay'];counts=dict(forwards=0,gradients=0,jvps=0,images=0)
    def ce(x,y,delta=None,absolute=None):
        counts['forwards']+=1;counts['images']+=len(y)
        with torch.no_grad():
            if absolute is not None:
                logits=torch.func.functional_call(model,dict(zip(op.names,op.parts(absolute))),(x,)).float()
            else:logits=op.evaluate(x,delta)
            return float(F.cross_entropy(logits.double(),y))
    original_gradient=surface.gradient_at;original_response=surface.response
    def counted_gradient(operator,x,y,delta=None):
        counts['gradients']+=1;counts['forwards']+=1;counts['images']+=len(y)
        return original_gradient(operator,x,y,delta)
    def counted_response(operator,x,v):
        counts['jvps']+=1;counts['forwards']+=1;counts['images']+=len(x)
        return original_response(operator,x,v)
    # Instrument only call counters around the frozen equations, including calls
    # inside its Chart and optimizer. Files and mathematical behavior stay fixed.
    surface.gradient_at=counted_gradient;surface.response=counted_response
    surface.loss=lambda operator,x,y,delta=None:ce(x,y,delta)
    def gradient(x,y,delta=None):return surface.gradient_at(op,x,y,delta)
    def response(x,v):return surface.response(op,x,v)
    def adamw(g):
        # Clone all optimizer state; the real model and archived moments are untouched.
        params=[torch.nn.Parameter(p.detach().clone()) for p in op.params]
        old=saved['optimizer'];group=old['param_groups'][0]
        if len(group['params'])!=len(params):raise ValueError('Optimizer parameter order mismatch')
        optimizer=torch.optim.AdamW(params,lr=group['lr'],betas=tuple(group['betas']),eps=group['eps'],
            weight_decay=group['weight_decay'],amsgrad=group.get('amsgrad',False),fused=True)
        for p,pid,part in zip(params,group['params'],op.parts(g)):
            optimizer.state[p]={k:v.detach().to(p.device).clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v)
                for k,v in old['state'][pid].items()}
            p.grad=part.detach().clone()
        optimizer.step()
        value=torch.cat([(p.detach()-origin.detach()).flatten() for p,origin in zip(params,op.params)])
        del optimizer,params;return value
    history=[r['surface'] for r in saved['records'] if 36<r['step']<=216 and 'surface' in r]
    accepted=sum(r['coordinate_norm']>0 for r in history)
    fit_failed=sum(r['fit_ce_gain']<=0 or not r['coordinate_iterations'] for r in history)
    report=dict(checkpoint_step=len(saved['records']),historical_source_commit='6c78fee73670f651b06495d5da305955fe9495e8',
        protocol='Three paired replay contexts at the identical immutable step-432 state, 24 fit / up to24 disjoint held training images; all arms share actual AdamW norm radius',
        state_sha256=base_hash,provenance=provenance,reference_lr=saved['optimizer']['param_groups'][0]['lr'],
        replay_id_namespace='EpochData index within the seed-0 training subset, not raw ImageFolder index',
        historical_37_216=dict(steps=len(history),accepted=accepted,rejected=len(history)-accepted,
            fit_no_gain_or_no_iterations=fit_failed,positive_fit_but_unwritten=sum(r['fit_ce_gain']>0 and r['coordinate_iterations']>0 and r['coordinate_norm']==0 for r in history),
            gate_vs_fp32_split='Unidentifiable from historical records; original states/virtual held CE absent'),
        adamw_is_observed_reference_not_upper_bound=True,training_trajectory_modified=False,
        fiber_algorithm_implemented=False,cases=[])
    report['diagnostic_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    partial=args.output/'partial_results.json'
    if partial.exists():
        prior=json.loads(partial.read_text())
        if prior['state_sha256']!=base_hash or prior['diagnostic_source_sha256']!=report['diagnostic_source_sha256']:
            raise ValueError('Partial diagnostic identity mismatch')
        report=prior
    gramF=(frames.flatten(0,1)@frames.flatten(0,1).T).double().cpu().numpy()
    for case,(fit_index,held_index) in enumerate(((0,1),(3,4),(6,7))):
        if any(row['case']==case for row in report['cases']):continue
        started=time.monotonic();xi,yi,ids=replay[fit_index];xh,yh,held_ids=replay[held_index]
        keep=[i for i,v in enumerate(held_ids.tolist()) if v not in set(ids.tolist())]
        if not keep:raise ValueError('No disjoint held batch')
        x,y=xi.cuda(),yi.cuda();hx,hy=xh[keep].cuda(),yh[keep].cuda();metric=x
        before_counts=dict(counts)
        with torch.no_grad(),probe_mode(model):
            before,g=gradient(x,y);held_before=ce(hx,hy)
            clipped=g*min(1.,1./max(float(g.norm()),1e-30));raw=adamw(clipped);radius=float(raw.norm())
            if radius<=0:raise ValueError('Zero FP32 AdamW reference displacement')
            T=surface.chart_tangent(state['z'],state['q']);C=orthogonal_coefficients(T,gramF)
            U=torch.as_tensor(C.T,device='cuda',dtype=frames.dtype)@frames.flatten(0,1);rank=len(U)
            projected=U.T@(U@raw);gr=-g-U.T@(U@(-g));ur=raw-projected
            norm_gr,norm_ur=float(gr.norm()),float(ur.norm())
            gr=gr/gr.norm().clamp_min(1e-30);ur=ur/ur.norm().clamp_min(1e-30)
            bank=torch.cat((U,gr[None],ur[None]),0);del gr,ur
            gram=(bank@bank.T).double().cpu().numpy();baseC=np.eye(rank+2)[:,:rank]
            R=np.stack([response(metric,v) for v in bank],1);rawR=response(metric,raw)
            smallg=(bank@g).double().cpu().numpy();weak=int(np.argmin(smallg[:rank]**2))
            def vector(c):return torch.as_tensor(c,device='cuda',dtype=bank.dtype)@bank
            def stats(delta):
                return dict(fit_CE=ce(x,y,delta),held_CE=ce(hx,hy,delta),parameter_norm=float(delta.norm()))
            def linear(Q):
                p=Q.T@gram@Q;rr=R@Q;fisher=rr.T@rr
                mu=max(1e-8,.001*np.trace(fisher)/max(np.trace(p),1e-12));inverse=np.linalg.pinv(fisher+mu*p,rcond=1e-8,hermitian=True)
                a=np.zeros(Q.shape[1]);value=before;grad=g;rounds=0
                for iteration in range(4):
                    b=Q.T@(bank@grad).double().cpu().numpy();d=-inverse@b
                    d=trust(d,p,radius)
                    for fraction in (1.,.5,.25):
                        trial=trust(a+fraction*d,p,radius);delta=vector(Q@trial)
                        score=ce(x,y,delta)
                        if float(delta.norm())>radius*(1+2e-6):continue
                        if score<value-1e-10:
                            a=trial;value=score;rounds+=1
                            if iteration<3:value,grad=gradient(x,y,delta)
                            break
                    else:break
                delta=vector(Q@a);return dict(**stats(delta),rounds=rounds,rank=Q.shape[1],coefficients=a.tolist())
            row=dict(case=case,fit_replay_index=fit_index,held_replay_index=held_index,fit_ids=ids.tolist(),held_ids=held_ids[keep].tolist(),
                radius=radius,base_fit_CE=before,base_held_CE=held_before,tangent_rank=rank,group_count=len(state['z']),
                parameter_coverage=float(projected.square().sum()/raw.square().sum()),
                functional_best_span_coverage=float(np.linalg.norm((R[:,:rank]@np.linalg.pinv(R[:,:rank],rcond=1e-8))@rawR)**2/max(rawR@rawR,1e-30)),
                projected_JVP_energy_ratio=float(np.linalg.norm(R[:,:rank]@(U@raw).double().cpu().numpy())**2/max(rawR@rawR,1e-30)),
                adamw=stats(raw),adamw_projection=stats(projected),
                adamw_projection_norm_matched=stats(projected*(radius/max(float(projected.norm()),1e-30))),
                linear_old_tangent=linear(baseC),residual_norms=dict(negative_gradient=norm_gr,adamw=norm_ur))
            chart=surface.Chart(op,frames,state['z'],state['q'],state['scale'],x)
            a,w,fit_after,fit_before,iterations,_=surface.optimize_coordinates(op,x,y,chart,radius)
            delta=chart.vector(w);virtual_held=ce(hx,hy,delta)
            trial=[d+(torch.as_tensor(w[3*i:3*i+3],device='cuda',dtype=frames.dtype)@frame).cpu()
                   for i,(d,frame) in enumerate(zip(state['cumulative'],frames))]
            rebuilt=surface.canonical_total(dict(state,cumulative=trial)).float().cuda()
            canonical_held=ce(hx,hy,absolute=rebuilt)
            actual=torch.cat([p.detach().flatten() for p in op.params]);virtual=actual+delta
            row['quadratic_old_solver']=dict(fit_CE=fit_after,held_virtual_CE=virtual_held,held_rebuilt_CE=canonical_held,
                rounds=iterations,parameter_norm=float(delta.norm()),fp32_reconstruction_difference_norm=float((rebuilt-virtual).norm()),
                reason=refusal(fit_before-fit_after,iterations,held_before,virtual_held,canonical_held))
            # Exactly the same coordinates for the curvature comparison; apply one
            # common shrink if either the linear or quadratic image exceeds radius.
            factor=1.
            for _ in range(40):
                linear_delta=chart.vector(state['scale']*T@(a*factor))
                curved_delta=chart.vector(chart.coefficients(a*factor))
                if max(float(linear_delta.norm()),float(curved_delta.norm()))<=radius*(1+2e-6):break
                factor*=.5
            row['same_coordinates_curvature']=dict(common_coordinate_factor=factor,linear=stats(linear_delta),quadratic=stats(curved_delta))
            row['rotations']={}
            for name,index,size in [('negative_gradient',rank,norm_gr),('adamw',rank+1,norm_ur)]:
                trials=[]
                for degrees in (-10.,10.):
                    Q=baseC.copy()
                    if size>1e-10:
                        Q[:,weak]*=math.cos(math.radians(degrees));Q[index,weak]=math.sin(math.radians(degrees))
                    Q=orthogonal_coefficients(Q,gram)
                    outcome=linear(Q);outcome['degrees']=degrees
                    outcome['gradient_coverage']=float(np.linalg.norm(Q.T@smallg)**2/max(float(g.square().sum()),1e-30))
                    outcome['adamw_coverage']=float(np.linalg.norm(Q.T@(bank@raw).double().cpu().numpy())**2/max(float(raw.square().sum()),1e-30))
                    rr=R@Q;prediction=rr@np.linalg.lstsq(rr,rawR,rcond=1e-8)[0]
                    outcome['best_functional_span_coverage']=float(prediction@prediction/max(rawR@rawR,1e-30))
                    trials.append(outcome)
                row['rotations'][name]=dict(trials=trials,selected_by_fit_only=min(trials,key=lambda t:t['fit_CE']))
            Q=baseC.copy()
            if norm_gr>1e-10:Q[:,weak]=0.;Q[rank,weak]=1.
            Q=orthogonal_coefficients(Q,gram)
            row['same_rank_relearned']=dict(**linear(Q),method='Replace weakest old tangent axis with current fit-gradient residual; held data never used to construct/select basis',matched_rank=Q.shape[1]==rank)
            row['linear_gradient_coverage']=float(np.linalg.norm(smallg[:rank])**2/max(float(g.square().sum()),1e-30))
            tilt=surface.chart_tangent(state['z'],state['q'])[2::3,:]
            row['tangent_tilt_coefficients']=tilt.tolist()
            row['seconds']=time.monotonic()-started;row['logical_calls_delta']={k:counts[k]-before_counts[k] for k in counts}
            row['source_solver_cost_note']='Wrapper counters include frozen Chart/optimizer calls; equal round limits and radius do not imply equal realized calls or FLOPs'
            report['cases'].append(row)
            report['peak_gpu_allocated_bytes']=max(report.get('peak_gpu_allocated_bytes',0),torch.cuda.max_memory_allocated())
            write(args.output/'progress.json',dict(stage='case_complete',case=case,completed=len(report['cases']),seconds=row['seconds']))
            write(args.output/'partial_results.json',report)
            print(json.dumps(dict(case=case,radius=radius,parameter_coverage=row['parameter_coverage'],old_fit_gain=before-fit_after,
                old_gate_reason=row['quadratic_old_solver']['reason'],seconds=row['seconds'])),flush=True)
            del bank,U,g,raw,projected,R,chart,delta,trial,rebuilt,actual,virtual,linear_delta,curved_delta;gc.collect();torch.cuda.empty_cache()
    final=torch.cat([p.detach().flatten().cpu() for p in op.params])
    if hashlib.sha256(final.numpy().data).hexdigest()!=base_hash:raise RuntimeError('Counterfactuals mutated actual model')
    report.update(state_after_sha256=base_hash,actual_model_unchanged=True,
        wrapper_calls={k:sum(row['logical_calls_delta'][k] for row in report['cases']) for k in counts},
        maximum_process_fraction=.35,
        resume_scope='Completed fixed-state cases retained; an interrupted uncommitted case is replayed and its wasted calls are not included')
    write(args.output/'results.json',report)
    write(args.output/'progress.json',dict(stage='complete',cases=len(report['cases'])))

if __name__=='__main__':main()
