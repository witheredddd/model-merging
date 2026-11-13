"""Single-task raw-coordinate control at fixed theta128 and fixed Fisher10.

Candidate masks are hypotheses. Only real held-out interventions evaluate control.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)


def minimum_norm_control(H,desired,cap=.01,inverse_information=None):
    if inverse_information is None:
        inverse_information=torch.linalg.pinv(H.T@H,hermitian=True,rtol=1e-8)
    delta=H@(inverse_information@desired)
    norm=float(delta.norm());scale=min(1.,cap/max(norm,1e-30))
    return delta*scale,norm,scale


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    out.mkdir(exist_ok=True,parents=True);private=out/'private';private.mkdir(exist_ok=True)
    root=Path('/home/shangxueyuan/merging/tsv');sys.path.insert(0,str(root/'scripts/fisher_rao_online_v5_98f36a1'))
    from clip_runtime import ClipTask
    from checkpointing import load_model
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    raw=ClipTask(root,out,128,32,task='DTD');torch.set_grad_enabled(False)
    support=Path('/media/shangxueyuan/data1/DTD_fixed10_support_20261114')
    assert (support/'probe_0/report.json').exists(),'Wait until A probe0 is complete'
    support_config=json.loads((support/'config.json').read_text());groups=support_config['groups']
    old=Path('/media/shangxueyuan/data1/DTD_fixed10_20261113')
    fixed=torch.load(old/'fixed_basis.pt',map_location='cpu',mmap=True,weights_only=True)
    assert fixed['identity']['base']==raw.rt.input['base_sha256'] and fixed['identity']['head']==raw.rt.input['head_sha256']
    B=fixed['B_rows'].float().cuda();D=B.shape[1];budget=int(D*.01)
    checkpoints=Path('/media/shangxueyuan/data1/fisher_rao_online_v5_DTD_medium_seed11_98f36a1_20261112/checkpoints')
    saved=torch.load(next(checkpoints.glob('step_000128_*.pt')),map_location='cpu',mmap=True,weights_only=True)
    theta=load_model(saved['state'],raw).parameters().detach();theta_version=theta._version
    fit_ids=list(raw.rt.split['Xprime'][:128]);held_ids=list(raw.rt.split['Xprime'][256:512])
    assert fit_ids==support_config['probe_ids'][0] and not set(held_ids)&set(sum(support_config['probe_ids'],[]))
    historical=json.loads(Path('/media/shangxueyuan/data1/DTD_fixed10_mechanisms_20261113/config.json').read_text())
    gradient_ids=historical['B_gradient_ids'];assert not set(sum(gradient_ids,[]))&set(held_ids+fit_ids)
    config=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),state=128,
        basis_sha256=support_config['basis_sha256'],base_sha256=raw.rt.input['base_sha256'],head_sha256=raw.rt.input['head_sha256'],
        budget=budget,budget_fraction=.01,selection_probe_ids=fit_ids,held_ids=held_ids,gradient_ids=gradient_ids,
        targets='Four seeded unit coefficient directions with both signs, radius .001 and .003',
        target_seeds=[5021,5022,5023,5024],control_parameter_norm_cap=.01,
        solver='Two controls: selected-response minimum parameter norm; full-response least squares in the same masked-access span with 1e-6 scale-relative parameter ridge. Both capped afterward.',
        candidate='16-batch logdet-gain heuristic; not exact D-optimal subset and not causal attribution',
        candidate_ridge='0.01*trace(all_access_information)/10*budget/D',
        random_seeds=[14,15,16],position_seeds=[101,102,103],matrix_matched_seeds=[41,42,43],
        task_radii=[.001,.003,.01],training=False,basis_updated=False,fusion_model_used=False,official_test_opened=False)
    cfg=out/'config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:write(cfg,config)
    Rfit=torch.from_numpy(np.load(support/'private/probe_0_R.npz')['R']);C=Rfit.T@Rfit
    eig,V=torch.linalg.eigh(C);assert float(eig[0])>float(eig[-1])*1e-10
    inverse_root=(V/eig.sqrt())@V.T;square_root=(V*eig.sqrt())@V.T
    FB=torch.load(support/'private/probe_0_FB.pt',map_location='cpu',mmap=True,weights_only=True)['FB_rows']
    A=FB.T.cuda()@inverse_root.cuda();del FB
    assert A.shape==(D,10)
    masks_file=private/'masks.pt'
    if masks_file.exists():masks=torch.load(masks_file,map_location='cpu',weights_only=True)
    else:
        masks={};information=A.T@A
        ridge=float(information.trace())/10*budget/D*.01
        precision=torch.eye(10,dtype=torch.double,device='cuda')*ridge
        selected=torch.zeros(D,dtype=torch.bool,device='cuda');indices=[];gain=torch.empty(D,device='cuda')
        for batch in range(16):
            inverse=torch.linalg.inv(precision)
            for start in range(0,D,1048576):
                end=min(start+1048576,D);block=A[start:end]
                gain[start:end]=(block*(block@inverse)).sum(1).float()
            gain[selected]=-torch.inf
            count=budget//16+(1 if batch<budget%16 else 0)
            chosen=torch.topk(gain,count,sorted=False).indices;selected[chosen]=True;indices.append(chosen.cpu())
            block=A[chosen];precision+=block.T@block
            write(out/'status.json',dict(stage='candidate_selection',batch=batch+1,total=16))
        masks['joint_logdet']=torch.cat(indices)
        for name,mapname in [('top_g','g'),('top_leverage','ellB')]:
            values=np.load(support/f'private/p0_{mapname}.npy',mmap_mode='r')
            masks[name]=torch.from_numpy(np.argpartition(values,-budget)[-budget:].copy())
        for seed in config['random_seeds']:
            indices=np.random.default_rng(seed).choice(D,budget,replace=False)
            masks[f'random_{seed}']=torch.from_numpy(indices)
        for seed in config['position_seeds']:
            rng=np.random.default_rng(seed);parts=[];remaining=budget
            for gi in rng.permutation(len(groups)):
                group=groups[gi];count=min(remaining,group['numel'])
                parts.append(torch.arange(group['start'],group['start']+count));remaining-=count
                if not remaining:break
            masks[f'position_{seed}']=torch.cat(parts)
        candidate=masks['joint_logdet'].numpy()
        counts=[int(((candidate>=g['start'])&(candidate<g['end'])).sum()) for g in groups]
        for seed in config['matrix_matched_seeds']:
            rng=np.random.default_rng(seed);parts=[]
            for group,count in zip(groups,counts):
                if count:parts.append(rng.choice(group['numel'],count,replace=False)+group['start'])
            masks[f'matrix_matched_{seed}']=torch.from_numpy(np.concatenate(parts))
        torch.save(masks,masks_file)
        del precision,selected,indices,gain,information
    assert len(masks)==12
    for mask in masks.values():assert mask.numel()==budget and mask.unique().numel()==budget
    # Fit and held sets use the same frozen model; held samples did not select masks.
    datasets={}
    for name,ids in [('fit',fit_ids),('held',held_ids)]:
        raw.rt.split={**raw.rt.split,'X':ids};raw.rt.n=len(ids)
        batches=list(raw.rt.batches('X',batch=32,n=len(ids)))
        datasets[name]=[(x.cpu(),y.cpu()) for x,y in batches]
    def predict(point,name):
        zs=[];ys=[]
        for x,y in datasets[name]:
            zs.append(raw.rt.predict(point,x.cuda()).double().cpu());ys.append(y)
        return torch.cat(zs),torch.cat(ys)
    base={name:predict(theta,name) for name in datasets}
    R={'fit':Rfit}
    raw.rt.split={**raw.rt.split,'X':held_ids};raw.rt.n=len(held_ids)
    for j in range(10):
        response=raw.rt.jvp(theta,B[j]).cpu().double()
        if j==0:held_columns=[]
        held_columns.append(response)
        write(out/'status.json',dict(stage='held_geometry',column=j+1))
    R['held']=torch.stack(held_columns,1)
    inverses={name:torch.linalg.pinv(r.T@r,hermitian=True,rtol=1e-10) for name,r in R.items()}
    def embedding(logits):return 2*logits.softmax(1).sqrt().flatten()/np.sqrt(len(logits))
    def response_metrics(z,name,target=None):
        baseline,y=base[name];delta=embedding(z)-embedding(baseline);r=R[name]
        projection=r@(inverses[name]@(r.T@delta));outside=delta-projection
        ce=torch.nn.functional.cross_entropy(z,y,reduction='none')
        result=dict(CE=float(ce.mean()),CE_delta=float((ce-torch.nn.functional.cross_entropy(baseline,y,reduction='none')).mean()),
            accuracy=float((z.argmax(1)==y).double().mean()),
            KL_from_base=float((baseline.softmax(1)*(baseline.log_softmax(1)-z.log_softmax(1))).sum(1).mean()),
            selected_response_norm=float(projection.norm()),outside_response_norm=float(outside.norm()))
        if target is not None:
            wanted=embedding(target)-embedding(baseline);wanted_projection=r@(inverses[name]@(r.T@wanted))
            result.update(selected_target_relative_error=float((projection-wanted_projection).norm()/wanted_projection.norm()),
                full_target_relative_error=float((delta-wanted).norm()/wanted.norm()),
                outside_over_target_norm=float(outside.norm()/wanted.norm()),
                KL_target_to_control=float((target.softmax(1)*(target.log_softmax(1)-z.log_softmax(1))).sum(1).mean()))
        return result
    targets=[]
    for seed in config['target_seeds']:
        c=torch.randn(10,dtype=torch.double,generator=torch.Generator().manual_seed(seed));c/=c.norm()
        for radius in (.001,.003):
            for sign in (-1,1):
                coef=c*radius*sign;point=theta+coef.float().cuda()@B
                targets.append(dict(seed=seed,radius=radius,sign=sign,coefficient=coef,
                    logits={name:predict(point,name)[0] for name in datasets}))
    gradients=[]
    for i,ids in enumerate(gradient_ids):
        path=private/f'gradient_{i}.pt'
        if path.exists():g=torch.load(path,map_location='cpu',weights_only=True).cuda()
        else:
            g=torch.zeros_like(theta)
            for start in range(0,len(ids),16):
                items=[raw.rt.raw[j] for j in ids[start:start+16]]
                x=torch.stack([x for x,_ in items]).cuda();y=torch.tensor([int(y) for _,y in items],device='cuda')
                with torch.enable_grad():part,_=raw.rt.grad(theta,x,y)
                g.add_(part,alpha=len(y)/len(ids))
            torch.save(g.cpu(),path)
        gradients.append(g)
    results=[];control_dir=out/'controls';control_dir.mkdir(exist_ok=True)
    for method,mask in masks.items():
        path=control_dir/f'{method}.json'
        if path.exists():results.append(json.loads(path.read_text()));continue
        indices=mask.cuda();H=A[indices];K=H.T@H
        pinv=torch.linalg.pinv(K,hermitian=True,rtol=1e-8)
        values=torch.linalg.eigvalsh(K);realized=[];task=[]
        # A second controller penalizes the entire response, including outside B's image.
        # This is still a single algebraic intervention, not iterative model training.
        pathT=private/f'full_response_{method}.npz'
        if pathT.exists():T=torch.from_numpy(np.load(pathT)['T'])
        else:
            raw.rt.split={**raw.rt.split,'X':fit_ids};raw.rt.n=len(fit_ids)
            columns=[];direction=torch.zeros_like(theta)
            for j in range(10):
                direction.zero_();direction[indices]=H[:,j].float()
                columns.append(raw.rt.jvp(theta,direction).cpu().double())
                write(out/'status.json',dict(stage='full_response_controller',method=method,column=j+1))
            T=torch.stack(columns,1);np.savez(pathT,T=T.numpy());del columns,direction
        normal=T.T@T;ridge=1e-6*float(normal.trace())/float(K.trace())
        full_inverse=torch.linalg.pinv(normal+ridge*K.cpu(),hermitian=True,rtol=1e-8)
        for i,target in enumerate(targets):
            desired=square_root.cuda()@target['coefficient'].cuda()
            for controller in ('selected_min_norm','full_response_fit'):
                if controller=='selected_min_norm':
                    delta,uncapped,scale=minimum_norm_control(H,desired,inverse_information=pinv)
                else:
                    wanted=embedding(target['logits']['fit'])-embedding(base['fit'][0])
                    coefficient=full_inverse@(T.T@wanted)
                    delta=H@coefficient.cuda();uncapped=float(delta.norm());scale=min(1.,.01/max(uncapped,1e-30));delta*=scale
                point=theta.clone();point[indices]+=delta.float()
                metrics={name:response_metrics(predict(point,name)[0],name,target['logits'][name]) for name in datasets}
                realized.append(dict(controller=controller,seed=target['seed'],radius=target['radius'],sign=target['sign'],
                    uncapped_parameter_norm=uncapped,scale=scale,applied_parameter_norm=float((point-theta).norm()),
                    linear_selected_constraint_relative_error=float((H.T@delta-desired).norm()/desired.norm()),metrics=metrics))
            write(out/'status.json',dict(stage='response_control',method=method,target=i+1,total=16))
        for batch,g in enumerate(gradients):
            restricted=g[indices];restricted=restricted/restricted.norm()
            for radius in config['task_radii']:
                point=theta.clone();point[indices]-=radius*restricted
                metrics={name:response_metrics(predict(point,name)[0],name) for name in datasets}
                task.append(dict(batch=batch,radius=radius,applied_parameter_norm=float((point-theta).norm()),metrics=metrics))
        result=dict(method=method,budget=budget,information_eigenvalues=values.cpu().tolist(),
            response_control=realized,task_control=task,
            matrix_counts=[int(((mask>=g['start'])&(mask<g['end'])).sum()) for g in groups])
        write(path,result);results.append(result)
        print(json.dumps(dict(method=method,held_task_deltas=[t['metrics']['held']['CE_delta'] for t in task])),flush=True)
        del H,indices,point
    assert theta._version==theta_version
    report=dict(config=config,methods=results,
        target_baselines=[dict(seed=t['seed'],radius=t['radius'],sign=t['sign'],
            metrics={name:response_metrics(t['logits'][name],name) for name in datasets}) for t in targets],
        caveats=['Candidate jointly covers selected responses on fit samples, but control must generalize and avoid outside spill.',
                 'Linear minimum-norm solve can match selected response while perturbing other functions; real forward metrics are primary.',
                 'One coordinate budget, fixed anchor and small local radii; not a training or high-performance capacity result.',
                 'Position controls use randomly ordered complete matrices with a truncated last matrix for exact budget.',
                 'No cross-task conflict, EC search or fusion model used.'])
    write(out/'report.json',report);write(out/'status.json',dict(stage='complete',methods=len(results),training=False))


if __name__=='__main__':main()
