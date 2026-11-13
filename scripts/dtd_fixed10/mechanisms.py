"""A/B/C fixed-space interventions. No optimizer or committed model updates."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def write(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;out.mkdir(exist_ok=True,parents=True)
    root=Path('/home/shangxueyuan/merging/tsv')
    sys.path.insert(0,str(root/'scripts/fisher_rao_online_v5_98f36a1'))
    from clip_runtime import ClipTask
    from checkpointing import load_model
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    raw=ClipTask(root,out,128,32,task='DTD')
    torch.set_grad_enabled(False)
    asset=Path('/media/shangxueyuan/data1/DTD_fixed10_20261113')
    fixed=torch.load(asset/'fixed_basis.pt',map_location='cpu',weights_only=True,mmap=True)
    assert fixed['identity']['base']==raw.rt.input['base_sha256']
    assert fixed['identity']['head']==raw.rt.input['head_sha256']
    B=fixed['B_rows'].float().cuda();assert tuple(B.shape)==(10,84934656)
    gram=(B@B.T).double();assert float((gram-torch.eye(10,dtype=torch.double,device='cuda')).norm())<1e-4
    original_X=list(raw.rt.split['X']);check_ids=list(raw.rt.split['Xprime'])
    probe=check_ids[:128];raw.rt.split={**raw.rt.split,'X':probe}
    generator=torch.Generator().manual_seed(20261113)
    train=list(raw.rt.split['train']);order=torch.randperm(len(train),generator=generator).tolist()
    fits=[[train[i] for i in order[j*64:(j+1)*64]] for j in range(2)]
    assert not set(sum(fits,[])) & (set(check_ids)|set(original_X))
    radii=[.001,.003,.01];states=[0,32,64,128]
    config=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        basis_source=str(asset/'fixed_basis.pt'),basis_identity=fixed['identity'],
        states=states,radii=radii,A_probe_ids=probe,B_check_ids=check_ids,B_gradient_ids=fits,
        training=False,basis_updated=False,coefficients_optimized=False,
        A_replicates=4,A_signs=[-1,1],B_gradient_batches=2,B_equal_parameter_norm=True,
        ambient_scope='All 84934656 coordinates of the original48 visual matrices; not entire CLIP',
        C_response_comparison='Extrinsic sqrt-probability coordinates on identical images; not parallel transport',
        official_test_opened=False,control_caveat='Isotropic ambient controls are not sensitivity-matched; no PCA/random-subspace superiority claim')
    configfile=out/'config.json'
    if configfile.exists():assert json.loads(configfile.read_text())==config
    else:write(configfile,config)
    inputs=Path('/media/shangxueyuan/data1/fisher_rao_online_v5_DTD_medium_seed11_98f36a1_20261112/checkpoints')
    paths={int(p.name.split('_')[1]):p for p in inputs.glob('*.pt')}
    def point(step):
        state=torch.load(paths[step],map_location='cpu',weights_only=True,mmap=True)
        assert state['identity']['base_sha256']==raw.rt.input['base_sha256']
        return load_model(state['state'],raw).parameters().detach()
    def evaluate(theta,split,n=None,old=None):
        zs=[];ys=[]
        for x,y in raw.rt.batches(split,batch=32,n=n):
            zs.append(raw.rt.predict(theta,x).double().cpu());ys.append(y.cpu())
        z=torch.cat(zs);y=torch.cat(ys);ce=F.cross_entropy(z,y,reduction='none')
        result=dict(CE=float(ce.mean()),accuracy=float((z.argmax(1)==y).double().mean()),n=len(y))
        if old is not None:
            result['KL']=float((old.softmax(1)*(old.log_softmax(1)-z.log_softmax(1))).sum(1).mean())
            result['CE_delta']=float((ce-F.cross_entropy(old,y,reduction='none')).mean())
        return result,z
    def gradient(theta,ids):
        total=torch.zeros_like(theta);loss=0.
        for start in range(0,len(ids),16):
            items=[raw.rt.raw[i] for i in ids[start:start+16]]
            x=torch.stack([v for v,_ in items]).cuda();y=torch.tensor([int(y) for _,y in items],device='cuda')
            with torch.enable_grad():g,ce=raw.rt.grad(theta,x,y)
            total.add_(g,alpha=len(y)/len(ids));loss+=ce*len(y)/len(ids)
        return total,loss
    def ambient_direction(rep):
        gen=torch.Generator(device='cuda').manual_seed(90811+rep)
        v=torch.randn(B.shape[1],device='cuda',generator=gen);v/=v.norm()
        for _ in range(2):v-=B.T@(B@v)
        v/=v.norm();assert float((B@v).norm())<1e-5
        return v
    all_results=[]
    for step in states:
        target=out/f'state_{step:03d}.json';responses=out/f'response_{step:03d}.npz'
        if target.exists() and responses.exists():
            all_results.append(json.loads(target.read_text()));continue
        theta=point(step);write(out/'status.json',dict(state=step,stage='responses'))
        scoreA,oldA=evaluate(theta,'X',128);scoreB,oldB=evaluate(theta,'Xprime',512)
        columns=[]
        for j in range(10):
            columns.append(raw.rt.jvp(theta,B[j]).cpu().double())
            write(out/'status.json',dict(state=step,stage='JVP',column=j+1))
        R=torch.stack(columns,1);G=R.T@R
        eigen=torch.linalg.eigvalsh(G).clamp_min(0);p=eigen/eigen.sum()
        C=dict(G=G.tolist(),eigenvalues=eigen.tolist(),trace=float(eigen.sum()),
               participation_rank=float(eigen.sum().square()/eigen.square().sum()),
               entropy_rank=float(torch.exp(-(p*p.clamp_min(1e-300).log()).sum())),
               numerical_rank=int((eigen>eigen[-1]*1e-6).sum()),
               numerical_rank_relative_threshold=1e-6)
        # A is completed first, then B; the expensive ambient invariance audit is last.
        A=[]
        for rep in range(4):
            coeff=torch.randn(10,dtype=torch.double,generator=torch.Generator().manual_seed(5021+rep));coeff/=coeff.norm()
            for kind in ('inside','ambient_outside'):
                v=B.T@coeff.float().cuda() if kind=='inside' else ambient_direction(rep)
                v/=v.norm()
                response=raw.rt.jvp(theta,v).cpu().double()
                energy=float(response.square().sum())
                for radius in radii:
                    for sign in (-1,1):
                        delta=sign*radius*v;actual=theta+delta
                        score,_=evaluate(actual,'X',128,oldA)
                        A.append(dict(rep=rep,kind=kind,radius=radius,sign=sign,**score,
                            predicted_KL=.5*radius**2*energy,
                            realized_parameter_norm=float((actual-theta).norm()),
                            fp32_addition_error=float(((actual-theta)-delta).norm())))
                del v
                write(out/'status.json',dict(state=step,stage='A',rep=rep,kind=kind))
        write(out/f'A_{step:03d}.json',dict(state=step,rows=A))
        resultsB=[]
        for batch,ids in enumerate(fits):
            g,fitloss=gradient(theta,ids);projected=B.T@(B@g);remainder=g-projected
            norms=dict(full=float(g.norm()),inside=float(projected.norm()),outside=float(remainder.norm()))
            for kind,direction in [('full',g),('inside',projected),('outside',remainder),('random',ambient_direction(batch))]:
                v=direction/direction.norm()
                # The random control's sign uses only the gradient-fit batch.
                if kind=='random' and float(g@v)<0:v=-v
                for radius in radii:
                    score,_=evaluate(theta-radius*v,'Xprime',512,oldB)
                    resultsB.append(dict(batch=batch,kind=kind,radius=radius,**score,
                        gradient_norms=norms,fit_gradient_CE=fitloss,
                        fit_first_order_delta=-radius*float(g@v)))
            del g,projected,remainder,direction,v
            write(out/'status.json',dict(state=step,stage='B',gradient_batch=batch+1))
        write(out/f'B_{step:03d}.json',dict(state=step,rows=resultsB))
        energy_all=0.;energy_leak=0.;vjp_consistency=[]
        for j in range(10):
            with torch.enable_grad():fb=raw.rt.vjp(theta,R[:,j].to(device='cuda',dtype=theta.dtype))
            coordinates=B@fb;residual=fb-B.T@coordinates
            energy_all+=float(fb.double().square().sum());energy_leak+=float(residual.double().square().sum())
            vjp_consistency.append(float((coordinates.double().cpu()-G[:,j]).norm()/G[:,j].norm().clamp_min(1e-30)))
            del fb,residual
            write(out/'status.json',dict(state=step,stage='C_ambient_Fisher',column=j+1))
        C.update(ambient_FB_frobenius_norm=energy_all**.5,
                 ambient_leakage_frobenius_ratio=(energy_leak/energy_all)**.5,
                 vjp_vs_response_Gram_relative_errors=vjp_consistency)
        result=dict(state=step,baseline_A=scoreA,baseline_B=scoreB,A=A,B=resultsB,C=C)
        np.savez(responses,R=R.numpy(),G=G.numpy())
        write(target,result);all_results.append(result)
        print(json.dumps(dict(state=step,CE=scoreB['CE'],effective_rank=C['participation_rank'],
            leakage=C['ambient_leakage_frobenius_ratio'])),flush=True)
        del theta,R,columns;torch.cuda.empty_cache()
    relations=[]
    for old,new in zip(all_results,all_results[1:]):
        Ra=torch.from_numpy(np.load(out/f"response_{old['state']:03d}.npz")['R'])
        Rb=torch.from_numpy(np.load(out/f"response_{new['state']:03d}.npz")['R'])
        def span(R):
            U,s,_=torch.linalg.svd(R,full_matrices=False)
            return U[:,s>s[0]*1e-6]
        Ua,Ub=span(Ra),span(Rb);cos=torch.linalg.svdvals(Ua.T@Ub)
        Ga,Gb=Ra.T@Ra,Rb.T@Rb
        relations.append(dict(states=[old['state'],new['state']],extrinsic_principal_cosines=cos.tolist(),
            normalized_metric_distance=float((Ga/Ga.norm()-Gb/Gb.norm()).norm()),
            next_CE_change=new['baseline_B']['CE']-old['baseline_B']['CE']))
    report=dict(config=config,states=all_results,relations=relations,
        caveats=['Only four states and three transitions; no predictive-regression or causality claim.',
                 'No official test data used. Xprime was used in earlier evaluations, never basis selection.',
                 'Isotropic random ambient directions are not matched for Fisher sensitivity.',
                 'FP32 finite-perturbation rounding is reported; tiny KL values may reach a numerical floor.',
                 'State checkpoints and B depend on the old complete trajectory; not prospective discovery.'])
    write(out/'report.json',report);write(out/'status.json',dict(stage='complete',states=states,training=False))


if __name__=='__main__':main()
