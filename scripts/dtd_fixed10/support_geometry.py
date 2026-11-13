"""Frozen-space coordinate participation/access/alignment and matrix coupling.

No optimization, no support recovery, and no claim of causal formation.
"""
import argparse,hashlib,json,re,sys
from pathlib import Path
import numpy as np
import torch


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def concentration(values):
    values=np.abs(np.asarray(values));total=float(values.sum(dtype=np.float64))
    fractions=[.00001,.0001,.001,.01,.1]
    ks=[max(1,int(len(values)*f)) for f in fractions]
    partitioned=np.partition(values,[len(values)-k for k in ks])
    return dict(absolute_sum=total,signed_sum=float(np.asarray(values).sum(dtype=np.float64)),
        top_fraction_mass={str(f):float(partitioned[-k:].sum(dtype=np.float64))/total if total else 0.
                           for f,k in zip(fractions,ks)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    out=parser.parse_args().output;out.mkdir(exist_ok=True,parents=True)
    root=Path('/home/shangxueyuan/merging/tsv');source=root/'scripts/fisher_rao_online_v5_98f36a1'
    sys.path.insert(0,str(source));from clip_runtime import ClipTask
    from checkpointing import load_model
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    raw=ClipTask(root,out,128,32,task='DTD');torch.set_grad_enabled(False)
    old=Path('/media/shangxueyuan/data1/DTD_fixed10_20261113')
    basis_path=old/'fixed_basis.pt';basis_digest=sha(basis_path)
    basis=torch.load(basis_path,map_location='cpu',mmap=True,weights_only=True)
    assert basis['identity']['base']==raw.rt.input['base_sha256'] and basis['identity']['head']==raw.rt.input['head_sha256']
    Bcpu=basis['B_rows'];B=Bcpu.float().cuda();D=B.shape[1]
    qdir=Path('/media/shangxueyuan/data1/DTD_fisher_blocks_expanded_20261113')
    Q=torch.load(qdir/'common_basis.pt',map_location='cpu',mmap=True,weights_only=True)['Q_rows']
    checkpoints=Path('/media/shangxueyuan/data1/fisher_rao_online_v5_DTD_medium_seed11_98f36a1_20261112/checkpoints')
    checkpoint=next(checkpoints.glob('step_000128_*.pt'))
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
    assert saved['identity']['base_sha256']==raw.rt.input['base_sha256']
    theta=load_model(saved['state'],raw).parameters().detach()
    offsets=[];offset=0
    for name,param in raw.rt.found:
        offsets.append(dict(name=name,start=offset,end=offset+param.numel(),numel=param.numel(),
                            layer=int(re.search(r'resblocks\.(\d+)',name).group(1))))
        offset+=param.numel()
    assert offset==D
    expert=root/'results/EXPERTS8_FD_REVIEWED_20261110/DTD/adaptive_dynamics/epoch08.pt'
    expert_meta=json.loads(expert.with_name('complete.json').read_text())
    expert_digest=sha(expert);assert expert_digest==expert_meta['checkpoint_sha256']
    assert expert_meta['provenance']['base_sha256']==raw.rt.input['base_sha256']
    assert expert_meta['provenance']['head_sha256']==raw.rt.input['head_sha256']
    external=torch.load(expert,map_location='cpu',mmap=True,weights_only=True)['visual']
    expert_theta=torch.cat([external[n.removeprefix('model.visual.')].flatten() for n,_ in raw.rt.found]).double()
    updates=dict(old_v5_endpoint=theta.cpu().double()-raw.theta0.cpu().double(),
                 external_expert_endpoint=expert_theta-raw.theta0.cpu().double())
    probes=[list(raw.rt.split['Xprime'][:128]),list(raw.rt.split['Xprime'][128:256])]
    config=dict(source_sha256=sha(__file__),basis_sha256=basis_digest,expert_sha256=expert_digest,base_sha256=raw.rt.input['base_sha256'],
        head_sha256=raw.rt.input['head_sha256'],state=128,probe_ids=probes,groups=offsets,
        group_interpretation='48 original matrices are predefined reporting bins, not discovered groups',
        null_draws=1000,null_seed=14011,pseudoinverse_relative_cutoff=1e-10,
        actual_updates='Both expert endpoint displacements from theta0, restricted to48 matrices; local evaluation at old theta128',
        training=False,basis_optimized=False,official_test_opened=False,part='A only; actual fusion anchor not selected')
    cfg=out/'config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:write(cfg,config)
    private=out/'private';private.mkdir(exist_ok=True)
    names=['ellB','ellQ','g','rho','a_old','a_expert','used_old','used_expert','self_old','self_expert']
    all_results=[]
    for probe_index,ids in enumerate(probes):
        folder=out/f'probe_{probe_index}';folder.mkdir(exist_ok=True)
        if (folder/'report.json').exists():
            all_results.append(json.loads((folder/'report.json').read_text()));continue
        raw.rt.split={**raw.rt.split,'X':ids}
        cache=private/f'probe_{probe_index}_FB.pt'
        response_file=private/f'probe_{probe_index}_R.npz'
        if cache.exists() and response_file.exists():
            FB=torch.load(cache,map_location='cpu',mmap=True,weights_only=True)['FB_rows']
            R=torch.from_numpy(np.load(response_file)['R'])
        else:
            columns=[];fisher=[]
            for j in range(10):
                response=raw.rt.jvp(theta,B[j]).cpu().double();columns.append(response)
                with torch.enable_grad():fb=raw.rt.vjp(theta,response.to(device='cuda',dtype=theta.dtype))
                fisher.append(fb.cpu().double());del fb
                write(out/'status.json',dict(probe=probe_index,stage='FB',column=j+1))
            R=torch.stack(columns,1);FB=torch.stack(fisher)
            temporary=cache.with_suffix('.tmp');torch.save(dict(FB_rows=FB),temporary);temporary.replace(cache)
            np.savez(response_file,R=R.numpy());del fisher,columns
        C=R.T@R;eig,V=torch.linalg.eigh(C);keep=eig>eig[-1]*1e-10
        Cd=(V[:,keep]/eig[keep])@V[:,keep].T
        assert int(keep.sum())==10
        hs={name:FB@v for name,v in updates.items()}
        denominators={name:float(h@Cd@h) for name,h in hs.items()}
        assert min(denominators.values())>1e-12
        coefficients={name:Cd@h/denominators[name] for name,h in hs.items()}
        maps={n:np.lib.format.open_memmap(private/f'p{probe_index}_{n}.npy',mode='w+',dtype='float32',shape=(D,)) for n in names}
        matrix_sums=[];q_grams=[]
        for gi,group in enumerate(offsets):
            sums={n:0. for n in names};qm=torch.zeros(26,26,dtype=torch.double)
            for start in range(group['start'],group['end'],262144):
                end=min(start+262144,group['end']);b=Bcpu[:,start:end];q=Q[:,start:end];f=FB[:,start:end]
                ellB=b.square().sum(0);ellQ=q.square().sum(0);g=(f*(Cd@f)).sum(0).clamp_min(0);rho=(b*f).sum(0)
                assert float((ellB-ellQ).max())<1e-8
                values=dict(ellB=ellB,ellQ=ellQ,g=g,rho=rho)
                for tag,update in [('old','old_v5_endpoint'),('expert','external_expert_endpoint')]:
                    v=updates[update][start:end]
                    values['a_'+tag]=v*(coefficients[update]@f)
                    values['used_'+tag]=v.square();values['self_'+tag]=v.square()*g
                for name,value in values.items():
                    maps[name][start:end]=value.numpy();sums[name]+=float(value.sum())
                qm.add_(q@q.T)
            matrix_sums.append(dict(**group,scores=sums));q_grams.append(qm.numpy())
            write(out/'status.json',dict(probe=probe_index,stage='coordinate_scores',matrix=gi+1,total=48))
        for v in maps.values():v.flush()
        assert abs(sum(m['scores']['ellB'] for m in matrix_sums)-10)<1e-5
        assert abs(sum(m['scores']['rho'] for m in matrix_sums)-float(C.trace()))/float(C.trace())<1e-4
        for name in ('a_old','a_expert'):assert abs(sum(m['scores'][name] for m in matrix_sums)-1)<1e-7
        stats={};tops={};k=int(D*.01)
        for name,values in maps.items():
            stats[name]=concentration(values)
            stats[name]['signed_sum']=float(values.sum(dtype=np.float64))
            stats[name]['positive_sum']=float(np.maximum(values,0).sum(dtype=np.float64))
            stats[name]['negative_sum']=float(np.minimum(values,0).sum(dtype=np.float64))
            if name in ('ellB','g','rho','a_old','a_expert','used_old','used_expert'):
                tops[name]=np.argpartition(np.abs(values),-k)[-k:]
            write(out/'status.json',dict(probe=probe_index,stage='concentration',score=name))
        overlap={}
        for a,ia in tops.items():
            for b,ib in tops.items():
                if a>=b:continue
                intersection=len(np.intersect1d(ia,ib,assume_unique=True))
                overlap[a+'__'+b]=dict(intersection_fraction=intersection/k,jaccard=intersection/(2*k-intersection))
        del tops,maps
        # Random10 within Q: matrix-bin null, not per-coordinate selection or final groups.
        rng=np.random.default_rng(14011);null=[];q_grams=np.array(q_grams)
        for _ in range(1000):
            W=np.linalg.qr(rng.normal(size=(26,10)))[0]
            null.append(np.einsum('ai,gab,bi->g',W,q_grams,W))
        null=np.array(null);actual=np.array([m['scores']['ellB'] for m in matrix_sums]);expected=q_grams.trace(axis1=1,axis2=2)*10/26
        pvalues=(1+(null>=actual).sum(0))/1001
        order=np.argsort(pvalues);holm=np.zeros(48);holm[order]=np.minimum(1,np.maximum.accumulate((48-np.arange(48))*pvalues[order]))
        for i,matrix in enumerate(matrix_sums):
            matrix['enrichment']=dict(observed=float(actual[i]/expected[i]),
                null_2p5=float(np.quantile(null[:,i]/expected[i],.025)),null_97p5=float(np.quantile(null[:,i]/expected[i],.975)),
                upper_tail_p=float(pvalues[i]),holm_adjusted_upper_p=float(holm[i]))
        # Keep local masked JVPs per matrix, so interrupted work can resume here.
        response_dir=private/f'probe_{probe_index}_matrices';response_dir.mkdir(exist_ok=True)
        group_responses=[]
        for gi,group in enumerate(offsets):
            path=response_dir/f'matrix_{gi:02d}.npz'
            if path.exists():Rg=torch.from_numpy(np.load(path)['R'])
            else:
                columns=[];direction=torch.zeros_like(theta)
                for j in range(10):
                    direction.zero_();direction[group['start']:group['end']]=B[j,group['start']:group['end']]
                    columns.append(raw.rt.jvp(theta,direction).cpu().double())
                    write(out/'status.json',dict(probe=probe_index,stage='masked_JVP',matrix=gi+1,column=j+1,total=48))
                Rg=torch.stack(columns,1);np.savez(path,R=Rg.numpy());del columns,direction
            group_responses.append(Rg)
        RG=torch.stack(group_responses);flat=RG.flatten(1);Gamma=flat@flat.T
        summed=RG.sum(0);relative=float((summed-R).norm()/R.norm());assert relative<1e-4
        rho_direct=np.array([m['scores']['rho'] for m in matrix_sums]);rho_response=Gamma.sum(1).numpy()
        rho_consistency=float(np.linalg.norm(rho_direct-rho_response)/np.linalg.norm(rho_response));assert rho_consistency<1e-4
        total=float(R.square().sum());deletions=[]
        for gi,matrix in enumerate(matrix_sums):
            delta=total-float((R-RG[gi]).square().sum())
            predicted=2*float((RG[gi]*R).sum())-float(RG[gi].square().sum())
            assert abs(delta-predicted)<max(total*1e-10,1e-10)
            deletions.append(dict(name=matrix['name'],rho=float((RG[gi]*R).sum()),self_energy=float(Gamma[gi,gi]),
                                  energy_change_on_direction_masking=delta,relative_energy_change=delta/total))
        result=dict(probe=probe_index,C=C.tolist(),C_eigenvalues=eig.tolist(),pseudoinverse_rank=int(keep.sum()),
            projected_update_energies=denominators,coordinate_scores=stats,matrix_scores=matrix_sums,
            top1percent_overlap=overlap,Gamma=Gamma.tolist(),Gamma_sum=float(Gamma.sum()),total_response_energy=total,
            sum_masked_response_relative_error=relative,rho_vjp_vs_JVP_relative_error=rho_consistency,
            direction_masking_energy_changes=deletions,
            caveats=['Gamma and deletion are local masked-direction responses, not CE change or formation causality.',
                     'a is signed alignment with the total projected update; not task-conflict contribution.',
                     'g uses unit original parameter coordinates; parameter scaling matters.',
                     'rho can be negative and is not an eigenvalue-formation attribution.',
                     'Scores are not combined into a single importance score.'])
        write(folder/'report.json',result);all_results.append(result)
        print(json.dumps(dict(probe=probe_index,ell_sum=stats['ellB']['signed_sum'],rho_sum=stats['rho']['signed_sum'],
            masked_sum_error=relative)),flush=True)
        del FB,RG,group_responses;torch.cuda.empty_cache()
    repeat={}
    for metric in ('ellB','g','rho','a_old','a_expert'):
        x=np.array([m['scores'][metric] for m in all_results[0]['matrix_scores']]);y=np.array([m['scores'][metric] for m in all_results[1]['matrix_scores']])
        repeat[metric]=float(np.corrcoef(x,y)[0,1])
    report=dict(config=config,probes=all_results,matrix_sum_repeatability_Pearson=repeat,
        experiment_B='Pending exact actual fusion checkpoint/recipe; no substitute fusion invented')
    write(out/'report.json',report);write(out/'status.json',dict(stage='complete',probes=2,training=False,experiment_B='not started'))


if __name__=='__main__':main()
