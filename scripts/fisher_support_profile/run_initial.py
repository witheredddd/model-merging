"""Support-size profiles, not a fixed-budget controller. Dimension read from input."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    out.mkdir(exist_ok=True,parents=True);private=out/'private';private.mkdir(exist_ok=True)
    root=Path('/home/shangxueyuan/merging/tsv');sys.path.insert(0,str(root/'scripts/fisher_rao_online_v5_98f36a1'))
    from clip_runtime import ClipTask
    from checkpointing import load_model
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    raw=ClipTask(root,out,128,32,task='DTD');torch.set_grad_enabled(False)
    previous=Path('/media/shangxueyuan/data1/DTD_fixed10_support_20261114')
    old=json.loads((previous/'config.json').read_text())
    basis=torch.load('/media/shangxueyuan/data1/DTD_fixed10_20261113/fixed_basis.pt',map_location='cpu',mmap=True,weights_only=True)
    assert basis['identity']['base']==raw.rt.input['base_sha256']
    assert basis['identity']['head']==raw.rt.input['head_sha256']
    # Names follow descending eigenvalue; no rank is selected or optimized here.
    order=torch.argsort(basis['eigenvalues'],descending=True)
    B=basis['B_rows'][order].float().cuda();r,D=B.shape
    FB=torch.load(previous/'private/probe_0_FB.pt',map_location='cpu',mmap=True,weights_only=True)['FB_rows'][order]
    R=[torch.from_numpy(np.load(previous/f'private/probe_{i}_R.npz')['R'])[:,order] for i in range(2)]
    probes=old['probe_ids']
    matrices=np.load('/media/shangxueyuan/data1/DTD_fisher_blocks_expanded_20261113/block_matrices.npz')['F']
    ids=[1,3,5,7];average=sum(matrices[i]/np.linalg.norm(matrices[i]) for i in ids)/len(ids)
    eigen,U=np.linalg.eigh(average);U=U[:,-r:][:,::-1];values=eigen[-r:][::-1]
    stability=[]
    for omitted in ids:
        _,V=np.linalg.eigh(sum(matrices[i]/np.linalg.norm(matrices[i]) for i in ids if i!=omitted)/(len(ids)-1))
        V=V[:,-r:][:,::-1];stability.append(np.abs(np.diag(U.T@V)).tolist())
    fractions=[.00001,.0001,.001,.01,.05,.1,.2,.5,.8,.9,.95,.99,1.]
    qs=[.5,.8,.9,.95,.99]
    config=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        basis_sha256=old['basis_sha256'],base_sha256=old['base_sha256'],head_sha256=old['head_sha256'],
        input_dimension=r,D=D,anchor_step=128,probes=probes,ranking_probe=0,
        support_fractions=fractions,mass_targets=qs,extra_support_points='Exact mass-target k from each full concentration curve',
        eigenvalues_descending=values.tolist(),relative_eigen_gaps=((values[:-1]-values[1:])/values[:-1]).tolist(),
        leave_one_state_out_abs_cosines=stability,
        minimum_mode_LOO_cosine=np.min(stability,axis=0).tolist(),
        mode_stability_caveat='LOO history-operator robustness only; not a statistical perturbation bound. Close eigenvectors are labeled with caveats.',
        no_fixed_support_budget=True,no_coefficient_refit=True,no_mask_renormalization=True,training=False,
        scope='Support profile of supplied space, not automatic global Fisher rank discovery',official_test_opened=False)
    cfg=out/'config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:write(cfg,config)
    checkpoints=Path('/media/shangxueyuan/data1/fisher_rao_online_v5_DTD_medium_seed11_98f36a1_20261112/checkpoints')
    state=torch.load(next(checkpoints.glob('step_000128_*.pt')),map_location='cpu',mmap=True,weights_only=True)
    theta=load_model(state['state'],raw).parameters().detach()
    scores_dir=private/'orders';scores_dir.mkdir(exist_ok=True)
    def measure(indices,k,columns,probe):
        ref=R[probe][:,columns];responses=[];direction=torch.zeros_like(theta)
        raw.rt.split={**raw.rt.split,'X':probes[probe]};raw.rt.n=len(probes[probe])
        for column in columns:
            if k==D:responses.append(R[probe][:,column]);continue
            direction.zero_();chosen=indices[:k];direction[chosen]=B[column,chosen]
            responses.append(raw.rt.jvp(theta,direction).cpu().double())
        masked=torch.stack(responses,1);norm=float(ref.norm());mn=float(masked.norm())
        return dict(error=float((ref-masked).norm())/norm,
                    cosine=float((ref*masked).sum())/(norm*mn) if mn else None,
                    amplitude=mn/norm,
                    per_column_errors=((ref-masked).square().sum(0).sqrt()/ref.square().sum(0).sqrt()).tolist())
    entities=[(f'mode_{i+1:02d}',[i]) for i in range(r)]+[('whole_subspace',list(range(r)))]
    reports=[]
    for name,columns in entities:
        entity=out/name;entity.mkdir(exist_ok=True)
        for ranking in ['parameter','absolute_rho']:
            target=entity/f'{ranking}.json'
            if target.exists():reports.append(json.loads(target.read_text()));continue
            write(out/'status.json',dict(stage='full_coordinate_sort',entity=name,ranking=ranking))
            param=B[columns].double().square().sum(0)
            rho=(B[columns].double()*FB[columns].cuda()).sum(0)
            score=param if ranking=='parameter' else rho.abs()
            total=float(score.sum());assert total>0
            index=torch.argsort(score,descending=True)
            path=scores_dir/f'{name}_{ranking}.pt'
            if not path.exists():
                temporary=path.with_suffix('.tmp');torch.save(index.cpu(),temporary);temporary.replace(path)
            cumulative=torch.cumsum(score[index],0)
            mass_support={str(q):int(torch.searchsorted(cumulative,torch.tensor(q*total,device='cuda',dtype=torch.double)))+1 for q in qs}
            # Full concentration curve is exact; display it on a dense log/linear grid.
            plot_counts=sorted(set([max(1,int(f*D)) for f in np.r_[np.geomspace(1/D,1,120),np.linspace(.01,1,100)]]+[D]))
            curve=[dict(k=k,fraction=k/D,mass=float(cumulative[k-1])/total) for k in plot_counts]
            counts=sorted(set([max(1,int(f*D)) for f in fractions]+list(mass_support.values())+[D]))
            measurements=[];cache=entity/ranking;cache.mkdir(exist_ok=True)
            for k in counts:
                record=cache/f'k_{k:09d}.json'
                if record.exists():row=json.loads(record.read_text())
                else:
                    row=dict(k=k,fraction=k/D,parameter_mass=float(param[index[:k]].sum()/param.sum()),
                        signed_rho_fraction=float(rho[index[:k]].sum()/rho.sum()),
                        absolute_rho_fraction=float(rho[index[:k]].abs().sum()/rho.abs().sum()),probes=[])
                    for probe in range(2):
                        row['probes'].append(measure(index,k,columns,probe))
                        write(out/'status.json',dict(stage='masked_response',entity=name,ranking=ranking,k=k,fraction=k/D,probe=probe))
                    write(record,row)
                measurements.append(row)
            functional=[]
            for probe in range(2):
                result={}
                for quality in [.8,.9,.95,.99]:
                    hits=[row for row in measurements if row['probes'][probe]['error']<=1-quality]
                    first=hits[0] if hits else None
                    result[str(quality)]=dict(first_observed_support_fraction=first['fraction'] if first else None,
                        first_observed_k=first['k'] if first else None,
                        later_sampled_violations=sum(row['probes'][probe]['error']>1-quality for row in measurements if first and row['k']>first['k']))
                functional.append(result)
            result=dict(entity=name,columns_descending=columns,ranking=ranking,
                full_parameter_mass=float(param.sum()),signed_rho_total=float(rho.sum()),
                positive_rho_sum=float(rho.clamp_min(0).sum()),negative_rho_sum=float(rho.clamp_max(0).sum()),
                concentration_curve=curve,mass_support_fraction={q:k/D for q,k in mass_support.items()},
                mass_support_k=mass_support,measurements=measurements,functional_thresholds=functional,
                threshold_caveat='Functional quality means relative response error <=1-q, not energy retained. First observed along this sampled nested ranking; not global minimum support and no monotonicity assumed.',
                attribution_caveat='Absolute rho mass is signed Fisher allocation magnitude, not a functional response retention fraction.')
            write(target,result);reports.append(result)
            print(json.dumps(dict(entity=name,ranking=ranking,mass95=mass_support['0.95']/D)),flush=True)
            del index,cumulative,param,rho,score;torch.cuda.empty_cache()
    write(out/'report.json',dict(config=config,profiles=reports,
        limitations=['Per-mode profiles depend on basis choice; whole-space leverage and Frobenius profiles are rotation invariant.',
                     'JVP responses are local derivatives at theta128, not finite-step CE or expert accuracy.',
                     'Rho masks use probe0 only; probe1 checks use the exact same masks without reselection.',
                     'Support sizes are empirical profiles, never identified with supplied subspace dimension.']))
    write(out/'status.json',dict(stage='complete',profiles=len(reports),no_fixed_support_budget=True))


if __name__=='__main__':main()
