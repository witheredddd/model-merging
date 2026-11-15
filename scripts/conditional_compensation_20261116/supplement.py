"""Additive audit: freeze independent controls before held reads, evaluate after main release."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from cache import context,Space,OLD,norm,save
from program import solve,conditional


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8');tmp.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(x):
    x=np.asarray(x);rng=np.random.default_rng(202611160154)
    return np.quantile(x[rng.integers(len(x),size=(1000,len(x)))].mean(1),[.025,.975]).tolist()


def geometry(out,k):
    with np.load(out/f'private/anchor{k}/geometry.npz') as f:
        return {key:f[key] for key in ['K_DTD','K_Cars','h_DTD','h_Cars','a']}


def mapping(data,im,jn,damping=None):
    KD=data['K_DTD'];A=KD[np.ix_(jn,jn)];C=KD[np.ix_(jn,im)];e,V=np.linalg.eigh((A+A.T)/2)
    largest=max(float(abs(e).max()),1e-30);keep=e>largest*1e-10
    pinv=(V[:,keep]/e[keep])@V[:,keep].T if keep.any() else np.zeros_like(A)
    if damping is None:T=-pinv@C
    else:T=-np.linalg.solve(A+damping*np.eye(len(jn)),C)
    B=np.r_[np.eye(len(im)),T];W=np.linalg.qr(B,mode='reduced')[0]
    N=V[:,~keep];target_null=N.T@data['K_Cars'][np.ix_(jn,jn)]@N
    audit=dict(eigenvalues=e.tolist(),condition_number=float(e.max()/e.min()) if e.min()>0 else None,
       numerical_rank=int(keep.sum()),numerical_nullity=int((~keep).sum()),relative_rank_tolerance=1e-10,
       Cars_Fisher_trace_in_source_nullspace=max(float(np.trace(target_null)),0.),
       raw_Cars_trace_before_roundoff_clamp=float(np.trace(target_null)),T=T.tolist(),damping=damping,
       Cars_CE_gradient_norm_in_source_nullspace=float(np.linalg.norm(N.T@data['h_Cars'][jn])),
       source_nullspace_Cars_effect_not_assumed_zero=True)
    return T,W,audit


def freeze(out):
    parent=json.loads((out/'all_candidates_frozen.json').read_text())['models']
    positive=[m for m in parent if m['kind']=='conditional_pair' and m['id'].endswith('_r1')]
    basis=json.loads((OLD/'basis_audit.json').read_text());groups=[g for g in basis['groups'] if g['indices']]
    byname={g['key']:i for i,g in enumerate(groups)};data={k:geometry(out,k) for k in sorted(set(m['anchor'] for m in positive))}
    rng=np.random.default_rng(202611160154);bank=[];damping_audits=[];canonical={}
    def candidate(ident,kind,k,im,jn,W,R,ell,parent_id,extra):
        D=data[k];ids=im+jn if jn else im;KD=D['K_DTD'];KC=D['K_Cars'];hd=D['h_DTD'];hc=D['h_Cars']
        z,audit=solve(W.T@KD[np.ix_(ids,ids)]@W,W.T@KC[np.ix_(ids,ids)]@W,W.T@hd[ids],W.T@hc[ids],R,ell)
        v=np.zeros(len(hc));v[ids]=W@z
        bank.append(dict(id=ident,kind=kind,anchor=k,parent=parent_id,im=im,jn=jn,v=v.tolist(),radius=R,ell=ell,audit=audit,**extra))
    for p in positive:
        k=p['anchor'];D=data[k];m,n=[byname[name] for name in p['groups']];im=groups[m]['indices'];jn=groups[n]['indices']
        R=p['radius'];ell=p['ell'];tag=p['id'];KD=D['K_DTD']
        # Random choice depends only on rank and support sizes, never model outcomes.
        pool=[]
        for i,g in enumerate(groups):
            if len(g['indices'])!=len(im):continue
            for j,h in enumerate(groups):
                if i==j or {i,j}=={m,n} or len(h['indices'])!=len(jn):continue
                distance=abs(np.log(g['size']/groups[m]['size']))+abs(np.log(h['size']/groups[n]['size']))
                pool.append((distance,i,j))
        pool.sort();exact=[x for x in pool if x[0]<1e-14]
        eligible=exact if exact else pool[:min(20,len(pool))]
        distance,rm,rn=eligible[int(rng.integers(len(eligible)))];ri=groups[rm]['indices'];rj=groups[rn]['indices']
        W,op=conditional(KD,ri,rj)
        candidate(tag+'_random','matched_random_pair' if exact else 'random_pair_support_mismatch',k,ri,rj,W,R,ell,tag,
          dict(groups=[groups[rm]['key'],groups[rn]['key']],support_size_log_mismatch=distance,
               exact_support_size_match=bool(exact),random_seed=202611160154,operator=op))
        currentW,current=conditional(KD,im,jn);Tcurrent=np.array(current['T'])
        peak=max(float(np.linalg.eigvalsh(KD[np.ix_(jn,jn)]).max()),1e-30)
        variants=[None,peak*1e-10,peak*1e-8,peak*1e-6,peak*1e-4]
        audits=[]
        for damping in variants:
            T,W,info=mapping(D,im,jn,damping)
            info.update(T_change_Frobenius=float(np.linalg.norm(T-Tcurrent)),
                        relative_T_change=float(np.linalg.norm(T-Tcurrent)/max(np.linalg.norm(Tcurrent),1e-30)))
            driver=np.array(p['v'])[im];parts=np.r_[driver,T@driver]
            parts*=min(1.,R/max(np.linalg.norm(parts),1e-30));ids=im+jn
            info['fixed_driver_direction_diagnostic']=dict(
                source_response=float(np.sqrt(max(parts@KD[np.ix_(ids,ids)]@parts,0))),
                cancelled_DTD_energy=float(-2*parts[:len(im)]@KD[np.ix_(im,jn)]@parts[len(im):]),
                Cars_CE_slope=float(D['h_Cars'][ids]@parts),parameter_cost=float(np.linalg.norm(parts)),
                global_rescaling_only_if_needed_for_R=True)
            audits.append(info)
        damping_audits.append(dict(parent=tag,anchor=k,groups=p['groups'],current=current,variants=audits))
        for suffix,damping in [('pinv',None),('perturbed',max(current['damping']*10,peak*1e-6))]:
            T,W,info=mapping(D,im,jn,damping)
            candidate(tag+'_'+suffix,'damping_'+suffix,k,im,jn,W,R,ell,tag,dict(groups=p['groups'],operator=info))
        for suffix,ids in [('driver',im),('compensator',jn)]:
            candidate(tag+'_'+suffix+'_reoptimized','same_budget_single',k,ids,[],np.eye(len(ids)),R,ell,tag,
                      dict(single_role=suffix,groups=[p['groups'][0 if suffix=='driver' else 1]]))
        canonical.setdefault((m,n),p)
    # Same fixed driver orientation, re-evaluated through T at each anchor; not a selected control.
    transfer=[]
    for (m,n),p in canonical.items():
        im=groups[m]['indices'];jn=groups[n]['indices'];driver=np.array(p['v'])[im]
        maps=[]
        for k,D in data.items():
            W,info=conditional(D['K_DTD'],im,jn);T=np.asarray(info['T']);v=np.zeros(len(D['h_Cars']))
            v[im]=driver;v[jn]=T@driver;v*=min(1.,p['radius']/max(np.linalg.norm(v),1e-30))
            maps.append((k,T));xm=v[im];xn=v[jn]
            transfer.append(dict(pair=[groups[m]['key'],groups[n]['key']],anchor=k,T=T.tolist(),
                source_response=float(np.sqrt(max(v@D['K_DTD']@v,0))),Cars_CE_slope=float(D['h_Cars']@v),
                cancelled_DTD_energy=float(-2*xm@D['K_DTD'][np.ix_(im,jn)]@xn),
                scope='Fixed canonical driver, cached local response; finite cross-anchor evaluation in supplementary bank'))
            bank.append(dict(id=f'transfer_{m}_{n}_a{k}',kind='cross_anchor_fixed_driver',anchor=k,parent=p['id'],
                im=im,jn=jn,v=v.tolist(),radius=p['radius'],ell=None,groups=[groups[m]['key'],groups[n]['key']]))
        for i in range(len(maps)):
            for j in range(i+1,len(maps)):
                ki,Ti=maps[i];kj,Tj=maps[j];den=float(np.linalg.norm(Ti)*np.linalg.norm(Tj))
                transfer.append(dict(pair=[groups[m]['key'],groups[n]['key']],anchor_pair=[ki,kj],
                    T_difference_Frobenius=float(np.linalg.norm(Ti-Tj)),cosine=float(np.sum(Ti*Tj)/den) if den>1e-30 else None))
    before=not any((out/'private').glob('finite_p4_*.pt')) and not any((out/'private').glob('finite_p5_*.pt'))
    lock=dict(models=bank,random_seed=202611160154,random_selection_uses_no_p4_p5=True,
       locked_before_first_main_held_model_file=before,main_candidate_file_sha256=digest(out/'all_candidates_frozen.json'),
       support_matching='Exact ordered tensor element counts and coordinate ranks where alternatives exist; otherwise deterministic nearest20 pool, mismatch explicit',
       main_algorithm_candidates_and_fitness_unchanged=True,
       scope='Additional exploratory controls, no main-result replacement; p4/p5 previously consumed',
       p6='Not sampled or evaluated. Freeze this bank and comparison list; confirmation margins/multiplicity protocol required before opening new p6.')
    write(out/'extra_controls_lock.json',lock);write(out/'damping_nullspace_audit.json',damping_audits)
    write(out/'cross_anchor_map_audit.json',transfer)
    write(out/'p6_confirmation_bundle.json',dict(main_candidate_sha256=lock['main_candidate_file_sha256'],
       extra_controls_sha256=digest(out/'extra_controls_lock.json'),data_evidence=dict(p0='discovery',p4='reused stability',p5='reused frozen training holdout',p6='future confirmation, unopened'),
       comparisons=['selected_vs_matched_random','selected_vs_reoptimized_driver','selected_vs_reoptimized_compensator','current_vs_pinv_vs_damped'],
       existing_CIs_are_not_multiple_testing_discovery_certificates=True))
    return bank


def evaluate(out,bank):
    work=out/'extra_runtime';work.mkdir(exist_ok=True)
    for name in ['probes.json','independent_probes.json']:work.joinpath(name).write_bytes((out/name).read_bytes())
    c=context(work);state=torch.load(OLD/'private/space.pt',map_location='cpu',weights_only=True,mmap=True);s=Space(c,state)
    anchors=json.loads((out/'anchor_selection_lock.json').read_text())['selected'];allrows=[];baselines={};JG={}
    for row in bank:
        k=row['anchor'];a0=torch.tensor(anchors[k]['a'],device='cuda',dtype=torch.float64);point0=s.point(a0)
        v=torch.tensor(row['v'],device='cuda',dtype=torch.float64);point=s.point(a0+v)
        for unused in range(6):
            cost=norm(point.double()-point0.double())
            if cost<=row['radius']*(1+1e-8):break
            v*=row['radius']/max(cost,1e-30)*.9999;point=s.point(a0+v)
        for p in [4,5]:
            write(out/'extra_audit_status.json',dict(stage='post_main_forward_audit',point=row['id'],probe=p))
            result={key:value for key,value in row.items() if key not in ['v','operator']};result.update(probe=p,actual_parameter_cost=cost,metrics={})
            for t in ['DTD','Cars']:
                key=(k,p,t)
                if key not in baselines:
                    z0,y=c.baseline(t,p,point0);baselines[key]=(z0,y,F.cross_entropy(z0,y,reduction='none'))
                z0,y,L0=baselines[key];z,_=c.baseline(t,p,point)
                loss=F.cross_entropy(z,y,reduction='none');delta=(loss-L0).numpy()
                original=torch.load(out/f'private/finite_p{p}_{row["parent"]}_{t}.pt',map_location='cpu',weights_only=True)
                paired=(original['loss']-loss).numpy()
                result['metrics'][t]=dict(CE_change=float(delta.mean()),CI95=ci(delta),
                    accuracy=float((z.argmax(-1)==y).double().mean()),
                    KL=float((z0.softmax(-1)*(z0.log_softmax(-1)-z.log_softmax(-1))).sum(-1).mean()),
                    selected_parent_minus_control_CE=float(paired.mean()) if row['kind']!='cross_anchor_fixed_driver' else None,
                    parent_minus_control_CI95=ci(paired) if row['kind']!='cross_anchor_fixed_driver' else None)
                save(c.private/f'extra_p{p}_{row["id"]}_{t}.pt',dict(logits=z,labels=y,loss=loss))
                if t=='DTD':joint_phi=2*z.softmax(-1).sqrt()/np.sqrt(len(y));base_phi=2*z0.softmax(-1).sqrt()/np.sqrt(len(y))
            if row['jn']:
                vm=torch.zeros_like(v);vn=torch.zeros_like(v);vm[row['im']]=v[row['im']];vn[row['jn']]=v[row['jn']]
                rm,_=c.response(s.expand(vm),'DTD',p,point0);rn,_=c.response(s.expand(vn),'DTD',p,point0)
                denom=float(rm@rm+rn@rn);result['rho_cancel']=float(-2*rm@rn/max(denom,1e-30))
                result['DTD_joint_linear_energy']=float((rm+rn)@(rm+rn));result['DTD_separate_linear_energy']=denom
                components=[]
                for part in [vm,vn]:
                    zz,_=c.baseline('DTD',p,s.point(a0+part));components.append(2*zz.softmax(-1).sqrt()/np.sqrt(len(zz))-base_phi)
                dj=joint_phi-base_phi;dm,dn=components;separate=float((dm*dm).sum()+(dn*dn).sum())
                result['finite_DTD_energy_reduction']=separate-float((dj*dj).sum())
                result['finite_rho_energy_reduction']=result['finite_DTD_energy_reduction']/max(separate,1e-30)
                result['finite_additivity_error']=float((dj-dm-dn).norm()/max(float(dj.norm()),1e-30))
                rc,_=c.response(s.expand(v),'Cars',p,point0)
                key=(k,p)
                if key not in JG:JG[key]=np.stack([c.response(s.expand(s.global_basis[:,j]),'Cars',p,point0)[0] for j in range(2)],1)
                residual=rc-JG[key]@np.linalg.lstsq(JG[key],rc,rcond=1e-8)[0]
                result['eta_C_independent_response_norm']=float(np.linalg.norm(residual)/max(np.linalg.norm(rc),1e-30))
                result['parameter_fraction_outside_G']=s.novelty(v)['new_fraction']
                result['belongs_to_Q_by_construction']=True
            allrows.append(result);write(out/'extra_validation_audit.json',dict(points=allrows,
                individual_CIs_not_multiplicity_corrected=True,main_results_unchanged=True))
    write(out/'extra_audit_complete.json',dict(done=True,main_untouched=True,models=len(bank),evaluation_points=len(allrows)))


def main(out):
    write(out/'extra_audit_status.json',dict(stage='waiting_for_main_candidate_lock; no_GPU_use',main_untouched=True))
    while not (out/'all_candidates_frozen.json').exists():
        if (out/'run_failure.json').exists():raise RuntimeError('Main incomplete; audit leaves it untouched')
        time.sleep(3)
    bank=freeze(out)
    write(out/'extra_audit_status.json',dict(stage='controls_frozen; waiting_for_main_GPU_release',models=len(bank)))
    while json.loads((out/'status.json').read_text()).get('stage')!='conditional_compensation_complete':
        if (out/'run_failure.json').exists():raise RuntimeError('Main incomplete; no supplementary GPU run')
        time.sleep(5)
    time.sleep(3);evaluate(out,bank)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    try:main(a.output)
    except Exception:
        write(a.output/'extra_audit_failure.json',dict(traceback=traceback.format_exc(),main_untouched=True));raise
