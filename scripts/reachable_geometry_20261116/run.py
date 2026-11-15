"""CPU-only reachability audit of the existing 304D anchored experiment."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/home/shangxueyuan/merging/tsv/scripts/conditional_compensation_20261116')
from program import solve

OLD=Path('/media/shangxueyuan/data1/conditional_compensation_20261116')
SPACE=Path('/media/shangxueyuan/data1/compromise_geometry_20261115/private/space.pt')
TAUS=[.001,.003,.01,.03,.1,.3,1.,3.]

def write(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)

def spectrum(Y):
    s=np.linalg.svd(Y,compute_uv=False);e=s*s;total=e.sum()
    return dict(singular_values=s.tolist(),cumulative_energy=(np.cumsum(e)/total if total else e).tolist(),
                participation_rank=float(total**2/max(e@e,1e-300)),
                numerical_rank=int(np.sum(s>s.max()*max(Y.shape)*np.finfo(float).eps)) if len(s) else 0,
                numerical_rank_is_precision_dependent=True)

def root(K):
    e,V=np.linalg.eigh((K+K.T)/2);return np.sqrt(np.maximum(e,0))[:,None]*V.T

def linear_support(K,h,R,ell):
    """Exact two-ellipsoid support via a scalar dual root; K is positive definite."""
    from scipy.optimize import brentq
    e,V=np.linalg.eigh((K+K.T)/2);b=-h;d=V.T@b
    if e.min()<=0:raise ValueError('Positive source metric required')
    v=R*b/np.linalg.norm(b);alpha=np.linalg.norm(b)/R;beta=0.
    if v@K@v>ell**2*(1+1e-12):
        z=V@(d/e);beta=np.sqrt(b@z)/ell;v=z/beta;alpha=0.
        if np.linalg.norm(v)>R:
            peak=e.max()
            def direction(q):return V@(d/(1+np.exp(q)*e/peak))
            def f(q):
                z=direction(q);return (z@K@z)/(z@z)-(ell/R)**2
            q=brentq(f,-40.,60.,xtol=1e-12);z=direction(q)
            v=R*z/np.linalg.norm(z);alpha=np.linalg.norm(z)/R;beta=alpha*np.exp(q)/peak
    response=np.sqrt(max(v@K@v,0));cost=np.linalg.norm(v)
    stat=np.linalg.norm(h+alpha*v+beta*K@v)/np.linalg.norm(h)
    primal=max(cost/R-1,response/ell-1,0.)
    comp=abs(alpha*(v@v-R*R)+beta*(v@K@v-ell*ell))
    audit=dict(status='ScalarDualRoot',primal_relative_violation=float(primal),stationarity_relative_residual=float(stat),
      complementarity_absolute=float(comp),predicted_Cars_CE=float(h@v),passed=bool(primal<1e-7 and stat<1e-7 and comp<1e-9),
      radius_multiplier=float(alpha),source_multiplier=float(beta),source_response=float(response),cost=float(cost))
    return v,audit

def main(out):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);start=time.monotonic()
    state=torch.load(SPACE,map_location='cpu',mmap=True,weights_only=True)
    G=np.linalg.qr(np.stack([state['a_D'].numpy(),state['a_C'].numpy()],1))[0]
    R0=.003*np.linalg.norm(state['a_D'].numpy()-state['a_C'].numpy())
    groups={g['key']:g for g in state['groups']};records=[];spectra=[];banks=[]
    lock=dict(source=str(OLD),scope='First experiment only: local functional reachable set; no support growth, dynamic update or EC',
      candidate_domain='Existing complete expert-recombination E, 304 dimensions; not full 86M space',
      taus=TAUS,radii=[R0/3,R0,R0*10/3],source_norm_budget='epsilon=R*tau*||J_D(-h_C/||h_C||)||',
      problems=['linear_Cars_support_norm_only','GN_Cars_norm_only','GN_Cars_with_DTD_CE_protection'],
      conditional_family='Union of the eight previously p0-selected conditional planes per anchor; NOT all-pair or all-2D upper bound',
      evidence='p4/p5 already observed; fully exploratory follow-up. No new heldout claim; p6 and Test unopened',
      reachable_spectrum='Inner ellipsoid v^T(I/R^2+KD/epsilon^2)v<=1 is feasible; reachable set is contained in sqrt(2) times this ellipsoid. All axes reported, no Fisher cutoff.',
      timestamp_unix=time.time(),main_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(out/'experiment_lock.json',lock)
    for k in range(3):
        with np.load(OLD/f'private/anchor{k}/geometry.npz') as f:
            KD=f['K_DTD'];KC=f['K_Cars'];hd=f['h_DTD'];hc=f['h_Cars']
        LD=root(KD);LC=root(KC);source=float(np.linalg.norm(LD@(-hc/np.linalg.norm(hc))))
        selected=json.loads((OLD/f'anchor{k}_selected_pair_lock.json').read_text())['selected']
        planes=[]
        for j,r in enumerate(selected):
            if r['kind']!='conditional_pair':continue
            W=np.zeros((304,2));W[r['ids']]=np.array(r['W']);planes.append((f'pair_{j}',W,r['groups']))
        spaces=[('full304',np.eye(304),None),('global2D',G,None)]+planes
        candidate_bank=[]
        for tau in TAUS:
            epsilon=R0*tau*source;M=np.eye(304)/R0**2+KD/epsilon**2
            L=np.linalg.cholesky(M);Z=np.linalg.solve(L,LC.T).T
            sp=spectrum(Z);sp.update(anchor=k,tau=tau,radius=R0,epsilon=epsilon,
                max_target_response_energy_bounds=[sp['singular_values'][0]**2,2*sp['singular_values'][0]**2])
            spectra.append(sp)
            for factor in [1/3,1.,10/3]:
                R=R0*factor;ell=R*tau*source
                for problem in lock['problems']:
                    candidates=[]
                    for name,W,gs in spaces:
                        dc=W.T@KD@W;cc=W.T@KC@W
                        dd=W.T@hd if problem.endswith('protection') else np.zeros(W.shape[1])
                        target=np.zeros_like(cc) if problem.startswith('linear') else cc
                        x,audit=solve(dc,target,dd,W.T@hc,R,ell)
                        v=W@x;row=dict(anchor=k,tau=tau,factor=factor,radius=R,epsilon=ell,problem=problem,
                          space=name,groups=gs,solver=audit,linear_Cars_gain=float(-hc@v),
                          GN_Cars_gain=float(-hc@v-.5*v@KC@v),source_response=float(np.linalg.norm(LD@v)),
                          Cars_response=float(np.linalg.norm(LC@v)),DTD_CE_slope=float(hd@v),parameter_norm=float(np.linalg.norm(v)),
                          v=v.tolist())
                        records.append(row)
                        if name.startswith('pair_') and audit['passed']:candidates.append(row)
                        if name=='full304' and audit['passed']:
                            candidate_bank.append((v/R,problem,tau,factor))
                    best=min(candidates,key=lambda r:r['solver']['predicted_Cars_CE']) if candidates else None
                    if best:
                        chosen={**best,'space':'best_cached_conditional_plane','selected_plane':best['space']}
                        records.append(chosen)
                write(out/'status.json',dict(stage='CPU_reachability',anchor=k,tau=tau,factor=factor,records=len(records),seconds=time.monotonic()-start))
        B=np.stack([v for v,_,_,_ in candidate_bank],1)
        for name,Kroot in [('parameter',np.eye(304)),('DTD_response',LD),('Cars_response',LC),('joint_response',np.r_[LD,LC])]:
            banks.append(dict(anchor=k,metric=name,spectrum=spectrum(Kroot@B),
              scope='Linear span of sampled single-Cars-objective optima across budgets, not rank of entire reachable set or Pareto dimension',
              bank_labels=[dict(problem=p,tau=t,factor=f) for _,p,t,f in candidate_bank]))
        write(out/'programs.json',records);write(out/'reachable_spectra.json',spectra);write(out/'optimal_motion_spectra.json',banks)
    # Actual finite validation is reused exactly at the already evaluated matched budgets.
    prior=json.loads((OLD/'finite_results.json').read_text())
    actual=[r for r in prior if r['kind'] in ['full304_upper','global2D_local','conditional_pair']]
    write(out/'existing_finite_validation.json',dict(records=actual,no_new_models_evaluated=True,
      applies_only_to='Previous GN plus DTD-CE-protection programs, old four taus/radii; not validation of every new support-function optimum'))
    pairs=[]
    for row in records:
        if row['space']!='full304':continue
        case={key:row[key] for key in ['anchor','tau','factor','problem']}
        refs=[r for r in records if all(r[key]==v for key,v in case.items()) and r['space'] in ['global2D','best_cached_conditional_plane']]
        for ref in refs:
            gain='linear_Cars_gain' if row['problem'].startswith('linear') else 'GN_Cars_gain'
            pairs.append(dict(**case,reference=ref['space'],full_gain=row[gain],reference_gain=ref[gain],
              gain_fraction=ref[gain]/max(row[gain],1e-30),full_solver_passed=row['solver']['passed'],reference_solver_passed=ref['solver']['passed']))
    write(out/'capacity_gaps.json',pairs)
    write(out/'complete.json',dict(done=True,program_records=len(records),failed=sum(not r['solver']['passed'] for r in records),
       seconds=time.monotonic()-start,no_GPU_use=True,no_old_jobs_modified=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();main(a.output)
