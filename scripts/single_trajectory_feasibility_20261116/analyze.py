"""One-existing-trajectory feasibility: no training, JVPs, or comparison suite."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/home/shangxueyuan/merging/tsv/scripts/adaptive_fiber_20261116_r2')
from run_a100 import setup,all_metrics,write_json,OLD
from fast_online import Costs
DATA=Path('/media/shangxueyuan/data1/anchor0_coevolution_20261116')

def spherical_terms(ref,q):
    radius=2/np.sqrt(ref.shape[-2]);c=np.clip(np.sum(ref*q,axis=-1)/(radius*radius),-1,1)
    theta=np.arccos(c);s=np.sin(theta);small=theta<1e-4
    f=np.ones_like(theta);fc=np.full_like(theta,-1/3.)
    f[~small]=theta[~small]/s[~small]
    fc[~small]=(theta[~small]*c[~small]-s[~small])/s[~small]**3
    f[small]=1+theta[small]**2/6+7*theta[small]**4/360
    fc[small]=-1/3-2*theta[small]**2/15
    return radius,c,f,fc

def log_map(ref,q):
    _,c,f,_=spherical_terms(ref,q);v=f[...,None]*(q-c[...,None]*ref)
    return v-ref*(np.sum(v*ref,axis=-1)/np.sum(ref*ref,axis=-1))[...,None]

def dlog(ref,q,dq):
    single=dq.ndim==2
    if single:dq=dq[:,:,None]
    radius,c,f,fc=spherical_terms(ref,q)
    dc=np.einsum('nc,nck->nk',ref,dq)/radius**2
    out=f[:,None,None]*(dq-ref[:,:,None]*dc[:,None,:])
    out+=fc[:,None,None]*(q-c[:,None]*ref)[:,:,None]*dc[:,None,:]
    out-=ref[:,:,None]*(np.einsum('nc,nck->nk',ref,out)/np.sum(ref*ref,axis=1)[:,None])[:,None,:]
    return out[:,:,0] if single else out

def orth(A):
    if A.shape[1]==0:return A
    U,s,_=np.linalg.svd(A,full_matrices=False);return U[:,s>s[0]*1e-10]

def residual(y,B):
    U=orth(B);return float(np.linalg.norm(y-U@(U.T@y))/max(np.linalg.norm(y),1e-30))

def spectrum(A):
    s=np.linalg.svd(A,compute_uv=False);e=s*s
    return dict(singular_values=s.tolist(),cumulative_energy=(np.cumsum(e)/e.sum()).tolist(),
      participation_rank=float(e.sum()**2/(e@e)))

def centered_ensemble_covariance(curves):
    X=np.asarray(curves);X=X-X.mean(0,keepdims=True);X=X.reshape(len(X),-1)
    return X@X.T/len(X)

def main(out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    control=json.loads((DATA/'CONTROL_STATE.json').read_text());prior=json.loads((DATA/'diagnostic_lock.json').read_text())
    raw=[np.array(r['parameter_update']) for r in control];lengths=np.array([np.linalg.norm(v) for v in raw]);weights=lengths/lengths.sum()
    parameter=np.stack(raw,1);ensemble=centered_ensemble_covariance(parameter.T[None,:,:])
    assert np.array_equal(ensemble,np.zeros((1,1)))
    c,s,norm=setup(out);cost=Costs(c)
    def forbidden(*a,**kw):raise AssertionError('This analysis prohibits new JVPs')
    c.response=forbidden
    initial=np.array(json.loads((OLD/'anchor_selection_lock.json').read_text())['selected'][0]['a'])
    point=s.point(torch.tensor(initial,device='cuda',dtype=torch.float64))
    summary=dict(scope='Feasibility on the single existing trajectory only; no ensemble FPCA fitted',
      fpca_centered_single_curve_covariance=ensemble.tolist(),arc_progress=np.r_[0,np.cumsum(weights)].tolist(),
      parameter_temporal_energy_spectrum=spectrum(parameter/np.sqrt(weights)[None,:]),probes={})
    all_logs={};fit_rows=[]
    for probe in [0,4]:
        metrics=all_metrics(c,point,probe)
        with np.load(DATA/f'actual_trajectory_p{probe}.npz') as f:R=f['joint_raw']
        nD=metrics['DTD']['phi'].numel();parts={'DTD':R[:nD],'Cars':R[nD:]};log_parts={};qpoints={}
        for task in ['DTD','Cars']:
            ref=metrics[task]['phi'].numpy();q=np.r_[ref[None,:,:],ref[None,:,:]+np.cumsum(parts[task].T.reshape(5,*ref.shape),axis=0)]
            radius=2/np.sqrt(len(ref));norm_error=float(np.max(abs(np.linalg.norm(q,axis=-1)-radius)))
            assert norm_error<1e-10
            z=log_map(ref[None,:,:],q);delta=np.diff(z,axis=0).reshape(5,-1).T
            log_parts[task]=delta;qpoints[task]=(ref,q)
        L=np.r_[log_parts['DTD']/prior['fixed_joint_normalization']['DTD'],log_parts['Cars']/prior['fixed_joint_normalization']['Cars']]
        all_logs[probe]=L
        temporal=L/np.sqrt(weights)[None,:]
        mean_velocity=L.sum(1);velocity=L/weights[None,:]
        centered=(velocity-mean_velocity[:,None])*np.sqrt(weights)[None,:]
        summary['probes'][str(probe)]=dict(log_increment_spectrum=spectrum(L),
          arc_weighted_velocity_spectrum=spectrum(temporal),time_centered_velocity_spectrum=spectrum(centered),
          prefix_unseen_direction_residual=[residual(L[:,t],L[:,:t]) for t in range(1,5)],
          definition='Prefix subspace uses past increments only; geometric expressibility test, not future amplitude prediction or population FPCA')
        if probe==0:
            for t,r in enumerate(control):
                with np.load(DATA/f'mode_response_step{t}.npz') as f:
                    cols=[]
                    for task,key in [('DTD','J_D_B'),('Cars','J_C_B')]:
                        ref,q=qpoints[task];v=f[key].reshape(*ref.shape,-1)
                        cols.append(dlog(ref,q[t],v).reshape(ref.size,-1)/prior['fixed_joint_normalization'][task])
                    Psi=np.r_[cols[0],cols[1]];z=f['z'];pred=Psi@z;actual=L[:,t]
                    oldn=len(control[t-1]['modes']) if t else 0
                    fit_rows.append(dict(step=t,relative_log_prediction_error=float(np.linalg.norm(pred-actual)/np.linalg.norm(actual)),
                      cosine=float(pred@actual/(np.linalg.norm(pred)*np.linalg.norm(actual))),
                      actual_response_outside_current_old_mode_span=residual(actual,Psi[:,:oldn]) if oldn else 1.))
        np.savez_compressed(out/f'log_trajectory_p{probe}.npz',joint_log_increments=L,arc_weighted_velocity=temporal)
    summary['common_reference_fit']=fit_rows
    actual_run=json.loads(Path('/media/shangxueyuan/data1/adaptive_fiber_20261116_r2/archived_other_anchors/anchor0/adaptive/FAST_FINAL.json').read_text())
    summary['observed_task_budget_sequence']=[r['scenario'] for r in actual_run['trajectory']]
    write_json(out/'SUMMARY.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    for probe in [0,4]:
        d=summary['probes'][str(probe)]
        axs[0].plot(range(1,6),d['arc_weighted_velocity_spectrum']['cumulative_energy'],'-o',label=f'p{probe}: time-weighted')
        axs[1].plot(range(1,5),d['prefix_unseen_direction_residual'],'-o',label=f'p{probe}: past trajectory span')
    axs[1].plot(range(1,5),[r['actual_response_outside_current_old_mode_span'] for r in fit_rows[1:]],'--o',label='p0: current old controller span')
    axs[2].plot(range(5),[r['relative_log_prediction_error'] for r in fit_rows],'-o')
    axs[0].set(title='Single-path temporal energy; not ensemble FPCA',xlabel='Axes',ylabel='Cumulative energy')
    axs[1].set(title='Can available directions express the next motion?',xlabel='Step',ylabel='Relative residual')
    axs[2].set(title='DLog J B z vs actual common-reference increment',xlabel='Step',ylabel='Relative error')
    axs[0].legend(fontsize=7);axs[1].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'feasibility.png',dpi=170);plt.close(fig)
    write_json(out/'COMPLETE.json',dict(done=True,new_seed_runs=0,new_trajectories=0,new_JVPs=cost.data['task_JVP_calls'],
      dataset_forward_calls=cost.data['dataset_forward_calls'],wall_seconds=time.monotonic()-start,
      product_sphere_log_per_sample=True,parameter_updates_written=False))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();main(args.output)
