"""Existing-mode JVPs only: 50 requests, zero coordinate/full304 refreshes."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0,'/home/shangxueyuan/merging/tsv/scripts/adaptive_fiber_20261116_r2')
from run_a100 import setup,all_metrics,write_json,OLD
from fast_online import Costs

R1=Path('/media/shangxueyuan/data1/adaptive_fiber_20261116_r1/main/checkpoints')
R2=Path('/media/shangxueyuan/data1/adaptive_fiber_20261116_r2/archived_other_anchors/anchor0/adaptive')
CACHE=Path('/media/shangxueyuan/data1/anchor0_dynamics_audit_20261116')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()

def orth(A,tol=1e-10):
    if not A.shape[1]:return A
    U,s,_=np.linalg.svd(A,full_matrices=False);return U[:,s>s[0]*tol]

def subspace(A,B):
    A=orth(A);B=orth(B);s=np.clip(np.linalg.svd(A.T@B,compute_uv=False),0,1)
    return dict(rank_A=A.shape[1],rank_B=B.shape[1],principal_angles_degrees=np.degrees(np.arccos(s)).tolist(),
      matched_sin_Frobenius=float(np.sqrt(np.sum(1-s*s))),
      projector_Frobenius=float(np.sqrt(max(A.shape[1]+B.shape[1]-2*np.sum(s*s),0))),
      unmatched_dimensions=abs(A.shape[1]-B.shape[1]))

def fit(pred,actual):
    a=np.linalg.norm(actual);p=np.linalg.norm(pred)
    return dict(relative_error=float(np.linalg.norm(pred-actual)/max(a,1e-30)),
      cosine=float(np.dot(pred,actual)/max(a*p,1e-30)),predicted_norm=float(p),actual_norm=float(a))

def outside(y,B):return float(np.linalg.norm(y-orth(B)@(orth(B).T@y))/max(np.linalg.norm(y),1e-30)) if B.shape[1] else 1.

def trajectory(R):
    G=R.T@R;e,V=np.linalg.eigh((G+G.T)/2);e=np.maximum(e[::-1],0);total=e.sum()
    n=np.sqrt(np.maximum(np.diag(G),0));corr=G/np.maximum(n[:,None]*n[None,:],1e-30)
    return dict(Gram=G.tolist(),correlation=corr.tolist(),singular_values=np.sqrt(e).tolist(),
      cumulative_energy=(np.cumsum(e)/max(total,1e-30)).tolist(),participation_rank=float(total**2/max(e@e,1e-30)),
      numerical_rank=int(np.sum(e>max(e.max(),1e-30)*1e-12)),
      relative_innovation=[outside(R[:,j],R[:,:j]) for j in range(R.shape[1])],
      scope='Five finite increments, rank at most five; not intrinsic dimension or cross-seed evidence')

def main(out):
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    paths=[R1/f'anchor0_adaptive_step{i:03d}.pt' for i in [0,1]]+[R2/f'checkpoints/step{i:03d}.pt' for i in [2,3,4]]
    hashes=[sha(p) for p in paths];checks=[torch.load(p,map_location='cpu',mmap=True,weights_only=True) for p in paths]
    initial=np.array(json.loads((OLD/'anchor_selection_lock.json').read_text())['selected'][0]['a'])
    with np.load(OLD/'private/anchor0/geometry.npz') as f:
        scales={t:float(np.sqrt(np.trace(f['K_'+t]))) for t in ['DTD','Cars']}
    stored=[r.numpy() for r in checks[-1]['actual_responses']]
    assert len(stored)==5
    index=[];Bs=[];zs=[];before=[];mode_changes=[];parameter_subspaces=[]
    for t,x in enumerate(checks):
        assert x['step']==t and len(x['trajectory'])==t+1
        a=x['coordinates'].numpy();dv=np.array(x['trajectory'][-1]['parameter_update']);prev=initial if t==0 else checks[t-1]['coordinates'].numpy()
        assert np.max(abs(a-dv-prev))<1e-14
        B=np.stack([m['vector'] for m in x['modes']],1);z=np.linalg.lstsq(B,dv,rcond=None)[0]
        residue=float(np.linalg.norm(dv-B@z)/np.linalg.norm(dv));assert residue<1e-9
        before.append(prev.copy());Bs.append(B);zs.append(z)
        oldn=len(checks[t-1]['modes']) if t else 0
        old_fraction=outside(dv,B[:,:oldn])
        index.append(dict(step=t,checkpoint=str(paths[t]),sha256=hashes[t],rank=orth(B).shape[1],
          before_alignment_error=float(np.max(abs(a-dv-prev))),z=z.tolist(),span_residual=residue,
          parameter_update_outside_current_old_modes=old_fraction,
          new_modes=list(range(oldn,len(x['modes']))),supports=[dict(mode=j,driver=m['driver_key'],compensators=m['supports'],source_task=m['source_task']) for j,m in enumerate(x['modes'])]))
        if t:
            parameter_subspaces.append(dict(step=t,scope='same persistent modes only',**subspace(Bs[t-1],B[:,:oldn])))
            for j,old in enumerate(checks[t-1]['modes']):
                new=x['modes'][j];assert old['driver_key']==new['driver_key']
                T=np.array(old['map_matrix']);U=np.array(new['map_matrix']);same=old['supports']==new['supports'];n=np.linalg.norm(T)
                mode_changes.append(dict(step=t,mode=j,driver=old['driver_key'],supports_unchanged=same,
                  direction_change=float(np.linalg.norm(B[:,j]-Bs[t-1][:,j])/np.linalg.norm(Bs[t-1][:,j])),
                  map_absolute_change=float(np.linalg.norm(U-T)) if same else None,
                  map_relative_change=float(np.linalg.norm(U-T)/n) if same and n>1e-14 else None))
    write_json(out/'parameter_dynamics.json',dict(checkpoints=index,subspaces=parameter_subspaces,modes=mode_changes))
    c,s,norm=setup(out);cost=Costs(c)
    assert sha(out/'probes.json')==sha(R2/'probes.json')
    assert sha(out/'independent_probes.json')==sha(R2/'independent_probes.json')
    for x in checks:assert torch.equal(s.point(x['coordinates'].cuda()).cpu(),x['full_visual_flat'])
    oldindex=json.loads((CACHE/'checkpoint_index_audit.json').read_text()) if (CACHE/'checkpoint_index_audit.json').exists() else []
    cache_valid=len(oldindex)==5 and all(x['sha256']==h for x,h in zip(oldindex,hashes)) and sha(CACHE/'probes.json')==sha(out/'probes.json')
    lock=dict(scope='Parameter and actual functional coevolution only; no counterfactual optimization or training',
      full304_refresh=False,coordinate_JVPs=False,expected_mode_task_JVP_requests=50,
      fixed_joint_normalization=scales,normalization_definition='Initial anchor Frobenius norms sqrt(trace(K_D)),sqrt(trace(K_C)); same constants for p0 and p4',
      input_checkpoint_hashes=hashes,cache_checkpoint_alignment_validated=cache_valid,
      p4='Previously used exploratory stability data; finite trajectories only, no new p4 mode JVPs',p6_and_Test_opened=False)
    write_json(out/'diagnostic_lock.json',lock)
    points=[initial]+[x['coordinates'].numpy() for x in checks];measured={}
    for probe in [0,4]:
        values=[]
        for i,a in enumerate(points):
            write_json(out/'status.json',dict(stage='finite_response_check',probe=probe,point=i))
            values.append(all_metrics(c,s.point(torch.tensor(a,device='cuda',dtype=torch.float64)),probe))
        measured[probe]=values
    actual={};spectra={};measurements=[]
    split=len(measured[0][0]['DTD']['phi'].flatten())
    for probe in [0,4]:
        D=np.stack([(measured[probe][t+1]['DTD']['phi']-measured[probe][t]['DTD']['phi']).numpy().ravel() for t in range(5)],1)
        C=np.stack([(measured[probe][t+1]['Cars']['phi']-measured[probe][t]['Cars']['phi']).numpy().ravel() for t in range(5)],1)
        raw=np.r_[D,C];scaled=np.r_[D/scales['DTD'],C/scales['Cars']];actual[probe]=(raw,scaled)
        spectra[str(probe)]={name:trajectory(R) for name,R in [('DTD',D),('Cars',C),('joint_raw',raw),('joint_anchor_scaled',scaled),('joint_unit_columns',scaled/np.maximum(np.linalg.norm(scaled,axis=0),1e-30))]}
        for t in range(5):
            measurements.append(dict(probe=probe,step=t,metrics={task:dict(CE_change=measured[probe][t+1][task]['CE']-measured[probe][t][task]['CE'],
               Top1_change=measured[probe][t+1][task]['Top1']-measured[probe][t][task]['Top1']) for task in ['DTD','Cars']},
               saved_response_error=fit(stored[t],raw[:,t])['relative_error'] if probe==0 else None))
        np.savez_compressed(out/f'actual_trajectory_p{probe}.npz',joint_raw=raw,joint_anchor_scaled=scaled)
    write_json(out/'actual_trajectory_spectra.json',spectra);write_json(out/'finite_measurements.json',measurements)
    explanations=[];functional_subspaces=[];Psis=[];hits=0;misses=0;requests=0
    for t,B in enumerate(Bs):
        point=s.point(torch.tensor(before[t],device='cuda',dtype=torch.float64));parts={}
        for task in ['DTD','Cars']:
            columns=[]
            for j in range(B.shape[1]):
                requests+=1;v=B[:,j];name=hashlib.sha256(v.astype(np.float64).tobytes()).hexdigest()[:24]
                cache=CACHE/f'private/pre{t}/vectors_p0/{task}_{name}.npz'
                if cache_valid and cache.exists():
                    with np.load(cache) as f:r=f['response']
                    hits+=1
                else:
                    write_json(out/'status.json',dict(stage='mode_JVP',step=t,task=task,mode=j,requests=requests,cache_hits=hits,new_JVPs=misses))
                    r,_=c.response(s.expand(torch.tensor(v,device='cuda',dtype=torch.float64)),task,0,point);misses+=1
                columns.append(r)
            parts[task]=np.stack(columns,1)
        Psi=np.r_[parts['DTD']/scales['DTD'],parts['Cars']/scales['Cars']];Psis.append(Psi)
        prediction=np.r_[parts['DTD']@zs[t],parts['Cars']@zs[t]];prediction_scaled=Psi@zs[t]
        oldn=Bs[t-1].shape[1] if t else 0
        explanations.append(dict(step=t,rank=B.shape[1],z=zs[t].tolist(),
          DTD=fit(prediction[:split],actual[0][0][:split,t]),Cars=fit(prediction[split:],actual[0][0][split:,t]),
          joint_raw=fit(prediction,actual[0][0][:,t]),joint_anchor_scaled=fit(prediction_scaled,actual[0][1][:,t]),
          actual_response_outside_current_old_modes=outside(actual[0][1][:,t],Psi[:,:oldn]),
          predicted_response_outside_current_old_modes=outside(prediction_scaled,Psi[:,:oldn])))
        if t:functional_subspaces.append(dict(step=t,scope='Same persistent modes across adjacent parameter points',**subspace(Psis[t-1],Psi[:,:oldn])))
        np.savez_compressed(out/f'mode_response_step{t}.npz',J_D_B=parts['DTD'],J_C_B=parts['Cars'],Psi_anchor_scaled=Psi,z=zs[t])
        write_json(out/'response_explanation.json',explanations)
    assert requests==50 and cost.data['task_JVP_calls']==misses
    write_json(out/'functional_subspaces.json',functional_subspaces)
    history=json.loads((R2/'FAST_FINAL.json').read_text())
    feedback=[dict(step=r['step'],**r['feedback_counterfactual']) for r in history['records'] if 'feedback_counterfactual' in r]
    write_json(out/'historical_feedback.json',dict(cases=feedback,changed_choices=sum(r['different'] for r in feedback),
      scope='r2 steps2..4 logged same-point with/without-history comparisons; r1 gaps not invented'))
    report(out,index,mode_changes,parameter_subspaces,functional_subspaces,explanations,spectra,feedback)
    write_json(out/'COMPLETE.json',dict(done=True,training_started=False,coordinate_JVP_calls=0,full304_refreshes=0,
      mode_task_JVP_requests=requests,mode_JVP_cache_hits=hits,new_mode_task_JVPs=misses,cost=cost.snapshot(),
      wall_seconds_including_input_checks=time.monotonic()-start,original_checkpoint_hashes_unchanged=[sha(p) for p in paths]==hashes))
    write_json(out/'status.json',dict(stage='complete'))

def report(out,index,modes,param,func,explain,spectra,feedback):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps=list(range(5));fig,axs=plt.subplots(2,2,figsize=(12,8))
    axs[0,0].plot(steps,[r['rank'] for r in index],'-o');axs[0,0].set(title='Used control rank (birth-budget limited)',xlabel='Step',ylabel='Rank')
    axs[0,1].plot([r['step'] for r in param],[r['matched_sin_Frobenius'] for r in param],'-o',label='Parameter space')
    axs[0,1].plot([r['step'] for r in func],[r['matched_sin_Frobenius'] for r in func],'-o',label='Functional space')
    axs[0,1].set(title='Persistent modes only: subspace motion',xlabel='Step',ylabel='sin-angle Frobenius');axs[0,1].legend()
    for field,ax,title in [('direction_change',axs[1,0],'Old mode direction changes'),('map_relative_change',axs[1,1],'Old conditional-map changes')]:
        for j in range(7):
            rr=[r for r in modes if r['mode']==j and r[field] is not None]
            if rr:ax.plot([r['step'] for r in rr],[r[field] for r in rr],'-o',label=f'mode {j}')
        ax.set(title=title,xlabel='Step',ylabel='Relative change');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'1_control_evolution.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for probe in ['0','4']:
        for name,style in [('joint_anchor_scaled','-'),('joint_unit_columns','--')]:
            v=spectra[probe][name];axs[0,0].plot(range(1,6),v['cumulative_energy'],style+'o',label=f'p{probe}, {name}')
        axs[0,1].plot(steps,spectra[probe]['joint_anchor_scaled']['relative_innovation'],'-o',label=f'p{probe}')
    axs[0,0].set(title='Finite functional trajectory spectrum',xlabel='Axes (five increments only)',ylabel='Cumulative energy');axs[0,0].legend(fontsize=7)
    axs[0,1].set(title='Actual response outside previous finite history',xlabel='Step',ylabel='Relative innovation');axs[0,1].legend()
    for probe,ax in [('0',axs[1,0]),('4',axs[1,1])]:
        im=ax.imshow(spectra[probe]['joint_anchor_scaled']['correlation'],vmin=-1,vmax=1,cmap='coolwarm');fig.colorbar(im,ax=ax)
        ax.set(title=f'p{probe}: finite response correlation',xlabel='Step',ylabel='Step')
    fig.tight_layout();fig.savefig(out/'2_actual_functional_trajectory.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    for task in ['DTD','Cars','joint_anchor_scaled']:
        axs[0].plot(steps,[r[task]['relative_error'] for r in explain],'-o',label=task)
        axs[1].plot(steps,[r[task]['cosine'] for r in explain],'-o',label=task)
    axs[2].plot(steps,[r['parameter_update_outside_current_old_modes'] for r in index],'-o',label='Parameter increment')
    axs[2].plot(steps,[r['actual_response_outside_current_old_modes'] for r in explain],'-o',label='Actual functional increment')
    axs[2].plot(steps,[r['predicted_response_outside_current_old_modes'] for r in explain],'--o',label='Linear functional increment')
    for ax in axs:ax.set_xlabel('Step');ax.legend(fontsize=7)
    axs[0].set(title='Current-mode tangent prediction error',ylabel='Relative error')
    axs[1].set(title='Predicted vs actual response',ylabel='Cosine')
    axs[2].set(title='Movement requiring newly added modes',ylabel='Fraction outside current old-mode span')
    fig.tight_layout();fig.savefig(out/'3_response_explanation.png',dpi=170);plt.close(fig)
    summary=dict(rank_path=[r['rank'] for r in index],max_parameter_span_error=max(r['span_residual'] for r in index),
      max_old_direction_change=max(r['direction_change'] for r in modes),
      max_old_map_change=max(r['map_relative_change'] for r in modes if r['map_relative_change'] is not None),
      existing_supports_all_unchanged=all(r['supports_unchanged'] for r in modes),
      response_fit=[dict(step=r['step'],error=r['joint_anchor_scaled']['relative_error'],cosine=r['joint_anchor_scaled']['cosine'],new_mode_function_fraction=r['actual_response_outside_current_old_modes']) for r in explain],
      p0_joint_spectrum=spectra['0']['joint_anchor_scaled'],p4_joint_spectrum=spectra['4']['joint_anchor_scaled'],
      history_changed_birth_choices=sum(r['different'] for r in feedback),intrinsic_dimension_identified=False)
    write_json(out/'SUMMARY.json',summary)
    lines=['# 锚点0：低维控制与实际功能共同演化诊断','',
      '本轮没有新seed、训练、304列刷新、模式出生或参数写回。读取五个已接受检查点，模式对齐本次更新前位置；实际增量从原记录恢复并以固定点前向核对。仅请求已有模式上的50次任务级JVP，命中此前已验证缓存则直接复用。','',
      f"控制秩为{summary['rank_path']}；实际参数增量在所用模式空间中的最大相对残差为{summary['max_parameter_span_error']:.3g}。旧模式最大方向变化为{summary['max_old_direction_change']:.4%}，最大条件映射相对变化为{summary['max_old_map_change']:.4%}；已有支持是否全部不变：{summary['existing_supports_all_unchanged']}。",'',
      '## 切向响应是否解释真实变化','',
      '| 步骤 | 秩 | 联合响应相对误差 | 方向余弦 | 实际功能增量在旧模式响应空间之外的比例 |',
      '|---|---:|---:|---:|---:|']
    for r in explain:lines.append(f"|{r['step']}|{r['rank']}|{r['joint_anchor_scaled']['relative_error']:.4%}|{r['joint_anchor_scaled']['cosine']:.6f}|{r['actual_response_outside_current_old_modes']:.4%}|")
    lines+=['','联合响应先按初始锚点的两任务Jacobian Frobenius范数固定缩放再拼接；缩放常数对全部时刻和p4保持一致。原始不缩放谱、单任务谱以及逐列单位化的方向谱同时保留，避免将任务尺度或大步长造成的能量集中误认为普遍低维。实际功能增量的旧空间外分量也可能含非线性误差，需与参数增量和线性预测的旧空间外比例一起判断，不能全部归因于新模式。','',
      '## 实际轨迹谱','',
      '| 探针 | 第一轴能量 | 前两轴能量 | 谱参与率 |','|---|---:|---:|---:|']
    for p in ['0','4']:
        r=spectra[p]['joint_anchor_scaled'];lines.append(f"|p{p}|{r['cumulative_energy'][0]:.4%}|{r['cumulative_energy'][1]:.4%}|{r['participation_rank']:.4f}|")
    lines+=['','只有五个实际时间增量，轨迹矩阵秩上界天然为5。这些谱仅描述该条短轨迹，不能证明稳定内在维数或跨seed规律。p4只重新测量有限变化，没有额外计算p4模式Jacobian。p4此前已使用，是探索性稳定性检查，不是新盲测。','',
      f"r2记录的有历史/无历史反事实中，改变出生选择的次数为{summary['history_changed_birth_choices']}。历史确实进入过算法，不等于已经产生了额外结构选择价值。",'',
      '## 判断边界','',
      '若参数张成残差和切向解释误差很小，支持当前控制模式持续承担并解释了真实更新。非零旧模式漂移、出生后出现独立功能分量，可作为本轨迹的动态结构迹象；但新模式加入本身不能证明旧模式重整必要。秩1→3→5→7→9受出生速率与步数预算限制，不能称为九维收敛流形。冻结/重整/出生的必要性反事实已暂停，尚不作因果结论。完整参数域仍为304维专家重组空间，未验证完整86M参数内在维数。','',
      '## 图','',
      '![控制演化](1_control_evolution.png)','',
      '![实际功能轨迹](2_actual_functional_trajectory.png)','',
      '![响应解释](3_response_explanation.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:main(a.output)
    except Exception:
        import traceback
        write_json(a.output/'FAILURE.json',dict(traceback=traceback.format_exc(),no_training_started=True));raise
