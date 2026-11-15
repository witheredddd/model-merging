"""All tensor-pair compensation programs, matched controls and frozen finite evaluation."""
import argparse,csv,json,shutil,time,traceback
from pathlib import Path
import numpy as np
import torch
from cache import context,Space,write,save,OLD,norm
from program import solve,conditional
from torch.nn import functional as F


def restricted(KD,KC,hD,hC,ids,W):
    return (W.T@KD[np.ix_(ids,ids)]@W,W.T@KC[np.ix_(ids,ids)]@W,W.T@hD[ids],W.T@hC[ids])


def embed(r,ids,W,x):
    v=np.zeros(r);v[ids]=W@x;return v


def ci(x):
    rng=np.random.default_rng(202611160130);x=np.asarray(x)
    return np.quantile(x[rng.integers(len(x),size=(1000,len(x)))].mean(1),[.025,.975]).tolist()


def screen(out,k,groups,R):
    directory=out/'private'/f'anchor{k}'
    with np.load(directory/'geometry.npz') as f:data={key:f[key] for key in f.files}
    KD=data['K_DTD'];KC=data['K_Cars'];hd=data['h_DTD'];hc=data['h_Cars'];r=len(hc)
    Jd=data['J_DTD'];Jc=data['J_Cars'];unit=-hc/max(np.linalg.norm(hc),1e-30)
    source_scale=float(np.linalg.norm(Jd@unit));fractions=[.01,.1,.3,1.]
    sing=np.linalg.svd(Jd,compute_uv=False)
    write(out/f'anchor{k}_geometry_audit.json',dict(rank=r,source_singular_values=sing.tolist(),
          CE_gradient_norms=dict(DTD=float(np.linalg.norm(hd)),Cars=float(np.linalg.norm(hc))),
          source_reference=source_scale,definition='Labeled CE gradients; predictive Fisher/GN curvature, not exact CE Hessian'))
    singles={};screen_rows=[];selected=[]
    for tau in fractions:
        ell=R*tau*source_scale;best=None
        for g in groups:
            ids=g['indices'];W=np.eye(len(ids));x,audit=solve(*restricted(KD,KC,hd,hc,ids,W),R,ell)
            row=dict(kind='single',groups=[g['key']],ids=ids,W=W.tolist(),x=x.tolist(),audit=audit,tau=tau)
            if audit['passed'] and (best is None or audit['predicted_Cars_CE']<best['audit']['predicted_Cars_CE']):best=row
        assert best is not None
        singles[tau]=best
    # Every directed pair is examined. Only source compensation, never cluster assignments.
    bestpairs={tau:[] for tau in fractions};auditcounts=dict(programs=0,passed=0)
    pairrows=[]
    for m,gm in enumerate(groups):
        write(out/f'anchor{k}_screen_status.json',dict(completed_drivers=m,total=len(groups)))
        for n,gn in enumerate(groups):
            if n==m:continue
            im=gm['indices'];jn=gn['indices'];ids=im+jn;W,operator=conditional(KD,im,jn)
            block=restricted(KD,KC,hd,hc,ids,W)
            for tau in fractions:
                ell=R*tau*source_scale;x,audit=solve(*block,R,ell)
                auditcounts['programs']+=1;auditcounts['passed']+=int(audit['passed'])
                parts=W@x;xm=parts[:len(im)];xn=parts[len(im):]
                B=-2*float(xm@KD[np.ix_(im,jn)]@xn)
                score=singles[tau]['audit']['predicted_Cars_CE']-audit['predicted_Cars_CE']
                pairrows.append([m,n,tau,audit['predicted_Cars_CE'],score,B,audit['source_response'],audit['cost'],int(audit['passed'])])
                if audit['passed']:
                    row=dict(kind='conditional_pair',groups=[gm['key'],gn['key']],group_indices=[m,n],ids=ids,
                      W=W.tolist(),x=x.tolist(),audit=audit,tau=tau,operator=operator,
                      gain_beyond_best_single=score,DTD_cancelled_energy=B,driver_dimension=len(im))
                    candidates=bestpairs[tau];candidates.append(row)
                    candidates.sort(key=lambda z:z['gain_beyond_best_single'],reverse=True)
                    unique=[];seen=set()
                    for item in candidates:
                        key=tuple(sorted(item['group_indices']))
                        if key not in seen:unique.append(item);seen.add(key)
                        if len(unique)>=2:break
                    bestpairs[tau]=unique
        if m%10==0:write(out/f'anchor{k}_program_audit.json',auditcounts)
    np.savez_compressed(directory/'all_pair_scores.npz',values=np.asarray(pairrows),
        columns=np.array(['driver','compensator','source_fraction','predicted_Cars_CE','gain_beyond_best_single','cancelled_DTD_energy','source_response','cost','audit_passed']))
    write(out/f'anchor{k}_program_audit.json',auditcounts)
    for tau in fractions:
        selected.append(singles[tau]);selected+=bestpairs[tau]
    write(out/f'anchor{k}_selected_pair_lock.json',dict(selected=selected,source_reference=source_scale,
           selection_data='p0 cached geometry only',p4_p5_seen=False,
           score='Predicted Cars objective gain over best of all152 single tensors at same budgets',
           positive_gain_or_positive_cancellation_not_assumed=True,all_pairs_evaluated=True))
    return data,selected,source_scale


def main(out,minutes):
    started=time.monotonic();deadline=time.time()+60*minutes
    while not (out/'cache_ready.json').exists():
        if (out/'cache_failure.json').exists():raise RuntimeError('Anchor cache failed')
        if time.time()>deadline:raise TimeoutError('Cache wait budget')
        time.sleep(5)
    deadline=time.time()+60*minutes
    c=context(out);state=torch.load(OLD/'private/space.pt',map_location='cpu',weights_only=True,mmap=True);s=Space(c,state)
    for file in Path(__file__).parent.glob('*.py'):
        (out/'source_archive').mkdir(exist_ok=True);shutil.copy2(file,out/'source_archive'/file.name)
    groups=[g for g in s.groups if g['indices']];anchors=json.loads((out/'anchor_selection_lock.json').read_text())['selected']
    R=.003*norm(s.aD-s.aC);all_models=[];baselines={}
    for k,anchor in enumerate(anchors):
        c.status(stage='enumerating_conditional_tensor_pairs',anchor=k,pairs=11476,directions=22952)
        data,selected,source_scale=screen(out,k,groups,R)
        KD=data['K_DTD'];KC=data['K_Cars'];hd=data['h_DTD'];hc=data['h_Cars'];a0=torch.tensor(data['a'],device='cuda',dtype=torch.float64)
        source_factor=np.linalg.qr(data['J_DTD'],mode='r');models=[]
        def add(ident,kind,v,**kw):
            models.append(dict(id=f'a{k}_{ident}',kind=kind,v=np.asarray(v).tolist(),anchor=k,**kw))
        add('baseline','anchor',np.zeros(s.r))
        # All candidates and comparison families fixed before p4 or p5 evaluation.
        for tau in [.01,.1,.3,1.]:
            for factor in [1/3,1.,10/3]:
                radius=R*factor;ell=radius*tau*source_scale
                x,audit=solve(KD,KC,hd,hc,radius,ell,source_factor=source_factor)
                add(f'full_t{tau}_r{factor:g}','full304_upper',x,tau=tau,radius=radius,ell=ell,audit=audit)
                G=s.global_basis.cpu().numpy()
                x,audit=solve(G.T@KD@G,G.T@KC@G,G.T@hd,G.T@hc,radius,ell)
                add(f'global_t{tau}_r{factor:g}','global2D_local',G@x,tau=tau,radius=radius,ell=ell,audit=audit)
        for j,row in enumerate(selected):
            ids=row['ids'];W=np.asarray(row['W']);tau=row['tau']
            for factor in [1/3,1.,10/3]:
                radius=R*factor;ell=radius*tau*source_scale
                x,audit=solve(*restricted(KD,KC,hd,hc,ids,W),radius,ell)
                v=embed(s.r,ids,W,x)
                add(f'selected{j}_r{factor:g}',row['kind'],v,tau=tau,radius=radius,ell=ell,audit=audit,
                    groups=row['groups'],selection_index=j,operator=row.get('operator'),control_dimension=W.shape[1])
                if row['kind']=='conditional_pair' and factor==1:
                    m,n=row['group_indices'];im=groups[m]['indices'];jn=groups[n]['indices']
                    for side,index in [('driver',im),('compensator',jn)]:
                        part=np.zeros(s.r);part[index]=v[index]
                        add(f'selected{j}_{side}_only','component_ablation',part,tau=tau,radius=radius,ell=ell,
                            parent=f'a{k}_selected{j}_r1',component=side,groups=row['groups'])
                    x4,audit4=solve(KD[np.ix_(ids,ids)],KC[np.ix_(ids,ids)],hd[ids],hc[ids],radius,ell)
                    add(f'selected{j}_4D','joint4D_upper',embed(s.r,ids,np.eye(len(ids)),x4),tau=tau,
                        radius=radius,ell=ell,audit=audit4,groups=row['groups'],control_dimension=len(ids))
                    matched_R=max(float(np.linalg.norm(v)),R*1e-8);best_match=None
                    for single in groups:
                        si=single['indices'];SW=np.eye(len(si))
                        sx,sa=solve(*restricted(KD,KC,hd,hc,si,SW),matched_R,ell)
                        if sa['passed'] and (best_match is None or sa['predicted_Cars_CE']<best_match[2]['predicted_Cars_CE']):
                            best_match=(si,sx,sa,single['key'])
                    if best_match is not None:
                        si,sx,sa,key=best_match
                        add(f'selected{j}_costmatched_single','matched_single',embed(s.r,si,np.eye(len(si)),sx),
                            tau=tau,radius=matched_R,ell=ell,audit=sa,groups=[key],control_dimension=len(si),
                            parent=f'a{k}_selected{j}_r1')
        write(out/f'anchor{k}_frozen_models.json',dict(models=models,selection='p0 only',p4_p5_used_to_select=False,
             nonlinear_relinearization=False,actual_prediction_validation_only=True))
        all_models+=models
    # Freeze numerical materialization scales before evaluating any probe. Component
    # ablations inherit the parent's scale, so they sum to the intended joint motion.
    parent_scales={}
    for model in all_models:
        if model['kind']=='component_ablation':continue
        k=model['anchor'];a0=torch.tensor(anchors[k]['a'],device='cuda',dtype=torch.float64)
        base=s.point(a0);v=torch.tensor(model['v'],device='cuda',dtype=torch.float64);scale=1.
        for unused in range(6):
            cost=norm(s.point(a0+v).double()-base.double())
            if 'radius' not in model or cost<=model['radius']*(1+1e-8):break
            factor=model['radius']/max(cost,1e-30)*.9999;v*=factor;scale*=factor
        model['v']=v.cpu().tolist();model['frozen_materialization_scale']=scale;parent_scales[model['id']]=scale
    for model in all_models:
        if model['kind']=='component_ablation':
            scale=parent_scales[model['parent']];model['v']=(np.array(model['v'])*scale).tolist()
            model['frozen_materialization_scale']=scale
    write(out/'all_candidates_frozen.json',dict(models=all_models,stage='before any finite p4/p5 prediction',
      no_second_control_growth_or_clustering=True))
    results=[];effects={};pair_audits=[]
    for k,anchor in enumerate(anchors):
        with np.load(c.private/f'anchor{k}'/'geometry.npz') as f:data={key:f[key] for key in f.files}
        a0=torch.tensor(data['a'],device='cuda',dtype=torch.float64);point0=s.point(a0)
        models=[m for m in all_models if m['anchor']==k]
        for p in [0,4,5]:
            reference={}
            for t in ['DTD','Cars']:
                z,y=c.baseline(t,p,point0);reference[t]=(z,y,F.cross_entropy(z,y,reduction='none').numpy())
            for row in models:
                if time.time()>deadline:raise TimeoutError('Finite validation time budget')
                c.status(stage='finite_conditional_compensation',anchor=k,probe=p,point=row['id'])
                v=torch.tensor(row['v'],device='cuda',dtype=torch.float64);point=s.point(a0+v)
                actual_cost=norm(point.double()-point0.double());radial_scale=1.
                for unused in range(6):
                    if 'radius' not in row or actual_cost<=row['radius']*(1+1e-8):break
                    factor=row['radius']/max(actual_cost,1e-30)*.9999;v*=factor;radial_scale*=factor
                    point=s.point(a0+v);actual_cost=norm(point.double()-point0.double())
                metrics={}
                for t in ['DTD','Cars']:
                    z0,y,loss0=reference[t];z,_=c.baseline(t,p,point)
                    loss=F.cross_entropy(z,y,reduction='none').numpy();delta=loss-loss0
                    phi0=2*z0.softmax(-1).sqrt()/np.sqrt(len(y));phi=2*z.softmax(-1).sqrt()/np.sqrt(len(y))
                    metrics[t]=dict(CE=float(loss.mean()),CE_change=float(delta.mean()),CE_change_CI95=ci(delta),
                        accuracy=float((z.argmax(-1)==y).double().mean()),baseline_accuracy=float((z0.argmax(-1)==y).double().mean()),
                        KL=float((z0.softmax(-1)*(z0.log_softmax(-1)-z.log_softmax(-1))).sum(-1).mean()),
                        finite_phi_change_norm=float((phi-phi0).norm()))
                    save(c.private/f'finite_p{p}_{row["id"]}_{t}.pt',dict(logits=z,labels=y,loss=torch.from_numpy(loss)))
                result=dict(id=row['id'],kind=row['kind'],anchor=k,probe=p,groups=row.get('groups'),
                  tau=row.get('tau'),radius=row.get('radius'),ell=row.get('ell'),
                  parameter_norm=norm(v),materialized_norm=actual_cost,materialization_radial_scale=radial_scale,metrics=metrics,
                  normalized_top1_fitness=[-metrics[t]['accuracy']/value for t,value in [('DTD',.8209219858156028),('Cars',.8648648648648649)]])
                results.append(result);write(out/'finite_results.json',results)
                if row['kind']=='conditional_pair' and row['id'].endswith('_r1'):
                    g0=next(g for g in groups if g['key']==row['groups'][0]);g1=next(g for g in groups if g['key']==row['groups'][1])
                    vm=torch.zeros_like(v);vn=torch.zeros_like(v);vm[g0['indices']]=v[g0['indices']];vn[g1['indices']]=v[g1['indices']]
                    rm,unused=c.response(s.expand(vm),'DTD',p,point0);rn,unused=c.response(s.expand(vn),'DTD',p,point0)
                    rc,unused=c.response(s.expand(v),'Cars',p,point0)
                    global_response=np.stack([c.response(s.expand(s.global_basis[:,j]),'Cars',p,point0)[0] for j in range(2)],1)
                    coef=np.linalg.lstsq(global_response,rc,rcond=1e-8)[0];residual=rc-global_response@coef
                    pair_audits.append(dict(id=row['id'],anchor=k,probe=p,cancelled_DTD_energy=float(-2*rm@rn),
                        separate_DTD_energy=float(rm@rm+rn@rn),combined_DTD_energy=float((rm+rn)@(rm+rn)),
                        source_cancellation_fraction=float(-2*rm@rn/max(rm@rm+rn@rn,1e-30)),
                        Cars_response_norm=float(np.linalg.norm(rc)),Cars_response_independent_energy_fraction=float(residual@residual/max(rc@rc,1e-30)),
                        parameter_new_fraction=s.novelty(v)['new_fraction']))
                    write(out/'independent_cancellation_audit.json',pair_audits)
    report(out,results,pair_audits)
    c.status(stage='conditional_compensation_complete',seconds=time.monotonic()-started,anchors=len(anchors),models=len(all_models))


def report(out,rows,audits):
    lines=['# 条件功能补偿先导实验','',
      '不依赖旧未收敛轨迹；各锚点由p0线性Global EC Pareto选择。所有11500量级有向补偿映射从新的锚点Jacobian生成。',
      '单张量与条件双张量均最多2维控制；联合4维和全304维只作能力上界。真实fitness保持各任务负归一化top1。',
      'p4/p5为前阶段已用过的训练侧留出；本轮候选在查看它们之前冻结，不称全新盲测。',
      '预测Cars改善、DTD能量抵消、实际CE/Accuracy/KL及相对全局专家空间的新增响应需共同判读，不以增加支持数量本身作为补偿证据。','',
      '|锚点|类型|p5评估点数|最小Cars CE变化|','|---|---|---:|---:|']
    for k in sorted(set(r['anchor'] for r in rows)):
        for kind in ['single','conditional_pair','joint4D_upper','global2D_local','full304_upper']:
            rr=[r for r in rows if r['probe']==5 and r['anchor']==k and r['kind']==kind]
            if rr:lines.append(f"|{k}|{kind}|{len(rr)}|{min(r['metrics']['Cars']['CE_change'] for r in rr):.7g}|")
    lines+=['','上述最小值只为描述，跨预算和任务源损害不同，不是未经约束的胜负结论。完整逐点源预算、DTD CE漂移和配对数据见finite_results.json。',
       '本轮执行一次局部候选构造和多尺度有限验证，不声称完成非线性重求补偿或多张量结构增长。']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--minutes',type=float,default=240);a=p.parse_args()
    try:main(a.output,a.minutes)
    except Exception:
        write(a.output/'run_failure.json',dict(traceback=traceback.format_exc(),scientific_failure=False));raise
