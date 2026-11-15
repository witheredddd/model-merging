"""Fresh labeled local geometry at p0-selected Global EC anchors; no trajectory input."""
import argparse,json,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
OLD=Path('/media/shangxueyuan/data1/compromise_geometry_20261115')
CODE=Path('/home/shangxueyuan/merging/tsv/scripts/compromise_geometry_20261115')
sys.path.insert(0,str(CODE))
from space import context,Space,write,save,sha,norm
from numerics import nondominated


def select(out):
    rows=json.loads((OLD/'linear_Global_EC_archive.json').read_text())
    front=[rows[i] for i in nondominated([r['fitness'] for r in rows])]
    unique={}
    for r in front:unique.setdefault(tuple(r['fitness']),r)
    front=sorted(unique.values(),key=lambda r:r['metrics']['DTD']['accuracy'])
    values=np.array([r['fitness'] for r in front]);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(values,axis=0),axis=1))]
    selected=[]
    for q in [.25,.5,.75]:
        idx=int(np.argmin(abs(arc-q*arc[-1])))
        if front[idx]['id'] not in [r['id'] for r in selected]:selected.append(front[idx])
    for r in front:
        if len(selected)>=min(3,len(front)):break
        if r['id'] not in [x['id'] for x in selected]:selected.append(r)
    write(out/'anchor_selection_lock.json',dict(source=str(OLD/'linear_Global_EC_archive.json'),
        rule='p0 Pareto front,25/50/75% normalized-objective arc positions; first objective-duplicate representative',
        selected=selected,no_p4_p5_selection=True,fit_objective='Unchanged negative normalized top1 Pareto',
        no_failed_trajectory_or_clustering_used=True))
    return selected


def main(out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OLD/'independent_probes.json',out/'independent_probes.json')
    c=context(out);state=torch.load(OLD/'private/space.pt',map_location='cpu',mmap=True,weights_only=True);s=Space(c,state)
    anchors=select(out);R0=.003*norm(s.aD-s.aC)
    write(out/'experiment_lock.json',dict(stage='Conditional functional compensation, DTD protected, Cars CE descent',
        candidate_rank=s.r,tensor_groups=len(s.groups),metric='Parameter Euclidean I; original block support retained',
        radius_primary=R0,radius_sensitivity_factors=[1/3,1,10/3],
        source_norm_fractions=[.01,.1,.3,1.],source_reference='R*||J_D(-h_C/||h_C||)||, h_C in full304 coordinate space',
        source_CE_linear_budget=0.,source_budget_is_squared_for_energy_constraint=True,
        damping='Zero if numerically nonsingular2x2;1e-8*block max eigenvalue if condition>1e10',
        pair_count=11476,directed_pair_count=22952,
        single_control_dimension='at most2',conditional_pair_control_dimension='at most2',
        unconstrained_pair4D_is_upper_bound_only=True,full304_is_upper_bound=True,
        true_fitness='Per-task negative normalized top1 Pareto; CE gradient used only to construct directions',
        data_roles='p0 local geometry and candidate selection;p4 frozen diagnostic;p5 frozen final training-side evaluation',
        p4_p5_previously_consumed_in_prior_stage=True,official_Test_opened=False,
        no_sparse_penalty=True,no_group_count_preset=True,no_trajectory_clustering=True,
        multi_group_expansion='Only after pair/full-space comparison; no automatic grouping or EC-z learning',
        space_path=str(OLD/'private/space.pt'),space_sha256=sha(OLD/'private/space.pt')))
    for k,row in enumerate(anchors):
        directory=c.private/f'anchor{k}';directory.mkdir(exist_ok=True)
        a=torch.tensor(row['a'],device='cuda',dtype=torch.float64);point=s.point(a)
        Q={t:[] for t in ['DTD','Cars']};h={t:[] for t in Q}
        for j in range(s.r):
            c.status(stage='labeled_anchor_Jacobian_cache',anchor=k,anchor_id=row['id'],completed=j,total=s.r,
                     seconds=time.monotonic()-started)
            file=directory/f'column{j:03d}.npz'
            if file.exists():
                with np.load(file) as data:
                    for t in Q:Q[t].append(data['J_'+t]);h[t].append(float(data['h_'+t]))
                continue
            v=s.column(j);data={}
            for t in Q:
                response,slope=c.response(v,t,0,point)
                Q[t].append(response);h[t].append(float(slope.mean()));data['J_'+t]=response;data['h_'+t]=float(slope.mean())
            temp=file.with_suffix('.incomplete.npz');np.savez(temp,**data);temp.replace(file)
        payload={}
        for t in Q:
            J=np.stack(Q[t],1);payload['J_'+t]=J;payload['K_'+t]=J.T@J;payload['h_'+t]=np.array(h[t])
        payload['a']=a.cpu().numpy()
        path=directory/'geometry.npz';temp=path.with_suffix('.incomplete.npz');np.savez(temp,**payload);temp.replace(path)
        write(out/f'anchor{k}_ready.json',dict(index=k,source_id=row['id'],path=str(path),p0_metrics=row['metrics'],seconds=time.monotonic()-started))
    write(out/'cache_ready.json',dict(anchors=len(anchors),seconds=time.monotonic()-started))
    c.status(stage='conditional_geometry_ready',anchors=len(anchors),seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    try:main(a.output)
    except Exception:
        a.output.mkdir(parents=True,exist_ok=True);write(a.output/'cache_failure.json',dict(traceback=traceback.format_exc(),scientific_failure=False));raise
