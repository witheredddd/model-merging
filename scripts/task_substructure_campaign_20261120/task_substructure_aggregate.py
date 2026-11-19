"""Aggregate repeated local diagnostics without treating resampling as retraining."""
import argparse,csv,itertools,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np

def hypervolume(points,reference):
    rx,ry=reference;valid=sorted((x,y) for x,y in points if x<=rx and y<=ry)
    if not valid:return 0.
    area=0.;best=ry;previous=valid[0][0]
    for x,y in valid:area+=(x-previous)*(ry-best);best=min(best,y);previous=x
    return area+(rx-previous)*(ry-best)

def main(root):
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True)
    keys=['anchor','family','method','demand','radius_index','sign','alpha','mixture_weight','driver_amplitude','compensated','task_probe']
    metrics=['delta_CE','delta_weighted_CE','delta_Top1','prediction_flip_rate','margin_mean','relative_Log_JVP_error','parameter_norm','case_wall_seconds']
    groups=defaultdict(lambda:defaultdict(list));hvs=[];selection=[]
    runs=sorted(root.glob('large_seed*'))
    for run in runs:
        for folder in sorted((run/'analysis').glob('anchor_*')):
            file=folder/'all_metrics.csv'
            if file.exists():
                with file.open(newline='',encoding='utf8') as f:
                    for row in csv.DictReader(f):
                        key=tuple(row.get(k,'') for k in keys)
                        for metric in metrics:
                            value=row.get(metric,'')
                            try:value=float(value)
                            except (ValueError,TypeError):continue
                            if math.isfinite(value):groups[key][metric].append((run.name,value))
            file=folder/'compatibility.csv'
            if file.exists():
                pools=defaultdict(list)
                with file.open(newline='',encoding='utf8') as f:
                    for row in csv.DictReader(f):
                        pools[(row['anchor'],row['family'],row['method'],row['probe'])].append((float(row['DTD_delta_CE']),float(row['Cars_delta_CE'])))
                for key,points in pools.items():
                    for ref in ((0.,0.),(.01,.01)):
                        hvs.append(dict(seed=run.name,anchor=key[0],family=key[1],method=key[2],probe=key[3],
                            candidates=len(points),reference_DTD=ref[0],reference_Cars=ref[1],
                            hypervolume_CE_nats_squared=hypervolume(points+[(0.,0.)],ref),noop_added_to_every_family=True,
                            original_scale_is_different_budget=key[1]=='expert_increment_original_scale'))
            file=folder/'selection_and_final_evaluation.csv'
            if file.exists():
                with file.open(newline='',encoding='utf8') as f:
                    selection.extend(dict(seed=run.name,**r) for r in csv.DictReader(f))
    def write(name,rows):
        if not rows:return
        cols=list(dict.fromkeys(k for r in rows for k in r))
        with (out/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    rows=[]
    for key,values in sorted(groups.items()):
        for metric,observations in values.items():
            v=np.array([x[1] for x in observations]);seeds=sorted({x[0] for x in observations})
            rows.append(dict(zip(keys,key),metric=metric,n_observations=len(v),n_seed_resamplings=len(seeds),
                mean=float(v.mean()),sample_std=float(v.std(ddof=1)) if len(v)>1 else None,
                minimum=float(v.min()),maximum=float(v.max()),independent_training_trials=False))
    write('resampling_metrics.csv',rows);write('sampled_hypervolumes.csv',hvs);write('selected_controls.csv',selection)
    agreements=[]
    by_anchor=defaultdict(list)
    for run in runs:
        for f in (run/'regions').glob('a*_selected.json'):
            by_anchor[f.stem.split('_')[0]].append((run,f,json.loads(f.read_text())['supports']))
    for anchor,items in by_anchor.items():
        for (r1,f1,s1),(r2,f2,s2) in itertools.combinations(items,2):
            regions=json.loads((r1/'PARAMETER_REGIONS.json').read_text())
            for name in sorted(set(s1)&set(s2)):
                if not isinstance(s1[name],list) or not isinstance(s2[name],list):continue
                x,y=set(s1[name]),set(s2[name]);union=x|y;intersection=x&y
                total=sum(regions[i]['dimension'] for i in union);common=sum(regions[i]['dimension'] for i in intersection)
                agreements.append(dict(anchor=anchor,support_strategy=name,seed1=r1.name,seed2=r2.name,
                    region_Jaccard=len(intersection)/len(union) if union else 1.,
                    parameter_weighted_Jaccard=common/total if total else 1.,
                    support_strategy_not_necessarily_same_members=True))
    write('support_reproducibility.csv',agreements)
    (out/'AGGREGATE_SCOPE.json').write_text(json.dumps(dict(runs=[p.name for p in runs],metric_rows=len(rows),
        input_reports_can_be_partial=True,missing_seeds_not_imputed=True,resamplings_not_independent_training_trials=True,
        hypervolume_references_predeclared=[[0,0],[.01,.01]],hypervolume_is_sampled_local_CE_not_EC_success=True,
        matched_and_original_scale_budgets_separate=True),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
