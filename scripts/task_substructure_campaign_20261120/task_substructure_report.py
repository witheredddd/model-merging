"""Rebuild all-metric tables and a candid coverage report from atomic result files."""
import argparse,csv,json,time,gzip
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--anchor',type=int);a=p.parse_args();root=a.output
out=root/'analysis' if a.anchor is None else root/'analysis'/f'anchor_{a.anchor}';out.mkdir(exist_ok=True,parents=True)
def csvwrite(name,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    if name=='per_class.csv':
        for start in range(0,len(rows),200000):
            with gzip.open(out/(name[:-4]+f'_part{start//200000:04d}.csv.gz'),'wt',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows[start:start+200000])
    else:
        with (out/name).open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
records=[json.loads(p.read_text()) for p in sorted((root/'evaluations').glob('*.json')) if a.anchor is None or f'a{a.anchor}_' in p.name]
rows=[];perclass=[];compat=[]
for r in records:
    base={k:v for k,v in r.items() if isinstance(v,(str,int,float,bool)) or v is None}
    for key,m in r['metrics'].items():
        row=dict(base,task_probe=key)
        row.update({k:v for k,v in m.items() if isinstance(v,(int,float,str,bool)) or v is None})
        for i,q in enumerate(m['margin_quantiles']):row['margin_q'+str(i)]=q
        row.update(paired_CE_lower=m.get('paired_weighted_CE_approx95',[None,None])[0],paired_CE_upper=m.get('paired_weighted_CE_approx95',[None,None])[1])
        rows.append(row)
        for cls,c in m['per_class'].items():perclass.append(dict(case=r['case'],anchor=r['anchor'],task_probe=key,label=cls,**c))
    if r.get('family') in ('compatibility','expert_increment_matched','expert_increment_original_scale','expert_grid_matched','expert_grid_original_scale'):
        for probe in (2,3):
            d=r['metrics'][f'DTD_p{probe}'];c=r['metrics'][f'Cars_p{probe}']
            compat.append(dict(case=r['case'],anchor=r['anchor'],family=r['family'],method=r['method'],probe=probe,
                mixture_weight=r['mixture_weight'],parameter_norm=r['parameter_norm'],DTD_delta_CE=d['delta_CE'],Cars_delta_CE=c['delta_CE'],
                coefficient_DTD=r.get('coefficient_DTD'),coefficient_Cars=r.get('coefficient_Cars'),
                DTD_Top1=d['Top1'],Cars_Top1=c['Top1']))
for x in compat:
    pool=[y for y in compat if y['anchor']==x['anchor'] and y['family']==x['family'] and y['probe']==x['probe'] and y['method']==x['method']]
    x['nondominated_within_same_family']=not any(y['DTD_delta_CE']<=x['DTD_delta_CE'] and y['Cars_delta_CE']<=x['Cars_delta_CE'] and (y['DTD_delta_CE']<x['DTD_delta_CE'] or y['Cars_delta_CE']<x['Cars_delta_CE']) for y in pool)
csvwrite('all_metrics.csv',rows);csvwrite('per_class.csv',perclass);csvwrite('compatibility.csv',compat)
csvwrite('conditional_driver_metrics.csv',[r for r in rows if r['family']=='schur_conditional'])
csvwrite('common_objective_metrics.csv',[r for r in rows if r['family'] in ('fisher_common_objective','naive_freeze_common_objective')])
deletions=[]
for f in sorted((root/'deletions').glob('*.json')):
    if a.anchor is not None and not f.stem.startswith(f'a{a.anchor}_'):continue
    r=json.loads(f.read_text());deletions.append(dict(case=f.stem,proxy_deletion=r['approximate_proxy_deletion'],
        lower=r['certified_deletion_interval'][0],upper=r['certified_deletion_interval'][1],
        schur_energy=r['schur_energy_from_solution_difference'],identity_discrepancy=r['schur_identity_discrepancy'],
        compensation_proxy_gain=r['compensation_proxy_gain'],compensation_parameter_norm=r['compensation_parameter_norm'],
        restricted_stationarity=r['restricted']['relative_stationarity'],full_stationarity=r['full']['relative_stationarity'],
        full_q=r['full']['q'],restricted_q=r['restricted']['q'],full_norm=r['full_norm'],restricted_norm=r['restricted_norm']))
csvwrite('deletions.csv',deletions)
regions=[]
for f in sorted((root/'regions').glob('a[0-9]*.json')):
    if a.anchor is not None and f.stem!=f'a{a.anchor}':continue
    value=json.loads(f.read_text())
    if isinstance(value,dict):value=value.get('regions',[])
    if isinstance(value,list):regions.extend(dict(anchor=int(f.stem[1:]),**r) for r in value)
csvwrite('regions.csv',regions)
ranks=[]
for f in sorted((root/'bases').glob('a[0-9]*.json')):
    if a.anchor is not None and f.stem!=f'a{a.anchor}':continue
    r=json.loads(f.read_text());vals=np.array(r['singular_values_squared']);total=vals.sum()
    for i,tail in enumerate(r['proxy_tail_losses']):ranks.append(dict(anchor=int(f.stem[1:]),rank=i,
        proxy_tail=tail,proxy_energy_retained=1-2*tail/total,sample_rank_bound=r['sample_demand_rank_bound'],
        actual_resolved_basis_rank=len(r['basis']),rank_above_resolved_is_redundant=i>len(r['basis']),
        not_intrinsic_dimension=True))
csvwrite('rank_curves.csv',ranks)
# Choose within each rank family using selection data; never choose by final evaluation.
selections=[]
for anchor in sorted({r['anchor'] for r in records}):
    for demand in sorted({r['demand'] for r in records if r['anchor']==anchor}):
        pool=[r for r in records if r['anchor']==anchor and r['demand']==demand and r.get('family')=='sufficiency' and r.get('sign')==1 and r.get('radius_index')==1]
        if pool:
            best=min(pool,key=lambda r:r['metrics']['DTD_p2']['weighted_CE'])
            selections.append(dict(anchor=anchor,demand=demand,selected_on_probe2=best['method'],
                selection_delta=best['metrics']['DTD_p2']['delta_weighted_CE'],final_evaluation_delta=best['metrics']['DTD_p3']['delta_weighted_CE']))
csvwrite('selection_and_final_evaluation.csv',selections)
status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
complete=status.get('stage')=='COMPLETE' and (root/'SUPPLEMENT_COMPLETE.json').exists()
summary=dict(captured_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),complete=complete,status=status,
    evaluated_parameter_states=len(records),metric_rows=len(rows),per_class_rows=len(perclass),conditional_deletions=len(deletions),
    anchors_evaluated=sorted({r['anchor'] for r in records}),selections=selections,
    official_test_opened=False,full_validation_80_percent_target='Not evaluated by this local diagnostic',
    new_controlled_expert_training='Not performed',EC_Pareto_improvement='Not established by sampled local compatibility probes',
    metrics_scope='All raw case metrics, including harm, flips and per-class outcomes, retained; no CE acceptance gate')
(out/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
text=f'''# Task-driven parameter substructure diagnostic\n\nStatus: {'COMPLETE' if complete else 'PARTIAL'}; {len(records)} physical parameter evaluations, {len(deletions)} conditional deletion comparisons.\n\nAll scalar outcomes: `all_metrics.csv`; all class counts and metrics: `per_class_part*.csv.gz`; conditional objective gaps and residual certificates: `deletions.csv`; every native region: `regions.csv`; every resolved rank: `rank_curves.csv`; signed sampled compatibility: `compatibility.csv`. Raw JSON/NPZ retain paired per-image observations.\n\nBasis construction (p0), control selection (p2), final evaluation (p3) are disjoint image sets. Probes are training-side development data, not an official test set and not claimed unseen during expert training. Rank is bounded by the finite demand family and is not intrinsic dimension. Fixed physical radius comparisons are direction tests, not fixed-response protection optima. Fisher deletion gaps concern the quadratic proxy, not CE. Compatibility is sampled, not a globally certified nonlinear Pareto frontier.\n\nNo new controlled expert was trained in this diagnostic, so Val >=80% and EC search superiority remain untested. Negative, zero and uncertain outcomes are retained. Source tests and a successful finite step are not evidence of necessary groups or a manifold.\n'''
(out/'REPORT.md').write_text(text,encoding='utf8')
def fmt(v):return 'unresolved' if v is None else f'{v:.6g}'
with (out/'REPORT.md').open('a',encoding='utf8') as f:
    f.write('\n## Fixed-radius sufficiency and full-complement controls\n\n')
    f.write('| Anchor | Demand | Family | Method | DTD weighted CE change | Approx 95% interval | DTD Top-1 change | Cars CE change | Cars Top-1 change | Held-out Log/JVP error |\n|---|---|---|---|---:|---|---:|---:|---:|---:|\n')
    for r in records:
        if r.get('radius_index')!=1 or r.get('sign',1)!=1:continue
        if r['family'] not in ('sufficiency','motion_complement','euclidean_freeze'):continue
        d=r['metrics']['DTD_p3'];c=r['metrics']['Cars_p3'];ci=d['paired_weighted_CE_approx95']
        f.write(f'|{r["anchor"]}|{r["demand"]}|{r["family"]}|{r["method"]}|{fmt(d["delta_weighted_CE"])}|[{fmt(ci[0])}, {fmt(ci[1])}]|{fmt(d["delta_Top1"])}|{fmt(c["delta_CE"])}|{fmt(c["delta_Top1"])}|{fmt(d.get("relative_Log_JVP_error"))}|\n')
    f.write('\n## Conditional proxy gaps and numerical certificates\n\n')
    f.write('| Case | Proxy deletion | Lower | Upper | Compensation proxy gain | Restricted residual |\n|---|---:|---:|---:|---:|---:|\n')
    for r in deletions:f.write(f'|{r["case"]}|{fmt(r["proxy_deletion"])}|{fmt(r["lower"])}|{fmt(r["upper"])}|{fmt(r["compensation_proxy_gain"])}|{fmt(r["restricted_stationarity"])}|\n')
    f.write('\nFreezing the span used to construct a training gradient can trivially remove that same gradient. This is a construction identity, not evidence of irreplaceable physical parameters. Independent demands, independent images, later positions and actual finite outcomes must support any stronger interpretation. The arithmetic-resolution test is a numerical safeguard, not a statistical discovery threshold.\n')
    f.write('\nFisher comparisons include common-objective, budget-feasible native scaling as well as individually normalized direction probes. These are distinct experiments: use `fisher_common_objective` to compare the same quadratic objective, and `fisher_freeze` only to compare directions at equal parameter norm. The unscaled quadratic optimum need not lie in a region where the nonlinear CE approximation is accurate; inspect recorded finite response errors. Approximate paired intervals do not include a multiple-comparison correction.\n')
    f.write('\nNumerical residual bounds and deletion intervals are algebraic bounds for an ideal fixed Jacobian and exact projector. Floating-point operator error, uncertainty in data sampling and error of the nonlinear quadratic approximation are not enclosed by these bounds. The projected-gradient arithmetic floor is a conservative numerical heuristic, not a formally verified interval.\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('selections','status')},indent=2))
if root.name.startswith('large_seed'):
    config=json.loads((root/'CONFIG.json').read_text())
    if a.anchor==config['anchors'][-1]:
        from task_substructure_aggregate import main as aggregate
        aggregate(root.parent)
