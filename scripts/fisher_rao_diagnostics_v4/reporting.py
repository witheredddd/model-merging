"""Two model-selection estimands and training-seed, rather than sample-only, uncertainty."""
import itertools
import numpy as np
from model import state_to_device
from budget_training import checkpoint_indices


def evaluate_checkpoints(runtime,lift,result,costs):
    rows=[];cache={}
    for cost in costs:
        indices=checkpoint_indices(result,cost);row=dict(training_image_cost=cost)
        for rule,index in indices.items():
            if index not in cache:
                Q,a=state_to_device(result['states'][index],lift.S.device)
                theta=runtime.theta_w+lift.S@lift.coefficients(Q,a).float()
                score=runtime.scores(theta,'Xprime')
                cache[index]=dict(CE=float(score['ce'].mean()),accuracy=float(score['acc'].mean()),
                    per_sample_CE=score['ce'],per_sample_correct=score['acc'],
                    model_training_cost=result['history'][index]['training_images'],
                    Vcal_CE=result['history'][index]['metrics']['CE'])
            row[rule]=cache[index]
        assert row['instant']['model_training_cost']==cost
        rows.append(row)
    return rows


def exact_seed_interval(values):
    """Enumerate paired training-seed bootstrap for only four training seeds.

    The interval is descriptive with this small n; no sample-level replication
    claim is made and seed differences are always retained in the report.
    """
    x=np.asarray(values,dtype=float);n=len(x)
    means=np.array([np.mean(x[list(indices)]) for indices in itertools.product(range(n),repeat=n)])
    return dict(mean=float(x.mean()),per_seed=x.tolist(),training_seed_bootstrap_95=np.quantile(means,[.025,.975]).tolist(),
                n_training_seeds=n,small_seed_count=True)


def aggregate_reports(records,costs,seeds):
    tables=[]
    for rank in sorted({r['rank'] for r in records}):
        for cost in costs:
            for rule in ['instant','selected']:
                methods={}
                for case in sorted({r['case'] for r in records if r['rank']==rank}):
                    found={r['seed']:r for r in records if r['rank']==rank and r['case']==case}
                    if set(found)!=set(seeds):continue
                    values=[next(x for x in found[s]['checkpoints'] if x['training_image_cost']==cost)[rule]['CE'] for s in seeds]
                    methods[case]=exact_seed_interval(values)
                base=f'single_r{rank}_ce'
                gains={case:exact_seed_interval(np.asarray(methods[base]['per_seed'])-np.asarray(row['per_seed']))
                       for case,row in methods.items() if case!=base and base in methods}
                tables.append(dict(rank=rank,cost=cost,rule=rule,CE_by_method=methods,CE_gain_vs_single=gains))
    return dict(tables=tables,seeds=seeds,
        claims='instant is exact cost; selected uses identical Vcal grid; four-seed intervals are descriptive',
        independent_functional_groups_established=False)
