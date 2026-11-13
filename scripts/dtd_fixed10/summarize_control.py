"""Summarize all fixed-budget controls, retaining all radii and seeds."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    here=Path(__file__).resolve().parent;out=here/'evidence/20261114_control'
    r=json.loads((out/'report.json').read_text())
    assert hashlib.sha256((here/'masked_control.py').read_bytes()).hexdigest()==r['config']['source_sha256']
    assert r['config']['fusion_model_used'] is False and r['config']['training'] is False
    summaries=[]
    for method in r['methods']:
        assert method['budget']==849346 and sum(method['matrix_counts'])==849346
        assert len(method['response_control'])==32 and len(method['task_control'])==6
        for controller in ('selected_min_norm','full_response_fit'):
            for radius in (.001,.003):
                rows=[x for x in method['response_control'] if x['controller']==controller and x['radius']==radius]
                fields=['selected_target_relative_error','full_target_relative_error','outside_over_target_norm','KL_target_to_control']
                metrics={split:{field:float(np.mean([x['metrics'][split][field] for x in rows])) for field in fields} for split in ['fit','held']}
                summaries.append(dict(method=method['method'],controller=controller,radius=radius,metrics=metrics,
                    mean_parameter_norm=float(np.mean([x['applied_parameter_norm'] for x in rows])),capped_cases=sum(x['scale']<1 for x in rows)))
    task=[]
    for method in r['methods']:
        for radius in (.001,.003,.01):
            selected_rows=[x for x in method['task_control'] if x['radius']==radius]
            values=[x['metrics']['held']['CE_delta'] for x in selected_rows]
            task.append(dict(method=method['method'],radius=radius,mean_CE_delta=float(np.mean(values)),
                min_CE_delta=min(values),max_CE_delta=max(values),
                mean_KL_from_base=float(np.mean([x['metrics']['held']['KL_from_base'] for x in selected_rows])),
                mean_outside_response_energy_fraction=float(np.mean([
                    x['metrics']['held']['outside_response_norm']**2/(x['metrics']['held']['outside_response_norm']**2+
                    x['metrics']['held']['selected_response_norm']**2) for x in selected_rows]))))
    (out/'summary.json').write_text(json.dumps(dict(response=summaries,task=task),indent=2))
    names=[m['method'] for m in r['methods']];fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    x=np.arange(len(names));width=.38
    for j,controller in enumerate(('selected_min_norm','full_response_fit')):
        rows=[next(s for s in summaries if s['method']==name and s['controller']==controller and s['radius']==.003) for name in names]
        for ax,field,title in [(axes[0],'full_target_relative_error','Held full-response target error'),(axes[1],'outside_over_target_norm','Held outside response / target')]:
            ax.bar(x+(j-.5)*width,[s['metrics']['held'][field] for s in rows],width,label=controller)
            ax.set(title=title,ylabel='Ratio (lower is better)')
    values=[next(s['mean_CE_delta'] for s in task if s['method']==name and s['radius']==.003) for name in names]
    axes[2].bar(x,values);axes[2].axhline(0,color='grey');axes[2].set(title='Held CE change: masked gradient',ylabel='CE delta (negative improves)')
    for ax in axes:ax.set_xticks(x,names,rotation=75,fontsize=7);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=7);fig.savefig(out/'control.png',dpi=180);fig.savefig(out/'control.pdf')
    print(json.dumps(dict(response_at003=[s for s in summaries if s['radius']==.003],task_at003=[s for s in task if s['radius']==.003])))


if __name__=='__main__':main()
