"""Aggregate all preregistered interventions; no best-radius selection."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    here=Path(__file__).resolve().parent;out=here/'evidence/20261113_mechanisms'
    report=json.loads((out/'report.json').read_text())
    assert report['config']['source_sha256']==hashlib.sha256((here/'mechanisms.py').read_bytes()).hexdigest()
    assert report['config']['training'] is False
    assert [s['state'] for s in report['states']]==[0,32,64,128]
    summary=[]
    for state in report['states']:
        assert len(state['A'])==48 and len(state['B'])==24
        a=[];b=[]
        for radius in report['config']['radii']:
            groups={kind:[r for r in state['A'] if r['radius']==radius and r['kind']==kind]
                    for kind in ('inside','ambient_outside')}
            rows={kind:dict(mean_KL=float(np.mean([r['KL'] for r in rows])),
                           median_actual_over_predicted_KL=float(np.median([r['KL']/r['predicted_KL'] for r in rows])),
                           max_fp32_addition_error=max(r['fp32_addition_error'] for r in rows))
                  for kind,rows in groups.items()}
            a.append(dict(radius=radius,groups=rows))
            groups={kind:[r['CE_delta'] for r in state['B'] if r['radius']==radius and r['kind']==kind]
                    for kind in ('full','inside','outside','random')}
            b.append(dict(radius=radius,groups={kind:dict(mean_CE_delta=float(np.mean(v)),
                min_CE_delta=min(v),max_CE_delta=max(v),improved_batches=sum(x<0 for x in v)) for kind,v in groups.items()}))
        norms=state['B'][0]['gradient_norms']
        summary.append(dict(state=state['state'],A=a,B=b,
            gradient_squared_fraction_inside=norms['inside']**2/norms['full']**2,
            participation_rank=state['C']['participation_rank'],entropy_rank=state['C']['entropy_rank'],
            numerical_rank=state['C']['numerical_rank'],baseline_CE=state['baseline_B']['CE'],
            leakage=state['C']['ambient_leakage_frobenius_ratio'],
            max_Gram_vjp_relative_error=max(state['C']['vjp_vs_response_Gram_relative_errors'])))
    payload=dict(states=summary,relations=report['relations'])
    (out/'summary.json').write_text(json.dumps(payload,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    steps=[s['state'] for s in summary]
    for kind,label in [('inside','Inside fixed10'),('ambient_outside','Isotropic outside')]:
        axes[0].semilogy(steps,[s['A'][0]['groups'][kind]['mean_KL'] for s in summary],'o-',label=label)
    axes[0].set(xlabel='Trajectory step',ylabel='Mean finite KL',title='A: radius 0.001, both signs')
    axes[0].legend(fontsize=8)
    for kind in ('full','inside','outside','random'):
        axes[1].plot(steps,[s['B'][1]['groups'][kind]['mean_CE_delta'] for s in summary],'o-',label=kind)
    axes[1].axhline(0,color='grey',linestyle='--')
    axes[1].set(xlabel='Trajectory step',ylabel='Independent CE change',title='B: radius 0.003, mean of 2 gradients')
    axes[1].legend(fontsize=8)
    axes[2].plot(steps,[s['participation_rank'] for s in summary],'o-',label='Participation rank')
    axes[2].set(xlabel='Trajectory step',ylabel='Effective rank',title='C: response rank / ambient leakage')
    second=axes[2].twinx();second.plot(steps,[s['leakage'] for s in summary],'s--',color='tab:red',label='Leakage')
    second.set(ylabel='|| (I-BB^T) FB ||F / || FB ||F',ylim=(0,1))
    axes[2].legend(loc='lower left',fontsize=8);second.legend(loc='lower right',fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'mechanisms.png',dpi=180);fig.savefig(out/'mechanisms.pdf')
    print(json.dumps(payload))


if __name__=='__main__':main()
