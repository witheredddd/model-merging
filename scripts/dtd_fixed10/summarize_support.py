"""Standalone descriptive plots; no group discovery or combined importance score."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    here=Path(__file__).resolve().parent;out=here/'evidence/20261114_support'
    report=json.loads((out/'report.json').read_text())
    assert report['config']['source_sha256']==hashlib.sha256((here/'support_geometry.py').read_bytes()).hexdigest()
    assert report['config']['training'] is False
    first=report['probes'][0];scores=first['coordinate_scores'];groups=first['matrix_scores']
    fractions=np.array([.00001,.0001,.001,.01,.1])
    fig,ax=plt.subplots(figsize=(7,4.2),constrained_layout=True)
    for key,label in [('ellB','B leverage'),('ellQ','Q leverage'),('g','Functional access'),('rho','Absolute rho'),('a_old','Absolute alignment: old update'),('a_expert','Absolute alignment: external update')]:
        values=scores[key]['top_fraction_mass'];ax.semilogx(100*fractions,[100*values[str(f)] for f in fractions],'o-',label=label)
    ax.set(xlabel='Top fraction of original coordinates (%)',ylabel='Fraction of absolute score mass (%)',title='Fixed10 coordinate concentration (probe0)')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.savefig(out/'concentration.png',dpi=180);plt.close(fig)
    labels=[str(g['layer'])+':'+g['name'].split('.')[-2] for g in groups]
    null=np.array([[g['enrichment'][k] for k in ['observed','null_2p5','null_97p5']] for g in groups])
    fig,ax=plt.subplots(figsize=(13,4),constrained_layout=True);x=np.arange(48)
    ax.vlines(x,null[:,1],null[:,2],color='grey',alpha=.65,label='1000 within-Q random10: central95%')
    ax.plot(x,null[:,0],'o',markersize=4,label='Observed fixed10')
    ax.axhline(1,color='black',linestyle='--');ax.set_xticks(x,labels,rotation=90,fontsize=6)
    ax.set(ylabel='Leverage enrichment relative to Q',title='Predefined matrix bins, not discovered groups')
    ax.legend(fontsize=8);fig.savefig(out/'enrichment.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(12,5),constrained_layout=True)
    for ax,key,title in zip(axes,['ellB','g','rho','a_expert'],['Representation leverage','Unit-coordinate access','Signed energy allocation','Signed expert-update alignment']):
        total=sum(g['scores'][key] for g in groups)
        values=np.array([g['scores'][key]/g['numel']/(total/84934656) for g in groups]).reshape(12,4)
        if key in ('ellB','g'):
            image=ax.imshow(np.log10(np.maximum(values,1e-8)),aspect='auto',cmap='viridis')
            label='log10 relative per-coordinate density'
        else:
            image=ax.imshow(values,aspect='auto',cmap='coolwarm',vmin=-np.abs(values).max(),vmax=np.abs(values).max());label='signed relative per-coordinate density'
        ax.set(title=title,ylabel='Transformer block');ax.set_xticks(range(4),['QKV','attn out','MLP in','MLP out'],rotation=60)
        fig.colorbar(image,ax=ax,shrink=.6,label=label)
    fig.savefig(out/'distributions.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    G=np.array(first['Gamma']);diag=np.sqrt(np.diag(G));cos=G/np.maximum(diag[:,None]*diag[None,:],1e-30)
    im=axes[0].imshow(cos,vmin=-1,vmax=1,cmap='coolwarm');fig.colorbar(im,ax=axes[0],shrink=.7,label='Masked-response Frobenius cosine')
    axes[0].set(title='Joint response coupling: 48 matrix bins',xlabel='Matrix index',ylabel='Matrix index')
    deletion=[d['relative_energy_change'] for d in first['direction_masking_energy_changes']]
    axes[1].bar(np.arange(48),deletion);axes[1].axhline(0,color='grey')
    axes[1].set(title='Local removal effect on ||JB||F squared',xlabel='Matrix index',ylabel='(2 rho_G - ||R_G||F squared) / total')
    fig.savefig(out/'coupling.png',dpi=180);plt.close(fig)
    summary=dict(top1percent_mass={k:v['top_fraction_mass']['0.01'] for k,v in scores.items()},
        ellB_g_top1percent_overlap=first['top1percent_overlap']['ellB__g'],
        rho_negative_absolute_fraction=-scores['rho']['negative_sum']/scores['rho']['absolute_sum'],
        raw_energy=first['total_response_energy'],sum_matrix_self_energy=float(np.trace(G)),
        total_cross_energy=float(G.sum()-np.trace(G)),
        negative_cross_pairs=int((np.triu(G,1)<0).sum()),
        layer_or_matrix_groups_are_not_discovered=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
