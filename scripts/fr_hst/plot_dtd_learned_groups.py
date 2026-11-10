"""Plot matched controls, group relations, birth contrasts and regrouping."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def errorbars(ax, x, values, **kwargs):
    y = np.array([v['mean'] for v in values])
    lo,hi = np.array([v['ci95'] for v in values]).T
    ax.errorbar(x,y,yerr=np.maximum(0,[y-lo,hi-y]),fmt='o',capsize=2,**kwargs)


def plot(endpoint,births,path):
    fig,axes = plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    axis = axes[0,0]
    scale,coord = [],[]
    for group in range(1,6):
        rows = [r for r in endpoint['controls'] if r['group']==group]
        scale.append(max(r['scores'][1]['kl']['mean'] for r in rows if 'matched' in r['control']))
        coord.append(max(r['scores'][1]['kl']['mean'] for r in rows if 'coordinate' in r['control']))
    x = np.arange(1,6)
    axis.bar(x-.18,scale,.36,label='Group scaling',color='#47789D')
    axis.bar(x+.18,coord,.36,label='Internal coordinate',color='#C27256')
    axis.set(yscale='log',xticks=x,xlabel='Learned group',ylabel='Maximum validation KL',title='Equal parameter displacement')
    axis.legend(fontsize=8)

    axis = axes[0,1]
    rho = np.array(endpoint['geometry'][1]['rho'])
    raster = axis.imshow(rho,cmap='RdBu_r',vmin=-1,vmax=1)
    for i in range(5):
        for j in range(5):
            axis.text(j,i,f'{rho[i,j]:.2f}',ha='center',va='center',fontsize=8,
                      color='white' if abs(rho[i,j])>.62 else '#171D28')
    axis.set(xticks=x-1,yticks=x-1,xticklabels=x,yticklabels=x,title='Validation Fisher cosine')
    fig.colorbar(raster,ax=axis,shrink=.8)

    axis = axes[0,2]
    for fraction,color,shift in ((.1,'#47789D',-.16),(1.,'#C27256',.16)):
        rows = [r for r in endpoint['compensations'] if r['removed_fraction']==fraction]
        recovery = [1-r['restored']['kl']['mean']/r['removed']['kl']['mean'] for r in rows]
        axis.bar(x+shift,recovery,.32,color=color,label=f'{fraction:.0%} removal')
    axis.axhline(0,color='gray',lw=.7)
    axis.set(xticks=x,xlabel='Removed group',ylabel='Fraction of KL recovered',title='Compensation by the other four groups')
    axis.legend(fontsize=8)

    axis = axes[1,0]
    pos = np.arange(len(births))
    for i,(key,color) in enumerate(zip(('old','random_1','random_2','random_3'),('#244D78','#A77356','#BA917A','#755F91'))):
        errorbars(axis,pos+(.12*(i-1.5)),[b['real_gain_over'][key] for b in births],label=key,color=color,markersize=4)
    axis.axhline(0,color='gray',lw=.7)
    axis.set(xticks=pos,xticklabels=[b['step'] for b in births],xlabel='Optimizer step',
             ylabel='Real group CE gain over control',title='Birth necessity at equal budget')
    axis.legend(fontsize=7,ncol=2)

    axis = axes[1,1]
    plans = endpoint['regroupings']
    chosen = [plan['groups'][plan['fit_selected_group']-1]['validation']['ce_gain'] for plan in plans]
    errorbars(axis,np.arange(len(plans)),chosen,color='#47789D',markersize=4)
    axis.axhline(0,color='gray',lw=.7)
    axis.set(xticks=np.arange(len(plans)),xticklabels=['Original','Pair 1','Pair 2','Pair 3','Rot 1','Rot 2','Rot 3'],
             ylabel='Validation CE gain',title='Best fitted single group, same span')
    axis.tick_params(axis='x',rotation=24)

    axis = axes[1,2]
    rows = endpoint['interactions']
    strongest = sorted(rows,key=lambda r:abs(r['ce_interaction']['mean']),reverse=True)[:8]
    errorbars(axis,np.arange(len(strongest)),[r['ce_interaction'] for r in strongest],color='#93654F',markersize=4)
    axis.axhline(0,color='gray',lw=.7)
    labels = [f'{r["groups"][0]}{r["alpha"]:+g}/{r["groups"][1]}{r["beta"]:+g}' for r in strongest]
    axis.set(xticks=np.arange(len(labels)),xticklabels=labels,ylabel='Paired CE interaction',title='Largest finite group interactions')
    axis.tick_params(axis='x',rotation=33,labelsize=7)
    fig.suptitle('DTD learned-group geometry audit',fontsize=15)
    fig.savefig(path,dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('endpoint','births','output'):
        p.add_argument(name,type=Path)
    args = p.parse_args()
    endpoint = json.loads(args.endpoint.read_text(encoding='utf-8'))
    births = json.loads(args.births.read_text(encoding='utf-8'))['births']
    plot(endpoint,births,args.output)


if __name__ == '__main__':
    main()
