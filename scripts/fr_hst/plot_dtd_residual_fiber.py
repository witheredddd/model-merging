"""Plot the DTD residual diagnosis and keep sample counts separate from Fisher energy."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASSES = ('V', 'H1', 'H2', 'H3', 'undetermined')
COLORS = ('#6EADC0', '#499372', '#6556A8', '#C6674C', '#B5BDCA')
STAGES = ((37,80), (81,120), (121,160), (161,216))


def draw(data, output):
    rows = data['windows']
    steps = [r['step'] for r in rows]
    births = [r['step'] for r in rows if r['after_birth'] is not None]
    fig = plt.figure(figsize=(11,11), constrained_layout=True)
    grid = fig.add_gridspec(4,2,height_ratios=(1,1,1,1.1))
    absolute, novelty, gain = [fig.add_subplot(grid[i,:]) for i in range(3)]
    for axis in (absolute,novelty,gain):
        axis.set_xlim(36,218)
        for step in births:
            axis.axvline(step,color='#7c8493',linestyle=':',linewidth=.8,zorder=0)
        axis.set_xlabel('Optimizer step')
    for key,label in [('proposal_fisher','Candidate'), ('structured_fisher','Structured'),
                      ('residual_fisher','Residual'), ('random_fisher','Norm-matched random')]:
        y = [np.mean([g[key] for g in r['geometry']]) for r in rows]
        absolute.semilogy(steps,y,'.-',label=label,markersize=3)
    absolute.set_ylabel('Mean Fisher energy')
    absolute.legend(ncol=4,loc='upper right',fontsize=8)
    for i, name in enumerate(('Fit B → measure A','Fit A → measure B')):
        novelty.plot(steps,[r['projection']['cross_novelty'][i] for r in rows],'.-',label=name,markersize=4)
    for r in rows:
        if r['after_birth']:
            before = np.mean(r['projection']['cross_novelty'])
            after = np.mean(r['after_birth']['all_components']['cross_novelty'])
            novelty.annotate('',xy=(r['step'],after),xytext=(r['step'],before),
                             arrowprops=dict(arrowstyle='->',color='#2f2d3e',lw=1.5))
    novelty.axhspan(.35,.65,color='#E9EAF0',alpha=.5)
    novelty.set_ylabel('Cross-fit novel fraction')
    novelty.legend(fontsize=8)
    for label,stats,color in [('Whole residual',[r['ce_doses'][-1]['pooled'] for r in rows],'#265C9F'),
                              ('Beyond covered',[r['ce_novel_gain']['pooled'] for r in rows],'#86549A')]:
        y = [s['mean'] for s in stats]
        lo,hi = np.array([s['ci95'] for s in stats]).T
        gain.plot(steps,y,'.-',color=color,label=label,markersize=3)
        gain.fill_between(steps,lo,hi,color=color,alpha=.12,linewidth=0)
    gain.axhline(0,color='gray',linewidth=.8)
    gain.set_ylabel('Paired CE improvement')
    gain.legend(fontsize=8)
    for j,(axis_title,weight) in enumerate((('Share of sampled windows',lambda r: 1.),
                                          ('Share of residual Fisher energy',lambda r: np.mean([g['residual_fisher'] for g in r['geometry']])))):
        axis = fig.add_subplot(grid[3,j])
        previous = np.zeros(len(STAGES))
        buckets = [[r for r in rows if start <= r['step'] <= end] for start,end in STAGES]
        totals = [max(sum(weight(r) for r in bucket),1e-30) for bucket in buckets]
        for name,color in zip(CLASSES,COLORS):
            share = np.array([sum(weight(r) for r in bucket if r['residual_class']==name)/total
                              for bucket,total in zip(buckets,totals)])
            axis.bar(range(len(STAGES)),share,bottom=previous,color=color,label=name)
            previous += share
        axis.set_ylim(0,1.03)
        axis.set_xticks(range(len(STAGES)),[f'{a}–{b}' for a,b in STAGES])
        axis.set_title(axis_title,fontsize=10)
        axis.set_xlabel('Optimizer steps')
        if j==0:
            axis.legend(ncol=3,fontsize=7,loc='lower left')
    fig.suptitle('DTD residual response around the live training trajectory',fontsize=15,fontweight='bold')
    fig.savefig(output,dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    draw(json.loads(args.results.read_text(encoding='utf-8')),args.output)


if __name__ == '__main__':
    main()
