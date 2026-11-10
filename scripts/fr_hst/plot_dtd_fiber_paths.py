"""Visualize the cumulative reconstruction and held-out local path measurements."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw(data, output):
    anchors = data['anchors']
    names = ('A', 'AC', 'AR', 'ACR')
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax = axes[0,0]
    models = data['ledger']['models']
    bars = ax.bar(names,[models[name]['ce'] for name in names],color=['#A4ADBA','#466C9C','#73A18A','#384B63'])
    for name,bar in zip(names,bars):
        ax.annotate(f'{models[name]["accuracy"]:.2%}',(bar.get_x()+bar.get_width()/2,bar.get_height()),
                    xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
    ax.margins(y=.14)
    ax.set_title('Endpoint reconstruction')
    ax.set_ylabel('Historical Test CE')
    ax = axes[0,1]
    positions = np.arange(2*len(anchors))
    geometry = [g for a in anchors for g in a['probe_geometry']]
    fits = [max(g['eigenvalues'][0]/g['eigenvalues'][-1],1e-15) for g in geometry]
    ax.semilogy(positions,fits,'o',label='Fit',color='#466C9C')
    ax.semilogy(positions,[g['low_holdout_relative'] for g in geometry],'s',label='Held out',color='#C4644E')
    ax.axhline(.01,color='#757A87',linestyle=':')
    ax.set_ylabel('Low direction energy / maximum eigenvalue')
    ax.set_title('Transfer of the low-spectrum direction')
    ax.legend(fontsize=8)
    trials = [t for a in anchors for t in a['trials']]
    ax = axes[1,0]
    for label,values,color,shift in (
        ('Straight low',[t['path'][-1]['straight_low']['groups'][1]['kl']['mean'] for t in trials],'#466C9C',-.12),
        ('Corrected',[t['path'][-1]['corrected']['groups'][1]['kl']['mean'] for t in trials],'#C4644E',.12),
        ('Straight high',[t['straight'][-1]['high']['groups'][t['heldout_group']]['kl']['mean'] for t in trials],'#9096A4',0.),
    ):
        ax.semilogy(positions+shift,np.maximum(values,1e-14),'o',label=label,color=color,markersize=4)
    ax.set_ylabel('Held-out KL at final path progress')
    ax.set_title('Three-step finite path')
    ax.legend(fontsize=8)
    ax = axes[1,1]
    for i,(key,label,color) in enumerate((('horizontal','Full horizontal','#466C9C'),('covered','Carrier fit','#C4644E'))):
        metrics = [t[key]['groups'][t['heldout_group']]['ce_gain'] for t in trials]
        y = np.array([m['mean'] for m in metrics])
        lo,hi = np.array([m['ci95'] for m in metrics]).T
        ax.errorbar(positions+(.16 if i else -.16),y,yerr=np.maximum(0,[y-lo,hi-y]),fmt='o',
                    markersize=4,linewidth=1,label=label,color=color)
    ax.axhline(0,color='#757A87',linewidth=.8)
    ax.set_ylabel('Held-out CE gain with 95% interval')
    ax.set_title('Task effect of the horizontal approximation')
    ax.legend(fontsize=8)
    for ax in (axes[0,1],axes[1,0],axes[1,1]):
        ax.set_xticks(positions[::2]+.5,[str(a['step']) for a in anchors])
        ax.set_xlabel('Optimizer step, A/B fit per pair')
    fig.suptitle('DTD cumulative carrier contribution and restricted fiber geometry',fontsize=15)
    fig.savefig(output,dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('output',type=Path)
    args = p.parse_args()
    draw(json.loads(args.results.read_text(encoding='utf-8')),args.output)


if __name__ == '__main__':
    main()
