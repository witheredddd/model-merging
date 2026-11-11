from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=(Path(__file__).resolve().parent.parent if Path(__file__).parent.name=='medium_launch' else
      Path(__file__).parent/'fisher-rao-internal-groups-git/scripts/fisher_rao_online_v5/evidence/20261112')
fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
for column,task in enumerate(('DTD','SUN397')):
    report=json.loads((root/f'medium_{task}'/'report.json').read_text())
    rows=report['selection_reports'];cost=[r['exact_online_image_cost']/1000 for r in rows]
    ce=[r['instant_validation']['CE'] for r in rows]
    acc=[100*r['instant_validation']['accuracy'] for r in rows]
    axes[0,column].plot(cost,ce,color='#236A9B',lw=1.8,label='Committed model at cost grid')
    axes[1,column].plot(cost,acc,color='#236A9B',lw=1.8)
    final_cost=report['actual_online_images']/1000
    axes[0,column].scatter([final_cost],[report['final_validation']['CE']],color='#CF6636',s=45,zorder=5,label='Final model')
    axes[1,column].scatter([final_cost],[100*report['final_validation']['accuracy']],color='#CF6636',s=45,zorder=5)
    for row in range(2):
        ax=axes[row,column];ax.set_xlabel('Charged online images (thousands)');ax.grid(alpha=.2)
    axes[0,column].set_title(task+' | 128 main attempts, seed 11')
    axes[0,column].set_ylabel('Calibration CE (n=128)')
    axes[1,column].set_ylabel('Calibration accuracy (%)')
    axes[0,column].legend(fontsize=8)
    axes[1,column].text(.03,.96,f"Final independent dev accuracy: {100*report['independent_final']['accuracy']:.2f}% (n=512)",
        transform=axes[1,column].transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
fig.suptitle('Fisher-Rao v5 bounded medium runs | full recovery at main attempt 8',fontsize=13)
fig.savefig(root/'medium_launch'/'cost_curves.png',dpi=180)
print(root/'medium_launch'/'cost_curves.png')
