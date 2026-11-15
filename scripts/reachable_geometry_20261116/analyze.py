"""Summaries and functional omissions; does not fit or choose on p4/p5."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,argparse
from pathlib import Path
import numpy as np
from run import root,write,OLD,spectrum

def main(out):
    rows=json.loads((out/'programs.json').read_text());gaps=json.loads((out/'capacity_gaps.json').read_text())
    spectra=json.loads((out/'reachable_spectra.json').read_text());banks=json.loads((out/'optimal_motion_spectra.json').read_text())
    omissions=[];summary=[];normalized_banks=[]
    for k in range(3):
        with np.load(OLD/f'private/anchor{k}/geometry.npz') as f:KD=f['K_DTD'];KC=f['K_Cars']
        LD=root(KD);LC=root(KC);L=np.r_[LD,LC]
        full=[r for r in rows if r['anchor']==k and r['space']=='full304' and r['solver']['passed']]
        B=np.stack([np.array(r['v'])/r['radius'] for r in full],1)
        for name,A in [('parameter',np.eye(304)),('DTD_response',LD),('Cars_response',LC),('joint_response',L)]:
            Y=A@B;Y/=np.maximum(np.linalg.norm(Y,axis=0),1e-30)
            normalized_banks.append(dict(anchor=k,metric=name,spectrum=spectrum(Y),
              scope='Each response column normalized to unit norm; sampled single-objective directions only, not intrinsic controllability rank'))
        selected=json.loads((OLD/f'anchor{k}_selected_pair_lock.json').read_text())['selected']
        planes={}
        for j,r in enumerate(selected):
            if r['kind']=='conditional_pair':
                W=np.zeros((304,2));W[r['ids']]=np.array(r['W']);planes[f'pair_{j}']=W
        for r in rows:
            if r['anchor']!=k or r['space']!='full304' or r['factor']!=1.:continue
            ref=next(x for x in rows if all(x[key]==r[key] for key in ['anchor','tau','factor','problem']) and x['space']=='best_cached_conditional_plane')
            W=planes[ref['selected_plane']];v=np.array(r['v'])
            item={key:r[key] for key in ['anchor','tau','problem']};item['conditional_plane']=ref['selected_plane']
            for name,A in [('parameter',np.eye(304)),('Cars_response',LC),('joint_response',L)]:
                y=A@v;AW=A@W;res=y-AW@np.linalg.lstsq(AW,y,rcond=None)[0]
                item[name+'_fraction_outside_plane']=float(np.linalg.norm(res)/max(np.linalg.norm(y),1e-30))
            omissions.append(item)
        sig=np.sqrt(np.maximum(np.linalg.eigvalsh(KD),0))
        summary.append(dict(anchor=k,min_DTD_singular_value=float(sig.min()),max_DTD_singular_value=float(sig.max()),
           numerical_full_column_rank=bool(sig.min()>sig.max()*304*np.finfo(float).eps),
           exact_fiber_in_E='No nonzero kernel at observed numerical precision; exact epsilon=0 admits only zero if full rank holds',
           norm_bound_at_epsilon='||v|| <= min(R,epsilon/sigma_min(J_D)); no nonzero exact fiber is inferred'))
    write(out/'functional_omissions.json',omissions);write(out/'domain_rank_audit.json',summary)
    write(out/'direction_normalized_motion_spectra.json',normalized_banks)
    table=[r for r in gaps if r['factor']==1. and r['problem']=='GN_Cars_with_DTD_CE_protection' and r['tau'] in [.01,.1,.3,1.]]
    write(out/'summary.json',dict(primary_budget_capacity_fractions=table,rank_audit=summary,
      finite_validation='Existing full/global/conditional controls only; see existing_finite_validation.json; no new GPU models or untouched probes',
      important_limits=['The conditional comparator is the p0-selected cached plane family, not all possible 2D controls.',
      'One objective optimum bank can look low rank while the full reachable response ellipsoid is high rank.',
      'A different optimal 1D direction can solve each scalar case; that is not a single shared learned control space.',
      'Reported capacities and spectra are local, conditional on E and the chosen discovery Jacobians.',
      'Inner-ellipsoid spectra characterize a bounded approximation, not the exact spectrum of the nonellipsoidal intersection.']))
    lines=['# Functional reachability: first structural experiment','',
      'Completed using the existing three anchored 304D expert-recombination caches. No expert retraining, Jacobian recomputation, adaptive support growth, dynamic updates or EC. All follow-up analyses are exploratory; p4/p5 were already consumed. p6 and official Test remain unopened.','',
      '## What is optimized','',
      'The norm-only reachable set is {J_C v: ||J_D v|| <= epsilon, ||v|| <= R}. We solve its exact CE-linear support direction, and separately the predictive Fisher/Gauss-Newton CE objective. A third variant retains the prior h_D^T v <= 0 constraint for direct comparison with the original experiment. GN is not the exact CE Hessian. Solver residuals and failed cases are retained in programs.json.','',
      'We compare full304, global expert span G, and the union of the eight previously p0-selected conditional planes at each anchor. The latter is explicitly a cached conditional family, not an upper bound on every possible 2D space. New directions are fitted only with p0 geometry; existing finite evaluation is reused, not relabeled as fresh validation.','',
      '## Matched primary-radius capacity','',
      'Entries are fractions of the full304 optimal predicted GN gain, with the same radius, DTD-response budget and DTD CE slope constraint. These are not actual accuracy gains.','',
      '|Anchor|tau|Global2D / full304|Cached conditional / full304|','|---|---:|---:|---:|']
    for k in range(3):
        for tau in [.01,.1,.3,1.]:
            vals={r['reference']:r['gain_fraction'] for r in table if r['anchor']==k and r['tau']==tau}
            lines.append(f"|{k}|{tau}|{vals.get('global2D',float('nan')):.4f}|{vals.get('best_cached_conditional_plane',float('nan')):.4f}|")
    lines+=['','## Functional capacity and sampled-motion dimension','',
      'M=I/R^2+K_D/epsilon^2 defines an inner feasible ellipsoid. Every point in the exact two-budget intersection belongs to the ellipsoid with squared radius 2. Consequently, the largest achievable Cars squared response lies between sigma_1(J_C M^(-1/2))^2 and twice that value. All 304 singular values and cumulative energy are recorded without Fisher trace truncation. These spectra describe bounded local controllability, not an exact ellipsoidal representation of the reachable set.','',
      'optimal_motion_spectra.json separately records parameter/DTD/Cars/joint spectra of sampled full-space optimal updates. This is a one-Cars-objective optimal-motion family across budgets, not the dimension of the whole reachable set, a nonlinear manifold, or a Pareto front. functional_omissions.json measures how much full-space optimal response lies outside the best cached conditional plane; projection itself need not preserve budgets.','',
      'The restricted DTD Jacobian has positive minimum singular value at all three anchors. Under this observed full-rank model, exact zero source response permits only zero motion. Near-preservation capacities do not establish exact fiber directions.','',
      '## Evidence and artifacts','',
      'experiment_lock.json fixes the problem definitions and grid. capacity_gaps.json, reachable_spectra.json, optimal_motion_spectra.json and functional_omissions.json contain the complete analysis. existing_finite_validation.json preserves actual CE, accuracy and KL for the previously evaluated GN+DTD-CE controls, including failed protection on held probes; it does not validate every newly solved norm-only direction. complete.json records solver audit counts. No automatic dimension selection or new mechanism was started.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(14,10))
    colors={'full304':'black','global2D':'tab:blue','best_cached_conditional_plane':'tab:orange'}
    for k in range(3):
        for j,problem in enumerate(['linear_Cars_support_norm_only','GN_Cars_with_DTD_CE_protection']):
            for name,color in colors.items():
                rr=sorted([r for r in rows if r['anchor']==k and r['factor']==1 and r['problem']==problem and r['space']==name],key=lambda r:r['tau'])
                axes[k,j].plot([r['tau'] for r in rr],[r['linear_Cars_gain' if j==0 else 'GN_Cars_gain'] for r in rr],'.-',color=color,label=name)
            axes[k,j].set(xscale='log',yscale='log',xlabel='DTD budget fraction tau',ylabel='Local CE gain',title=f'Anchor {k}: '+('linear norm-only' if j==0 else 'GN + DTD CE protection'))
            axes[k,j].legend(fontsize=7)
        for tau in [.001,.01,.1,1.,3.]:
            r=next(r for r in spectra if r['anchor']==k and r['tau']==tau)
            axes[k,2].plot(np.arange(1,305),r['cumulative_energy'],label=f'tau={tau:g}')
        axes[k,2].set(xlabel='Retained axes (all 304 available)',ylabel='Inner-ellipsoid response energy',title=f'Anchor {k}: controllability spectrum')
        axes[k,2].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'reachability.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
