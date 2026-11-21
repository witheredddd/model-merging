#!/usr/bin/env python3
"""DTD/Cars parameter-balance hyperplane pilot.

Two modes:
  1. Physical parameters: aligned flattened .npy arrays of CE gradients and expert deltas.
  2. Legacy 304D: one or more local geometry.npz archives containing h_DTD/h_Cars.

Does NOT assume that coordinate conflict proves harmful model-merging interference.
Outputs diagnostic histograms/JSON/CSV and (optionally) diagnostic labels.
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np


def finite(arr):
    return np.isfinite(arr)


def weight_and_balance(gd, gc, ud=None, uc=None, safe=1e-30):
    """Return opposition mask, balance lambda*, gradient-scale and update-scale weights."""
    opp = (gd * gc < 0) & finite(gd) & finite(gc)
    ab = np.abs(gd) + np.abs(gc)
    lam = np.divide(np.abs(gc), ab, out=np.full_like(ab, np.nan), where=opp & (ab > safe))
    harmonic = np.divide(2 * np.abs(gd) * np.abs(gc), ab,
                         out=np.zeros_like(ab), where=ab > safe)
    if ud is not None and uc is not None:
        # First-order directional loss scale; NOT a proven finite-step damage measure.
        possible = np.maximum(np.abs(ud), np.abs(uc))
        weight = harmonic * possible
        harm_d_from_c = np.maximum(gd * uc, 0.0)
        harm_c_from_d = np.maximum(gc * ud, 0.0)
        cross_harm = harm_d_from_c + harm_c_from_d
    else:
        weight = harmonic
        cross_harm = np.full_like(weight, np.nan)
    return opp, lam, harmonic, weight, cross_harm


def load_npy(path):
    z = np.load(path, mmap_mode='r', allow_pickle=False)
    if z.ndim != 1 or z.dtype.kind not in 'fiu':
        raise ValueError(f'{path} must be a flat numeric .npy, got {z.shape} {z.dtype}')
    return z


def hist_bins(bins):
    return np.linspace(0,1,bins+1,dtype=np.float64)


def analyze_pair(gd, gc, ud=None, uc=None, gd2=None, gc2=None,
                 bins=40, chunk=500_000, seed=42, nulls=0, emit_labels=None, scale_d=1.0, scale_c=1.0):
    n=len(gd)
    for z in [gc, ud, uc, gd2, gc2]:
        if z is not None and len(z)!=n:raise ValueError('Different coordinate lengths / flatten orders')
    edges=hist_bins(bins)
    names=['count','weight','cross_harm','heldout_count','heldout_weight','null_count','null_weight']
    hist={k:np.zeros(bins,dtype=np.float64) for k in names}
    null_count=np.zeros((nulls,bins),dtype=np.float64)
    null_weight=np.zeros((nulls,bins),dtype=np.float64)
    totals=dict(n_coordinates=n,scale_d=scale_d,scale_c=scale_c,opposite=0,stable_opposite=0,lambda_stable=0,
                active_gradient_weight=0.,active_update_weight=0.,first_order_cross_harm=0.,
                first_order_cross_harm_conflict_coords=0.,
                heldout_lambda_abs_change_weighted_numer=0.,
                heldout_lambda_abs_change_weighted_denom=0.,
                edge_weight=0.,mid_weight=0.)
    labels=np.lib.format.open_memmap(emit_labels,mode='w+',dtype=np.int8,shape=(n,)) if emit_labels else None
    rng=np.random.default_rng(seed)
    for lo in range(0,n,chunk):
        hi=min(n,lo+chunk)
        d=np.asarray(gd[lo:hi],dtype=np.float64)*scale_d
        c=np.asarray(gc[lo:hi],dtype=np.float64)*scale_c
        u=np.asarray(ud[lo:hi],dtype=np.float64) if ud is not None else None
        v=np.asarray(uc[lo:hi],dtype=np.float64) if uc is not None else None
        opp,lam,imp,w,cross=weight_and_balance(d,c,u,v)
        imp=np.nan_to_num(imp,nan=0.,posinf=0.,neginf=0.)
        w=np.nan_to_num(w,nan=0.,posinf=0.,neginf=0.)
        totals['opposite']+=int(opp.sum())
        totals['active_gradient_weight']+=float(imp[opp].sum())
        totals['active_update_weight']+=float(w[opp].sum())
        totals['edge_weight']+=float(w[opp & ((lam<.1)|(lam>.9))].sum())
        totals['mid_weight']+=float(w[opp & ((lam>=.4)&(lam<=.6))].sum())
        if u is not None:
            totals['first_order_cross_harm']+=float(np.nansum(cross))
            totals['first_order_cross_harm_conflict_coords']+=float(np.nansum(cross[opp]))
        hist['count']+=np.histogram(lam[opp],bins=edges)[0]
        hist['weight']+=np.histogram(lam[opp],bins=edges,weights=w[opp])[0]
        if u is not None:hist['cross_harm']+=np.histogram(lam[opp],bins=edges,weights=cross[opp])[0]
        if gd2 is not None:
            d2=np.asarray(gd2[lo:hi],dtype=np.float64)*scale_d
            c2=np.asarray(gc2[lo:hi],dtype=np.float64)*scale_c
            opp2,lam2,_,w2,_=weight_and_balance(d2,c2,u,v)
            stable=opp & opp2
            good=stable & (np.abs(lam2-lam)<=0.10)
            totals['stable_opposite']+=int(stable.sum())
            totals['lambda_stable']+=int(good.sum())
            totals['heldout_lambda_abs_change_weighted_numer']+=float(np.nansum(w[stable]*np.abs(lam[stable]-lam2[stable])))
            totals['heldout_lambda_abs_change_weighted_denom']+=float(np.nansum(w[stable]))
            hist['heldout_count']+=np.histogram(lam[good],bins=edges)[0]
            hist['heldout_weight']+=np.histogram(lam[good],bins=edges,weights=w[good])[0]
        if labels is not None:
            # labels 0=unselected/weak/no-opposition, 1=left, 2=middle, 3=right.
            # Never claim these are discovered functional clusters.
            lab=np.zeros(hi-lo,dtype=np.int8)
            if gd2 is not None:
                eligible=good
            else:
                eligible=opp
            lab[eligible & (lam<1/3)]=1
            lab[eligible & (lam>=1/3)&(lam<=2/3)]=2
            lab[eligible & (lam>2/3)]=3
            labels[lo:hi]=lab
        for j in range(nulls):
            # Destructive pairing shuffle WITHIN THIS CHUNK, not across the dataset.
            # Real tensor-aligned null requires chunk boundaries to match matrix bounds.
            perm=rng.permutation(hi-lo)
            _,ln,_,wn,_=weight_and_balance(d,c[perm],u,v[perm] if v is not None else None)
            hit=np.isfinite(ln)
            null_count[j]+=np.histogram(ln[hit],bins=edges)[0]
            null_weight[j]+=np.histogram(ln[hit],bins=edges,weights=wn[hit])[0]
    if labels is not None:labels.flush()
    if nulls:
        hist['null_count']=null_count.mean(0)
        hist['null_weight']=null_weight.mean(0)
    else:
        hist.pop('null_count');hist.pop('null_weight')
    den=max(totals['active_update_weight'],1e-30)
    totals['edge_weight_fraction']=totals['edge_weight']/den
    totals['mid_weight_fraction']=totals['mid_weight']/den
    totals['opposition_fraction']=totals['opposite']/max(n,1)
    totals['stable_opposition_fraction']=totals['stable_opposite']/max(totals['opposite'],1) if gd2 is not None else None
    totals['weighted_lambda_MAE_conditional_stability']=(totals['heldout_lambda_abs_change_weighted_numer']/max(totals['heldout_lambda_abs_change_weighted_denom'],1e-30)) if gd2 is not None else None
    if nulls:
        hf=hist['null_weight']/max(hist['null_weight'].sum(),1e-30)
        real=hist['weight']/max(hist['weight'].sum(),1e-30)
        totals['null_weighted_edge_fraction']=float(hf[:max(1,bins//10)].sum()+hf[-max(1,bins//10):].sum())
        totals['observed_null_weighted_hist_L1']=float(np.abs(real-hf).sum())
    return edges,hist,totals


def save_results(out,edges,hist,totals,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps({'title':title,**totals},indent=2,ensure_ascii=False,allow_nan=False))
    with (out/'histogram.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['lambda_left','lambda_right']+list(hist.keys()))
        for i in range(len(edges)-1):
            writer.writerow([float(edges[i]),float(edges[i+1])]+[float(x[i]) for x in hist.values()])
    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True)
    x=(edges[:-1]+edges[1:])/2
    barw=edges[1]-edges[0]
    axs[0].bar(x,hist['count'],width=barw,alpha=.68,label='Opposite CE gradients')
    if 'null_count' in hist:axs[0].plot(x,hist['null_count'],label='Shuffled null (within chunk)')
    axs[0].set_ylabel('Coordinates');axs[0].legend()
    ww=hist['weight']/max(1e-30,hist['weight'].sum())
    axs[1].bar(x,ww,width=barw,alpha=.68,label='Weighted by gradient tension and expert step')
    if 'null_weight' in hist:
        ww0=hist['null_weight']/max(1e-30,hist['null_weight'].sum())
        axs[1].plot(x,ww0,label='Shuffled null (within chunk)')
    axs[1].set_ylabel('Normalized weight');axs[1].set_xlabel('DTD preference at coordinate cancellation, lambda*')
    axs[1].legend();fig.suptitle(title)
    fig.tight_layout();fig.savefig(out/'balance_histogram.png',dpi=170);plt.close(fig)


def archived(paths,out,bins,seed,nulls,scale_d=1.0,scale_c=1.0):
    data=[]
    for p in paths:
        with np.load(p,allow_pickle=False) as f:
            data.append((np.asarray(f['h_DTD'],dtype=np.float64),np.asarray(f['h_Cars'],dtype=np.float64)))
    if any(len(x[0]) !=len(data[0][0]) for x in data): raise ValueError('Different archived coordinate dictionaries')
    for a,(d,c) in enumerate(data):
        d2,c2=data[(a+1)%len(data)] if len(data)>1 else (None,None)
        e,h,s=analyze_pair(d,c,gd2=d2,gc2=c2,bins=bins,seed=seed,nulls=nulls,scale_d=scale_d,scale_c=scale_c)
        save_results(out/f'anchor_{a}',e,h,s,f'Archived 304D control coordinates, anchor {a}')
    (out/'SCOPE.md').write_text('ARCHIVED CONTROL COORDINATES ONLY: h_DTD/h_Cars are derivatives with respect to the 304D expert-recombination dictionary, NOT the original 85M physical parameters. Cross-anchor consistency is not independent-probe reproducibility. All gradients and metrics are local; no empirical merge improvement has been established.\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--geometry',nargs='+',type=Path,help='304D local geometry.npz files (archive mode)')
    mode.add_argument('--grad-d',type=Path,help='flat .npy gradient of DTD CE at COMMON anchor')
    p.add_argument('--grad-c',type=Path)
    p.add_argument('--grad-d-heldout',type=Path)
    p.add_argument('--grad-c-heldout',type=Path)
    p.add_argument('--delta-d',type=Path,help='expert delta DTD, same flattened physical weights')
    p.add_argument('--delta-c',type=Path,help='expert delta Cars, same flattened physical weights')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bins',type=int,default=40)
    p.add_argument('--chunk',type=int,default=500_000)
    p.add_argument('--nulls',type=int,default=5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--scale-d',type=float,default=1.0,help='DTD CE scaling; positive, used to test loss-scale sensitivity')
    p.add_argument('--scale-c',type=float,default=1.0,help='Cars CE scaling; positive, used to test loss-scale sensitivity')
    p.add_argument('--emit-labels',action='store_true')
    a=p.parse_args()
    if a.bins<4 or a.chunk<2 or a.nulls<0 or a.scale_d<=0 or a.scale_c<=0:raise ValueError('Invalid bins/chunk/nulls/scales')
    if a.geometry is not None:
        archived(a.geometry,a.output,a.bins,a.seed,a.nulls,a.scale_d,a.scale_c)
    else:
        if a.grad_c is None: p.error('--grad-c required with --grad-d')
        if bool(a.grad_d_heldout)!=bool(a.grad_c_heldout):p.error('heldout gradients require both tasks')
        if bool(a.delta_d)!=bool(a.delta_c):p.error('expert deltas require both tasks')
        gD=load_npy(a.grad_d);gC=load_npy(a.grad_c)
        uD=load_npy(a.delta_d) if a.delta_d else None
        uC=load_npy(a.delta_c) if a.delta_c else None
        gD2=load_npy(a.grad_d_heldout) if a.grad_d_heldout else None
        gC2=load_npy(a.grad_c_heldout) if a.grad_c_heldout else None
        label_path=a.output/'diagnostic_labels.npy' if a.emit_labels else None
        a.output.mkdir(parents=True,exist_ok=True)
        e,h,s=analyze_pair(gD,gC,uD,uC,gD2,gC2,a.bins,a.chunk,a.seed,a.nulls,label_path,a.scale_d,a.scale_c)
        save_results(a.output,e,h,s,'Physical CLIP parameters: DTD/Cars Pareto-balance diagnostic')
        (a.output/'SCOPE.md').write_text('Physical-parameter pilot: gradients at the same shared model anchor, with identical flatten ordering; train/heldout probes must be disjoint. Lambda* is a local coordinate-wise cancellation ratio, NOT a proof of semantic clusters or harmful finite-step interference. Null permutation is within chunks; use chunk boundaries aligned with physical tensors for stronger matched controls. Only an independent real-model merge evaluation can establish a benefit.\n')
    print('Saved diagnostics to:',a.output.resolve())

if __name__=='__main__':main()
