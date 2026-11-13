"""Read private coordinate maps; export only aggregate enrichment/stability."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import beta


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output
    report=json.loads((out/'report.json').read_text());private=out/'private'
    b=np.load(private/'p0_ellB.npy',mmap_mode='r');q=np.load(private/'p0_ellQ.npy',mmap_mode='r')
    valid=q>1e-20;E=b[valid].astype('float64')/(q[valid].astype('float64')*10/26)
    assert E.min()>=0 and E.max()<2.6001
    threshold=2.6*beta.ppf(.95,5,8)
    enrichment=dict(valid_coordinates=int(valid.sum()),minimum=float(E.min()),maximum=float(E.max()),
        quantiles=dict(zip(['1p','25p','50p','75p','99p'],np.quantile(E,[.01,.25,.5,.75,.99]).tolist())),
        weighted_mean=float(b.sum(dtype=np.float64)/(q.sum(dtype=np.float64)*10/26)),
        random_within_Q_marginal='E /2.6 follows Beta(5,8) for a fixed nonzero Q row and Haar random10 in26',
        marginal_null95=float(threshold),fraction_exceeding_marginal95=float((E>threshold).mean()),
        caveat='Correlated coordinates: no iid KS test or per-coordinate significance claim; matrix-bin null uses1000 shared Haar draws')
    del E,valid
    access=np.load(private/'p0_g.npy',mmap_mode='r');D=len(b);k=int(D*.01)
    top_access=np.argpartition(access,-k)[-k:]
    low_participation={}
    thresholds=np.quantile(b,[.25,.5,.9])
    for fraction,threshold in zip([.25,.5,.9],thresholds):
        chosen=b[top_access]<=threshold
        low_participation[str(fraction)]=dict(leverage_threshold=float(threshold),
            count_in_access_top1percent=int(chosen.sum()),fraction_of_access_top1percent=float(chosen.mean()),
            access_mass_fraction=float(access[top_access[chosen]].sum(dtype=np.float64)/access.sum(dtype=np.float64)))
    other_access=np.load(private/'p1_g.npy',mmap_mode='r')
    other_threshold=float(np.partition(other_access,len(other_access)-k)[len(other_access)-k])
    candidates=top_access[b[top_access]<=thresholds[1]]
    replicated=candidates[other_access[candidates]>=other_threshold]
    ranked=replicated[np.argsort(access[replicated])[-20:][::-1]]
    groups=report['config']['groups'];ends=np.array([g['end'] for g in groups])
    shapes={a['name']:a['shape'] for a in json.loads((out/'assets.json').read_text())['active']}
    examples=[]
    for index in ranked:
        group=groups[int(np.searchsorted(ends,index,side='right'))];shape=shapes[group['name']]
        flat=int(index)-group['start']
        examples.append(dict(global_index=int(index),matrix=group['name'],matrix_index=[flat//shape[1],flat%shape[1]],
            ellB=float(b[index]),g_probe0=float(access[index]),g_probe1=float(other_access[index])))
    localization_examples=dict(rule='ellB below its global median AND g in global top1% on both probes',
        probe0_candidates=int(len(candidates)),replicated_coordinates=int(len(replicated)),examples=examples,
        caveat='Examples show low relative representation and high unit-coordinate access, not causal importance or task conflict')
    repeat={};D=len(b);k=int(D*.01)
    for metric in ['g','rho','a_old','a_expert']:
        x=np.load(private/f'p0_{metric}.npy',mmap_mode='r');y=np.load(private/f'p1_{metric}.npy',mmap_mode='r')
        sx=sy=sxx=syy=sxy=0.
        for start in range(0,D,1048576):
            a=x[start:start+1048576].astype('float64');c=y[start:start+1048576].astype('float64')
            sx+=a.sum();sy+=c.sum();sxx+=a@a;syy+=c@c;sxy+=a@c
        corr=(sxy-sx*sy/D)/np.sqrt((sxx-sx*sx/D)*(syy-sy*sy/D))
        ix=np.argpartition(np.abs(x),-k)[-k:];iy=np.argpartition(np.abs(y),-k)[-k:]
        intersection=len(np.intersect1d(ix,iy,assume_unique=True))
        repeat[metric]=dict(coordinate_Pearson=float(corr),top1percent_Jaccard=intersection/(2*k-intersection),
                            top1percent_intersection_fraction=intersection/k)
    layers=[]
    for probe in report['probes']:
        Gamma=np.array(probe['Gamma']);groups=probe['matrix_scores']
        assignment=np.zeros((12,48))
        for i,g in enumerate(groups):assignment[g['layer'],i]=1
        G=assignment@Gamma@assignment.T
        layers.append(dict(probe=probe['probe'],Gamma=G.tolist(),rho=G.sum(1).tolist(),
            deletion_energy_change=(2*G.sum(1)-np.diag(G)).tolist(),
            caveat='Layer bins are descriptive aggregations, not discovered functional groups'))
    result=dict(coordinate_enrichment=enrichment,low_participation_high_access=low_participation,localization_examples=localization_examples,
        coordinate_repeatability=repeat,layer_aggregates=layers,
        scope_amendment='User clarified on Nov14: B is single-task DTD raw-coordinate control; no actual fusion anchor is required. Original A administrative pending-fusion fields are superseded.',
        paired_probe_ids_disjoint=not bool(set(report['config']['probe_ids'][0])&set(report['config']['probe_ids'][1])))
    (out/'aggregate_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(enrichment=enrichment,repeat=repeat)))


if __name__=='__main__':main()
