"""Paired finite-task substitution gaps, separate from quadratic proxy gaps."""
import json
from pathlib import Path
import numpy as np

def weighted_paired_gap(restricted,full,labels,demand):
    labels=np.asarray(labels);classes=full.shape[1]
    def ce(z):
        z=z-z.max(1,keepdims=True)
        return np.log(np.exp(z).sum(1))-z[np.arange(len(labels)),labels]
    x=2*np.pi*np.arange(classes)/classes
    if demand=='mean':weights=np.ones(classes)
    else:
        trig,freq=demand.split('_');weights=1+.75*(np.sin(int(freq)*x) if trig=='sin' else np.cos(int(freq)*x))
    w=weights[labels];diff=ce(restricted)-ce(full);normalized=len(labels)*w*diff/w.sum()
    gap=float(normalized.mean());se=float(normalized.std(ddof=1)/np.sqrt(len(labels))) if len(labels)>1 else None
    return dict(weighted_CE_gap=gap,paired_approx_SE=se,paired_approx95_lower=gap-1.96*se if se is not None else None,
        paired_approx95_upper=gap+1.96*se if se is not None else None,
        Top1_gap=float((restricted.argmax(1)==labels).mean()-(full.argmax(1)==labels).mean()),
        classes_observed=len(np.unique(labels)),classes_total=classes,missing_classes=classes-len(np.unique(labels)),
        positive_gap_means_restricted_task_loss=True,not_global_nonlinear_optimum_gap=True)

def finite_gaps(root,records):
    root=Path(root);by_case={r['case']:r for r in records};rows=[]
    for r in records:
        if r['family']!='fisher_common_objective' or r['method']=='full':continue
        tail=f'_{r["method"]}_native_r{r["radius_index"]}'
        if not r['case'].endswith(tail):continue
        name=r['case'][:-len(tail)]+f'_full_native_r{r["radius_index"]}'
        if name not in by_case:continue
        full=by_case[name]
        with np.load(root/r['arrays']) as restricted_data,np.load(root/full['arrays']) as full_data:
            shared=np.load(root/full['shared_baseline_file']) if 'shared_baseline_file' in full else full_data
            try:
                for task in ('DTD','Cars'):
                    for probe in (0,2,3):
                        prefix=f'{task}_p{probe}';labels=shared[prefix+'_labels']
                        gap=weighted_paired_gap(restricted_data[prefix+'_logits'],full_data[prefix+'_logits'],labels,r['demand'] if task=='DTD' else 'mean')
                        proxy=r['physical_proxy_optimum']-full['physical_proxy_optimum']
                        rows.append(dict(case=r['case'],anchor=r['anchor'],demand=r['demand'],method=r['method'],alpha=r['alpha'],
                            radius_index=r['radius_index'],task_probe=prefix,restricted_parameter_norm=r['parameter_norm'],full_parameter_norm=full['parameter_norm'],
                            proxy_task='DTD',physical_proxy_deletion=proxy,
                            physical_proxy_deletion_lower=proxy-r['physical_proxy_gap_upper'],
                            physical_proxy_deletion_upper=proxy+full['physical_proxy_gap_upper'],
                            physical_lambda=r['physical_lambda'],physical_beta=r['physical_beta'],**gap))
            finally:
                if shared is not full_data:shared.close()
    return rows
