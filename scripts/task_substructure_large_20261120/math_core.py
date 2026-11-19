"""Small numerical primitives, independent of the model and CUDA."""
import numpy as np

def class_weights(classes,kind):
    x=2*np.pi*np.arange(classes)/classes
    if kind=='mean':return np.ones(classes)
    trig,freq=kind.split('_');return 1+.75*(np.sin(int(freq)*x) if trig=='sin' else np.cos(int(freq)*x))

def gram_modes(gram):
    values,V=np.linalg.eigh((gram+gram.T)/2);order=np.argsort(values)[::-1]
    values=values[order];V=V[:,order]
    cutoff=np.finfo(float).eps*max(gram.shape)*max(float(values[0]),np.finfo(float).tiny)
    keep=values>cutoff
    return values,V[:,keep]/np.sqrt(values[keep]),dict(machine_cutoff=cutoff,resolved_rank=int(keep.sum()),
        proxy_tail_losses=[.5*float(np.maximum(values[r:],0).sum()) for r in range(len(values)+1)])

def metric_stats(logits,labels,weights,baseline=None):
    logits=np.asarray(logits,dtype=float);labels=np.asarray(labels,dtype=int)
    z=logits-logits.max(1,keepdims=True);logp=z-np.log(np.exp(z).sum(1,keepdims=True));prob=np.exp(logp)
    n=len(labels);ce=-logp[np.arange(n),labels];w=np.asarray(weights)[labels]
    correct=logits.argmax(1)==labels;other=logits.copy();other[np.arange(n),labels]=-np.inf
    margin=logits[np.arange(n),labels]-other.max(1)
    result=dict(n=n,CE=float(ce.mean()),weighted_CE=float(w@ce/w.sum()),Top1=float(correct.mean()),
        correct=int(correct.sum()),margin_mean=float(margin.mean()),margin_quantiles=np.quantile(margin,[0,.1,.5,.9,1]).tolist(),
        per_class={str(c):dict(n=int((labels==c).sum()),CE=float(ce[labels==c].mean()),Top1=float(correct[labels==c].mean())) for c in np.unique(labels)})
    if baseline is not None:
        base=metric_stats(baseline,labels,weights);a=baseline-baseline.max(1,keepdims=True);p0=np.exp(a)/np.exp(a).sum(1,keepdims=True)
        ce0=-np.log(np.maximum(p0[np.arange(n),labels],np.finfo(float).tiny));diff=ce-ce0
        weighted_diff=n*w*diff/w.sum();se=float(weighted_diff.std(ddof=1)/np.sqrt(n)) if n>1 else 0.
        result.update(delta_CE=result['CE']-base['CE'],delta_weighted_CE=result['weighted_CE']-base['weighted_CE'],
            delta_Top1=result['Top1']-base['Top1'],paired_weighted_CE_SE=se,
            paired_weighted_CE_approx95=[float(weighted_diff.mean()-1.96*se),float(weighted_diff.mean()+1.96*se)],
            prediction_flip_rate=float((logits.argmax(1)!=baseline.argmax(1)).mean()),
            probability_L2=float(np.linalg.norm(prob-p0)/np.sqrt(n)),
            sqrt_probability_chord=float(np.linalg.norm(2*(np.sqrt(prob)-np.sqrt(p0)))/np.sqrt(n)))
    return result

def dense_restricted(H,g,P):
    """Reference only: exact quadratic optimum on the image of a projector."""
    eig,U=np.linalg.eigh(P);Q=U[:,eig>.5]
    v=-Q@np.linalg.solve(Q.T@H@Q,Q.T@g)
    return v,float(g@v+.5*v@H@v)
