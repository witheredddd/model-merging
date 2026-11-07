"""Train-only discovery, independent gate/audit, bootstrap and rotation null."""
import argparse,json
from pathlib import Path
import numpy as np
from functional_commutant_core import recursive_blocks,random_rotation_null,offblock_ratio,commutant_spectrum

def kernel(Q,sizes):
    pos=0;S=np.zeros_like(Q)
    for s in sizes:
        P=Q[:,pos:pos+s]@Q[:,pos:pos+s].T;S+=P**2;pos+=s
    return S

def analyze(path,trials=20,bootstraps=10):
    with np.load(path) as d:
        tr=d['train'];ho=d['holdout'];audit=d['audit'];h=d['h'];samples=d['train_samples'] if 'train_samples' in d else None
    result=dict(reference=Path(path).stem,rank=len(tr[0]),tasks=8,sample_count_per_split=8,
        sampling_warning='8 images per task per split: exploratory estimates, not a population-level guarantee')
    try:
        z=recursive_blocks(tr,ho);ev,_=commutant_spectrum(tr);null=random_rotation_null(tr,ho,n=trials)
        ae=offblock_ratio(audit,z['Q'],z['sizes']);S=kernel(z['Q'],z['sizes']);bootstrap=[]
        rng=np.random.default_rng(4729)
        if samples is not None and len(z['sizes'])>1:
            for b in range(bootstraps):
                bt=np.array([s[rng.integers(0,len(s),len(s))].mean(0) for s in samples])
                fitted=recursive_blocks(bt,ho)
                distance=float(np.linalg.norm(kernel(fitted['Q'],fitted['sizes'])-S)/max(np.linalg.norm(S),1e-12))
                bootstrap.append(dict(sizes=fitted['sizes'],relative_projector_kernel_distance=distance))
        stability=sum(b['relative_projector_kernel_distance']<.25 for b in bootstrap)/len(bootstrap) if bootstrap else 0.
        p=(1+sum(n['count']>=len(z['sizes']) for n in null))/(1+len(null))
        valid=len(z['sizes'])>1 and ae<=.08 and stability>=.8 and p<=.05
        blocks=[];pos=0
        for size in z['sizes']:
            q=z['Q'][:,pos:pos+size];pos+=size
            blocks.append(dict(dimension=size,lambda_max=[float(np.linalg.eigvalsh(q.T@g@q)[-1]) for g in tr],
                signed_CE_responses=(h@q).tolist(),CE_response_gram=(h@q@(h@q).T).tolist()))
        result.update(structure_validated=valid,conclusion='stable approximate common blocks detected' if valid else 'no stable common block detected',
            proposed_sizes=z['sizes'],train_error=z['train_error'],heldout_error=z['heldout_error'],third_audit_error=ae,
            commutant_eigenvalues=ev.tolist(),steps=z['steps'],rotation_null=null,rotation_null_p=p,bootstrap=bootstrap,
            bootstrap_stability_fraction=stability,blocks=blocks)
        Q=z['Q'] if valid else np.eye(len(tr[0]));sizes=z['sizes'] if valid else [len(tr[0])]
    except ValueError as ex:
        result.update(structure_validated=False,conclusion='no stable common block detected',reason=str(ex))
        Q=np.eye(len(tr[0]));sizes=[len(tr[0])]
    target=Path(path).with_suffix('.analysis.json');target.write_text(json.dumps(result,indent=2))
    np.savez(Path(path).with_suffix('.blocks.npz'),Q=Q,sizes=np.array(sizes))
    return result,Q,sizes

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('path');a=ap.parse_args();r,_,_=analyze(a.path);print(json.dumps(r,indent=2))
