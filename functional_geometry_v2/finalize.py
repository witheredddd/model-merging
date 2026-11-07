"""Final common-reference geometry, coefficient EC pilot and locked test audit."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from engine import Engine,OUT,ROOT,TASKS,write,save
from train import heartbeat,record
from analyze import analyze

def finish(args):
    E=Engine(micro=args.micro);E.basis_load()
    ck=torch.load(OUT/'selected_experts.pt',map_location='cpu',weights_only=False);zs=ck['z'].cuda();zero=torch.zeros(E.reserve,device='cuda')
    mid=torch.load(OUT/'common_intermediate_reference.pt',map_location='cuda',weights_only=True) if (OUT/'common_intermediate_reference.pt').exists() else zero.clone()
    refs={'final_pool_theta0':zero,'common_intermediate':mid,'baseline_mean_merge':zs.mean(0)};analyses={};partitions={}
    for label,ref in refs.items():
        path=OUT/'functional'/f'{label}.npz'
        if not path.exists():E.matrices(label,ref)
        heartbeat('commutant_analysis',reference=label,rank=E.r)
        report,Q,sizes=analyze(path);analyses[label]=report;partitions[label]=(Q,sizes)
    # Same locked reference selects a validated partition; otherwise eight global coefficients.
    Q,sizes=partitions['baseline_mean_merge'];structured=analyses['baseline_mean_merge']['structure_validated']
    coeff=np.zeros((8,len(sizes),E.reserve),dtype=np.float32);pos=0
    for m,s in enumerate(sizes):
        q=torch.tensor(Q[:,pos:pos+s],device='cuda',dtype=torch.float32);pos+=s
        coeff[:,m,:E.r]=(zs[:,:E.r]@q@q.T).cpu().numpy()
    reconstruction=float(np.max(np.abs(coeff.sum(1)-zs.cpu().numpy())))
    assert reconstruction<1e-5, reconstruction
    coeff=torch.tensor(coeff,device='cuda')
    def point(lam):
        lam=np.asarray(lam)
        # Preserve the exact learned expert for one-hot reconstruction requests.
        for k in range(8):
            one=np.zeros_like(lam);one[k,:]=1
            if np.array_equal(lam,one):return zs[k].clone()
        return torch.einsum('km,kmd->d',torch.tensor(lam,device='cuda',dtype=torch.float32),coeff)
    for k in range(8):
        one=np.zeros((8,len(sizes)));one[k,:]=1
        assert torch.equal(point(one),zs[k])
    selection_path=OUT/'selection_locked_before_test.json'
    if selection_path.exists():
        lock=json.loads(selection_path.read_text())
        assert lock['rank']==E.r and lock['block_sizes']==sizes
        best=(None,np.array(lock['coefficients']),None)
        merged=point(best[1]);mergeval=lock['merge_validation']
    else:
        # A held-fixed subset of VALIDATION only for an explicitly exploratory pilot.
        pilot=[]
        for k,loader in enumerate(E.validation):
            ds=loader.dataset;ids=np.random.default_rng(915+k).choice(len(ds),min(64,len(ds)),replace=False).tolist()
            pilot.append(torch.utils.data.DataLoader(torch.utils.data.Subset(ds,ids),batch_size=32,num_workers=0))
        def objectives(lam):
            p=point(lam);return np.array([E.evaluate(k,p,pilot[k])['ce'] for k in range(8)])
        history=[];rng=np.random.default_rng(931);shape=(8,len(sizes));best=None
        for index,a in enumerate([.0625,.125,.25,.5]):
            lam=np.full(shape,a);loss=objectives(lam);row=dict(generation=0,index=index,lambda_values=lam.tolist(),validation_probe_ce=loss.tolist(),score=float(loss.mean()))
            history.append(row)
            if best is None or row['score']<best[0]:best=(row['score'],lam,loss)
        # Three generations; population eight. Mean CE selection, all task objectives retained.
        for generation in range(1,4):
            parent=best[1].copy()
            for i in range(8):
                heartbeat('EC_pilot',generation=generation,candidate=i,rank=E.r)
                lam=np.clip(parent+rng.normal(0,.1/(generation**.5),shape),0,1)
                loss=objectives(lam);row=dict(generation=generation,index=i,lambda_values=lam.tolist(),validation_probe_ce=loss.tolist(),score=float(loss.mean()))
                history.append(row)
                if row['score']<best[0]:best=(row['score'],lam,loss)
                write(OUT/'ec_pilot_history.json',history)
        pareto=[]
        for i,row in enumerate(history):
            a=np.array(row['validation_probe_ce'])
            if not any(np.all(np.array(b['validation_probe_ce'])<=a) and np.any(np.array(b['validation_probe_ce'])<a) for j,b in enumerate(history) if j!=i):pareto.append(i)
        merged=point(best[1]);mergeval={}
        for k,task in enumerate(TASKS):
            heartbeat('final_validation',task=task,rank=E.r);mergeval[task]=E.evaluate(k,merged)
        lock=dict(time=time.time(),tasks=TASKS,rank=E.r,expert_selection='minimum full validation CE per task',
            merge_selection='minimum mean CE on fixed validation probe; exploratory EC pilot',structured=structured,
            block_sizes=sizes,coefficients=best[1].tolist(),coordinate_reconstruction_max_error=reconstruction,
            pareto_indices=pareto,merge_validation=mergeval,test_accessed=False)
        write(OUT/'selection_locked_before_test.json',lock)
        save(OUT/'merged_coordinates.pt',dict(z=merged.cpu(),lambda_values=best[1],Q=Q,sizes=sizes))
    # Per-task one-hot reconstruction follows from orthogonal block decomposition.
    # Test outputs are cached so a process restart never reruns a completed selection/test pair.
    from src.datasets.registry import get_dataset
    testdir=OUT/'locked_tests';testdir.mkdir(exist_ok=True);tests={}
    for k,task in enumerate(TASKS):
        ds=get_dataset(task,E.encoder.val_preprocess,str(ROOT/'datasets'),num_workers=0)
        loader=E.u.loader(ds.test_dataset)
        for name,coords in [('expert',zs[k]),('merge',merged)]:
            path=testdir/f'{task}_{name}.json'
            if not path.exists():
                heartbeat('locked_test',task=task,model=name,rank=E.r);write(path,E.evaluate(k,coords,loader))
            tests.setdefault(task,{})[name]=json.loads(path.read_text())
    E.place(zero)
    for k,m in enumerate(E.models):assert E.u.frozen_hash(m)==E.frozen[k]
    result=dict(version='functional_geometry_ec_v2.1',rank=E.r,D=E.D,tasks=TASKS,
        expert_validation=ck['best_validation'],merge_validation=mergeval,tests=tests,
        DTD_validation_80_percent_gate_passed=ck['best_validation'][1]['accuracy']>=.8,
        common_structure_validated=structured,geometry_conclusions={k:r['conclusion'] for k,r in analyses.items()},
        EC_is_exploratory_pilot=True,peak_gpu_bytes=torch.cuda.max_memory_allocated(),time=time.time(),
        limitations=['Functional matrices use 8 images/task/split; bootstrap/null cannot replace larger-sample replication.',
            'No full-finetuning comparator was added; historical DTD comparison is not a new matched baseline.',
            'Failure of the 80% DTD gate must be reported as failure, even if other tasks perform well.'])
    write(OUT/'geometry.json',analyses)
    history={}
    malformed=[]
    for name in ['training','validation','events']:
        path=OUT/f'{name}.jsonl'
        history[name]=[]
        for index,line in enumerate(path.read_text().splitlines() if path.exists() else []):
            try:history[name].append(json.loads(line))
            except json.JSONDecodeError:malformed.append(dict(file=path.name,line=index+1))
    history['malformed_lines_after_interrupted_writes']=malformed
    write(OUT/'history.json',history)
    write(OUT/'result.json',result)
    from report import build
    build(result,history)
    write(OUT/'final_complete.json',dict(time=time.time(),result='result.json',report='run_report.md',evidence='final_evidence.zip'));heartbeat('completed',rank=E.r)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--micro',type=int,default=32);finish(ap.parse_args())
