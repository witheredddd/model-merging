"""Single-task, multi-state projected Fisher family; preserved endpoint residual."""
from pathlib import Path
import argparse,json,sys,hashlib,time
import numpy as np
import torch
from blocks import discover,leakage

def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.30)
    root=Path('/home/shangxueyuan/merging/tsv');source=root/'scripts/fisher_rao_online_v5_98f36a1'
    sys.path.insert(0,str(source));from clip_runtime import ClipTask
    from checkpointing import load_model
    raw=ClipTask(root,a.output,128,32,task='DTD')
    inputs=Path('/media/shangxueyuan/data1/fisher_rao_online_v5_DTD_medium_seed11_98f36a1_20261112')
    paths={}
    for path in (inputs/'checkpoints').glob('*.pt'):
        step=int(path.name.split('_')[1]);paths[step]=path
    states=[0,16,32,48,64,80,96,112,128];train=[1,3,5,7];held=[0,2,4,6,8]
    base=raw.theta0.cpu().double()
    def point(step):
        payload=torch.load(paths[step],map_location='cpu',weights_only=True,mmap=True)
        if payload['identity']['base_sha256']!=raw.rt.input['base_sha256']:raise ValueError('Base mismatch')
        return load_model(payload['state'],raw).parameters().detach()
    final=point(128).cpu().double();tau=final-base
    axes=[]
    def add(v):
        v=v.clone().double();original=float(v.norm())
        for _ in range(2):
            for axis in axes:v.add_(axis,alpha=-float(axis@v))
        norm=float(v.norm())
        if norm>max(original*1e-5,1e-12):axes.append(v/norm)
    add(tau)
    previous=base
    # No fixed dimension cap: actual increments plus ALL stored dictionary axes.
    # The finite candidate pool and numerical rank determine the dimension.
    for step in range(8,129,8):
        current=point(step).cpu().double();add(current-previous);previous=current
    for step in range(0,129,8):
        payload=torch.load(paths[step],map_location='cpu',weights_only=True,mmap=True)
        for direction in payload['state']['S'].T:add(direction)
        dump(a.output/'status.json',dict(stage='common_basis',checkpoint=step,current_numerical_rank=len(axes),dimension_cap=None))
        del payload
    Q=torch.stack(axes);del axes,previous,current
    orth_error=float((Q@Q.T-torch.eye(len(Q),dtype=torch.double)).norm())
    if orth_error>1e-8:raise RuntimeError('Common-space orthogonality failed')
    torch.save(dict(Q_rows=Q,endpoint_step=128,base_sha256=raw.rt.input['base_sha256']),a.output/'common_basis.pt')
    fixed=list(raw.rt.split['X']);independent=list(raw.rt.split['train'][:128]);matrices=[];check=[]
    started=time.monotonic()
    for step in states:
        path=a.output/f'fisher_step_{step:03d}.npz'
        if path.exists():
            cached=np.load(path);matrices.append(cached['F']);check.append(cached['independent_F']);continue
        theta=point(step)
        families=[]
        for ids in (fixed,independent):
            raw.rt.split={**raw.rt.split,'X':ids}
            responses=[]
            for j,direction in enumerate(Q):
                responses.append(raw.jvp(theta,direction.to(device=theta.device,dtype=theta.dtype)).flatten().cpu().double())
                dump(a.output/'status.json',dict(stage='JVP',state=step,column=j+1,dimension=len(Q),independent_probe=ids==independent))
            R=torch.stack(responses,1);families.append((R.T@R).numpy())
        matrices.append(families[0]);check.append(families[1])
        np.savez(path,F=families[0],independent_F=families[1],step=step)
        print(json.dumps(dict(state=step,dimension=len(Q),seconds=time.monotonic()-started)),flush=True)
    raw.rt.split={**raw.rt.split,'X':fixed}
    matrices=np.array(matrices);check=np.array(check)
    result=discover(matrices,train,.05);O=result.pop('Q');labels=result.pop('labels')
    transformed=np.array([O.T@F@O for F in matrices]);tested=np.array([O.T@F@O for F in check])
    blocks=[np.where(labels==x)[0] for x in sorted(set(labels))]
    def pairs(family):
        out=[]
        for A in family:
            values=[]
            for i,b in enumerate(blocks):
                for c in blocks[:i]:
                    values.append(float(np.linalg.norm(A[np.ix_(b,c)])/max(np.sqrt(max(0.,np.trace(A[np.ix_(b,b)])*np.trace(A[np.ix_(c,c)]))),1e-30)))
            out.append(max(values,default=0.))
        return out
    fit_leak=leakage(transformed[train],labels);held_leak=leakage(transformed[held],labels);probe_leak=leakage(tested,labels)
    rng=np.random.default_rng(20261113);random_control=[]
    for _ in range(100):
        rotation=np.linalg.qr(rng.normal(size=(len(Q),len(Q))))[0]
        random_control.append(float(np.mean(leakage(np.array([rotation.T@A@rotation for A in matrices[held]]),labels))))
    coordinates=(Q@tau).numpy();new_coordinates=O.T@coordinates
    residual=tau-Q.T@torch.from_numpy(coordinates)
    reconstructed=residual.clone()
    for indices in blocks:reconstructed.add_(Q.T@torch.from_numpy(O[:,indices]@new_coordinates[indices]))
    torch.save(dict(rotation=torch.from_numpy(O),labels=torch.from_numpy(labels),group_coordinates=[torch.from_numpy(O[:,i]@new_coordinates[i]) for i in blocks],
        full_residual=residual,endpoint_tau=tau,common_basis_file='common_basis.pt'),a.output/'complete_update_decomposition.pt')
    np.savez(a.output/'block_matrices.npz',states=states,F=matrices,rotated_F=transformed,independent_F=check,rotated_independent_F=tested,rotation=O,labels=labels)
    for i,step in enumerate(states):
        np.savetxt(a.output/f'block_matrix_step_{step:03d}.csv',transformed[i],delimiter=',')
    nontrivial=len(blocks)>1
    passed=nontrivial and max(held_leak+probe_leak)<=.05 and max(pairs(transformed[held])+pairs(tested))<=.1
    result.update(task='DTD',trajectory='Completed v5 128-main-attempt trajectory; not a converged expert',
        parameter_scope='48 visual matrices, frozen other weights/head',states=states,training_state_indices=train,held_state_indices=held,
        matrix_definition='Q^T F_D,t Q = (D_theta [2sqrt(p)/sqrt(N)] Q)^T (D_theta [2sqrt(p)/sqrt(N)] Q)',
        dimension=len(Q),orthogonality_error=orth_error,block_sizes=[len(i) for i in blocks],labels=labels.tolist(),
        dimension_cap=None,common_space_candidates='Endpoint first, all 16 actual increments, every dictionary column from all17 stored states; dependent axes removed by twice-reorthogonalized FP64 MGS',
        candidate_nontrivial=nontrivial,held_out_support=bool(passed),fit_leakage=fit_leak,held_leakage=held_leak,independent_probe_leakage=probe_leak,
        held_pair_normalized_coupling=pairs(transformed[held]),independent_pair_normalized_coupling=pairs(tested),
        random_same_sizes_mean_leakage=random_control,selection_uses_held_data=False,
        full_residual_norm=float(residual.norm()),endpoint_update_norm=float(tau.norm()),
        complete_update_reconstruction_error=float((reconstructed-tau).norm()),
        group_update_norms=[float(np.linalg.norm(new_coordinates[i])) for i in blocks],
        fixed_probe_ids=fixed,independent_probe_ids=independent,official_test_opened=False,
        ambient_invariance_claimed=False,loss_conflict_claimed=False,cross_task_diagonalization=False,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    dump(a.output/'results.json',result);dump(a.output/'status.json',dict(stage='complete',held_out_support=bool(passed),block_sizes=result['block_sizes']))
    print(json.dumps(dict(stage='complete',block_sizes=result['block_sizes'],held_out_support=bool(passed))),flush=True)

if __name__=='__main__':main()
