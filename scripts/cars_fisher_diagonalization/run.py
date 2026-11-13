"""Cars learned-carrier Fisher spectrum/diagonalization. No support profiling."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output
    out.mkdir(exist_ok=True,parents=True);private=out/'private';private.mkdir(exist_ok=True)
    root=Path('/home/shangxueyuan/merging/tsv')
    sys.path[:0]=[str(root),str(root/'scripts'),str(root/'scripts/fr_hst'),str(root/'scripts/tfpml'),str(root/'scripts/dtd_fisher_blocks_expanded_20261113')]
    import dtd_grassmann4 as legacy
    from runtime import build_model
    from task_data import train_set,test_set
    from train_fd_task import tangent
    from blocks import discover,leakage
    legacy.ROOT=root
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False)
    folder=root/'results/EXPERTS8_FD_REVIEWED_20261110/Cars/adaptive_dynamics'
    metadata=json.loads((folder/'complete.json').read_text());historical=json.loads((folder/'historical_test.json').read_text())
    checkpoint=folder/'epoch08.pt';digest=sha(checkpoint);assert digest==metadata['checkpoint_sha256']==historical['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
    frozenfile=folder/'frozen_decoder/epoch08.pt';frozen=torch.load(frozenfile,map_location='cpu',mmap=True,weights_only=True)
    head=root/'results/functional_geometry_v2_no_expert/assets/heads/Cars.pt'
    assert sha(head)==metadata['provenance']['head_sha256']
    assert sha(root/'models/checkpoints/ViT-B-16/pretrained.pt')==metadata['provenance']['base_sha256']
    encoder,model,_,unused=build_model(legacy,legacy.ClassificationHead.load(str(head)));del unused
    model.requires_grad_(False).eval();torch.set_grad_enabled(False)
    items=list(encoder.model.visual.named_parameters());keys=[n for n,_ in items];sizes=[v.numel() for _,v in items];shapes=[v.shape for _,v in items]
    names=['image_encoder.model.visual.'+n for n in keys]
    base=torch.cat([v.detach().flatten().cpu() for _,v in items]).double();D=len(base)
    assert saved['frames'].shape[2]==D
    def flatten(visual):return torch.cat([visual[n].flatten() for n in keys]).double()
    expert=flatten(saved['visual']);decoded=flatten(frozen['visual'])
    data=train_set('Cars',encoder.val_preprocess);test=test_set('Cars',encoder.val_preprocess)
    assert len(data)==8144 and len(test)==8041 and model.classification_head.weight.shape[0]==196
    order=torch.randperm(len(data),generator=torch.Generator().manual_seed(20261114)).tolist()
    probes=[order[:128],order[128:256]]
    config=dict(source_sha256=sha(__file__),block_solver_sha256=sha(discover.__code__.co_filename),checkpoint_sha256=digest,frozen_decoder_sha256=sha(frozenfile),
        provenance=metadata['provenance'],head_sha256=sha(head),D=D,parameter_keys=keys,parameter_shapes=[list(s) for s in shapes],
        probes=probes,probe_origin='Two disjoint training-side probes; original expert trained on all8144, so not unseen training data',
        candidate='ALL saved nonlinear frame axes, full expert update, and full frozen-decoder update; numerical rank only, no dimension cap',
        states=['pretrained','saved_frozen_decoder','expert'],states_are_training_trajectory=False,
        joint_block_fit_state_indices=[0,2],joint_block_held_state_indices=[1],
        note='No old DTD v5 space, no preset Fisher rank, no masks/leverage/support profile, no cross-task conflict claims',
        historical_reference=historical,learned_components=len(saved['frames']),coordinate_count=2*len(saved['frames']),
        original_component_cap=metadata['provenance']['max_components'],component_cap_reached=len(saved['frames'])==metadata['provenance']['max_components'],
        decoder_relative_endpoint_error=metadata['learned'].get('decoder_relative_endpoint_error'))
    cfg=out/'config.json'
    if cfg.exists():assert json.loads(cfg.read_text())==config
    else:write(cfg,config)
    Qfile=private/'candidate_basis.pt'
    if Qfile.exists():
        q=torch.load(Qfile,map_location='cpu',mmap=True,weights_only=True);Q=q['Q_rows'];frame_coordinates=q['frame_coordinates']
    else:
        axes=[];coefficients=[];candidate_info=[]
        vectors=list(saved['frames'].flatten(0,1))+[expert-base,decoded-base]
        for i,original in enumerate(vectors):
            v=original.double().clone();norm=float(v.norm());coeff=[]
            for _ in range(2):
                for j,axis in enumerate(axes):
                    a=float(axis@v);v.add_(axis,alpha=-a)
                    if len(coeff)<=j:coeff.extend([0.]*(j+1-len(coeff)))
                    coeff[j]+=a
            residual=float(v.norm())
            if residual>max(1e-5*norm,1e-12):
                axes.append(v/residual);coeff.append(residual)
            coefficients.append(coeff);candidate_info.append(dict(index=i,norm=norm,residual=residual))
            write(out/'status.json',dict(stage='candidate_basis',candidate=i+1,rank=len(axes)))
        Q=torch.stack(axes);frame_coordinates=torch.zeros(len(Q),saved['frames'].shape[0]*3,dtype=torch.double)
        for i,coeff in enumerate(coefficients[:frame_coordinates.shape[1]]):frame_coordinates[:len(coeff),i]=torch.tensor(coeff,dtype=torch.double)
        torch.save(dict(Q_rows=Q,frame_coordinates=frame_coordinates,candidates=candidate_info),Qfile)
    d=len(Q);orth=float((Q@Q.T-torch.eye(d,dtype=torch.double)).norm());assert orth<1e-8
    T=frame_coordinates@torch.from_numpy(tangent(np.array(saved['z']),np.array(saved['q'])))*saved['scale']
    Ut,st,_=torch.linalg.svd(T,full_matrices=False);keep=st>st[0]*1e-6;Torth=Ut[:,keep]
    geometry=dict(candidate_dimension=d,candidate_dimension_cap=None,orthogonality_error=orth,
        actual_coordinate_tangent_singular_values=st.tolist(),actual_coordinate_tangent_rank=int(keep.sum()),
        tangent_relative_singular_cutoff=1e-6,
        distinction='Candidate frame span plus updates versus current2M nonlinear coordinate tangent; neither is a full-space Fisher eigenspace')
    write(out/'geometry.json',geometry)
    def predict(point,x):
        overrides={n:v.view(shape) for n,v,shape in zip(names,point.split(sizes),shapes)}
        return torch.func.functional_call(model,overrides,(x,)).float()
    def phi(point,batches,N):
        return torch.cat([(2/np.sqrt(N)*predict(point,x.cuda()).double().softmax(-1).sqrt()).flatten() for x,_ in batches])
    batches=[[(x,y) for x,y in DataLoader(Subset(data,ids),batch_size=32,num_workers=4)] for ids in probes]
    def logits(point,batch):return torch.cat([predict(point,x.cuda()).double().cpu() for x,_ in batch])
    def scores(z,y):return dict(CE=float(F.cross_entropy(z,y)),accuracy=float((z.argmax(1)==y).double().mean()),n=len(y))
    states=[base,decoded,expert];families=[]
    for state_index,point_cpu in enumerate(states):
        point=point_cpu.float().cuda();pair=[]
        for probe in range(2):
            path=private/f'response_{state_index}_{probe}.npz'
            if path.exists():R=torch.from_numpy(np.load(path)['R'])
            else:
                columns=[]
                for j,axis in enumerate(Q):
                    _,r=torch.func.jvp(lambda p:phi(p,batches[probe],128),(point,),(axis.float().cuda(),))
                    columns.append(r.cpu().double());write(out/'status.json',dict(stage='JVP',state=state_index,probe=probe,column=j+1,dimension=d))
                R=torch.stack(columns,1);np.savez(path,R=R.numpy())
            pair.append((R.T@R).numpy())
        families.append(pair)
    families=np.array(families);Fexpert=families[2,0];eigen,V=np.linalg.eigh(Fexpert);order=np.argsort(eigen)[::-1];eigen=eigen[order];V=V[:,order]
    transformed=np.array([[V.T@A@V for A in pair] for pair in families])
    positive=np.maximum(eigen,0);weights=positive/positive.sum()
    evals,Vheld=np.linalg.eigh(families[2,1]);Vheld=Vheld[:,::-1]
    spectrum=dict(eigenvalues=eigen.tolist(),effective_rank_relative1e_6=int((eigen>eigen[0]*1e-6).sum()),
        participation_rank=float(positive.sum()**2/(positive@positive)),entropy_rank=float(np.exp(-(weights*np.log(np.maximum(weights,1e-300))).sum())),
        trace_energy_dimensions={str(q):int(np.searchsorted(np.cumsum(weights),q))+1 for q in [.8,.9,.95,.99]},
        eigenvector_same_order_abs_cosines=np.abs(np.diag(V.T@Vheld)).tolist(),
        expert_fit_offdiagonal_ratio=float(np.linalg.norm(transformed[2,0]-np.diag(np.diag(transformed[2,0])))/np.linalg.norm(transformed[2,0])),
        expert_other_probe_offdiagonal_ratio=float(np.linalg.norm(transformed[2,1]-np.diag(np.diag(transformed[2,1])))/np.linalg.norm(transformed[2,1])),
        tangent_Fisher_eigenvalues={str(i):np.linalg.eigvalsh(Torth.numpy().T@families[i,0]@Torth.numpy())[::-1].tolist() for i in range(3)})
    block=discover(families[:,0],[0,2],.05);rotation=block.pop('Q');labels=block.pop('labels')
    rotated=np.array([[rotation.T@A@rotation for A in pair] for pair in families])
    leakage_values=[leakage(rotated[:,probe],labels) for probe in range(2)]
    block.update(labels=labels.tolist(),sizes=[int((labels==i).sum()) for i in np.unique(labels)],
        leakage=leakage_values,held_support=bool(len(set(labels))>1 and max(leakage_values[0][1:2]+leakage_values[1])<=.05),
        caveat='Common-block diagnostic across saved model variants, not training trajectory or cross-task validation')
    write(out/'spectrum.json',spectrum);write(out/'blocks.json',block)
    np.savez(out/'matrices.npz',F=families,eigenbasis=V,eigenvalues=eigen,diagonalized=transformed,
             common_rotation=rotation,common_rotated=rotated,block_labels=labels,tangent_coordinates=Torth.numpy())
    for i,name in enumerate(config['states']):
        for probe in range(2):
            np.savetxt(out/f'raw_{name}_probe{probe}.csv',families[i,probe],delimiter=',')
            np.savetxt(out/f'diagonalized_{name}_probe{probe}.csv',transformed[i,probe],delimiter=',')
    # Actual finite prediction interventions in every fitted eigenmode; no optimizer.
    point=expert.float().cuda();baseline=[logits(point,b) for b in batches];finite=[]
    for j in range(d):
        path=out/f'finite_mode_{j:02d}.json'
        if path.exists():finite.extend(json.loads(path.read_text()));continue
        v=torch.zeros(D,device='cuda')
        for k in range(d):v.add_(Q[k].float().cuda(),alpha=float(V[k,j]))
        v/=v.norm();rows=[]
        for radius in [.001,.003]:
            for sign in [-1,1]:
                for probe in range(2):
                    z=logits(point+sign*radius*v,batches[probe]);oldz=baseline[probe]
                    actual=float((oldz.softmax(1)*(oldz.log_softmax(1)-z.log_softmax(1))).sum(1).mean())
                    predicted=.5*radius**2*float(V[:,j]@families[2,probe]@V[:,j])
                    y=torch.cat([y for _,y in batches[probe]])
                    rows.append(dict(mode=j,radius=radius,sign=sign,probe=probe,KL=actual,predicted_KL=predicted,
                        CE_delta=float(F.cross_entropy(z,y)-F.cross_entropy(oldz,y))))
        write(path,rows);finite.extend(rows);write(out/'status.json',dict(stage='finite_KL',mode=j+1,dimension=d))
    # Locked historical Test is for checkpoint reproduction, never for selecting directions.
    test_batches=DataLoader(test,batch_size=64,num_workers=4);total=correct=0;ce=0.
    for x,y in test_batches:
        z=predict(point,x.cuda()).double().cpu();correct+=int((z.argmax(1)==y).sum());ce+=float(F.cross_entropy(z,y,reduction='sum'));total+=len(y)
    reproduction=dict(n=total,correct=correct,accuracy=correct/total,CE=ce/total,
        historical_reference=historical,used_for_selection=False)
    report=dict(config=config,geometry=geometry,spectrum=spectrum,common_blocks=block,finite_interventions=finite,
        test_reproduction=reproduction,training=False,subspace_not_full_Fisher=True,
        DTD_status='82.07% adaptive_dynamics structure available;81.86% adaptive_allocation has archived metrics but checkpoint not located here; these are distinct versions')
    write(out/'report.json',report);write(out/'status.json',dict(stage='complete',dimension=d,training=False))


if __name__=='__main__':main()
