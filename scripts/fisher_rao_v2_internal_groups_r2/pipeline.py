"""DTD-first within-task experiment. Cross-task clustering has no entry point here."""
import argparse
import gc
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from runtime_task import TaskRuntime,read,write,save,load,sha
from fr_math import decoder
from structure import orth_dictionary,reachable,choose_structure,group_bases,parameter_labels,transport,match_groups,overlaps
from learn import train,select,combine,decode,cpu_state,device_state,rank_basis,ablations,paired_ci,adamw_reference,endpoint_diagnostics,group_trajectory


def assign(found,theta):
    offset=0
    with torch.no_grad():
        for _,p in found:p.copy_(theta[offset:offset+p.numel()].view_as(p));offset+=p.numel()
def flat(found):return torch.cat([p.detach().flatten() for _,p in found])


def warmup(rt,seed,folder,steps=100):
    path=folder/'warmup.pt'
    if path.exists():
        state=load(path);rt.theta_w=(rt.theta0+state['offset'].cuda()).detach();return state
    rt.legacy.seed_all(seed);assign(rt.found,rt.theta0)
    for _,p in rt.found:p.requires_grad_(True)
    opt=torch.optim.AdamW([p for _,p in rt.found],lr=1e-5,weight_decay=.01)
    previous=rt.theta0;windows=[];sources=[];start=0
    resume=folder/'warmup_resume.pt'
    if resume.exists():
        state=load(resume);assign(rt.found,rt.theta0+state['offset'].cuda());opt.load_state_dict(state['optimizer'])
        previous=flat(rt.found);windows=state['windows'];sources=state['sources'];start=state['step']
    frozen=rt.legacy.frozen_hash(rt.model)
    for step in range(start,steps):
        b,y,ids=rt.train_batch(seed,step);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(rt.model(b).float(),y)
        loss.backward();torch.nn.utils.clip_grad_norm_([p for _,p in rt.found],1.,error_if_nonfinite=True);opt.step()
        rt.calls['images']+=len(y);rt.calls['training_images']+=len(y)
        if (step+1)%25==0 or step+1==steps:
            current=flat(rt.found);windows.append((current-previous).cpu());previous=current
            sources.append(dict(kind='early_window',start_step=max(0,step+1-25),end_step=step+1))
            save(resume,dict(offset=(current-rt.theta0).cpu(),step=step+1,optimizer=opt.state_dict(),windows=windows,sources=sources))
            rt.status('warmup',seed=seed,step=step+1,budget=steps)
    assert frozen==rt.legacy.frozen_hash(rt.model),'Frozen weights changed during warmup'
    for _,p in rt.found:p.requires_grad_(False)
    for _,p in rt.found:p.grad=None
    rt.theta_w=flat(rt.found).detach();state=dict(offset=(rt.theta_w-rt.theta0).cpu(),windows=windows,sources=sources,
        seed=seed,steps=steps,Vcap={k:float(v.mean()) for k,v in rt.scores(rt.theta_w,'Vcap').items()},
        baseline_Vcap={k:float(v.mean()) for k,v in rt.scores(rt.theta0,'Vcap').items()})
    save(path,state);resume.unlink(missing_ok=True);del opt;return state


def candidates(rt,seed,folder,warm,n=8):
    path=folder/'candidates.pt'
    if path.exists():return load(path)
    vectors=[];metadata=[]
    gen=torch.Generator().manual_seed(seed+707);order=torch.randperm(len(rt.split['train']),generator=gen).tolist()
    for j in range(n):
        ids=[rt.split['train'][i] for i in order[j*32:(j+1)*32]]
        b,y,_=rt.train_batch(seed,2000+j,ids=ids);g,loss=rt.grad(rt.theta_w,b,y)
        norm=float(g.norm());assert norm>1e-12
        v=-g/norm;vectors.append(v.cpu());metadata.append(dict(kind='minibatch',batch=j,sample_ids=ids,native_norm=norm,
             native_CE_derivative=-norm**2,unit_CE_derivative=-norm,loss=loss,anchor_step=warm['steps']))
        rt.status('same_anchor_candidates',seed=seed,candidate=j+1,count=n)
    for vector,source in zip(warm['windows'],warm['sources']):
        size=float(vector.norm())
        if size>1e-12:
            vectors.append(vector/size);metadata.append(dict(**source,native_norm=size,anchor_step=warm['steps'],response_recomputed_at_theta_w=True))
    audit=-vectors[0]*metadata[0]['native_norm']
    for v,m in zip(vectors,metadata):
        derivative=0.
        for a,b in zip(audit.split(1048576),v.split(1048576)):derivative+=float(a.double()@b.double())
        m['shared_audit_CE_derivative']=derivative;m['audit_batch']=0
    state=dict(vectors=vectors,metadata=metadata,seed=seed,anchor_step=warm['steps'],future_endpoint_used=False)
    save(path,state);return state


def geometry(rt,folder,bank):
    path=folder/'geometry.pt'
    if path.exists():return load(path)
    with torch.no_grad():
        qw=rt.phi(rt.theta_w);repeat=rt.phi(rt.theta_w);noise=float(rt.dist(qw,repeat));qv=rt.phi(rt.theta_w,'Vcal')
    responses=[];independent=[];metadata=[];vectors=[]
    for j,(v,meta) in enumerate(zip(bank['vectors'],bank['metadata'])):
        R=rt.jvp(rt.theta_w,v.cuda());Rp=rt.jvp(rt.theta_w,v.cuda(),'Xprime')
        responses.append(R);independent.append(Rp);metadata.append(meta);vectors.append(v)
        rt.status('within_task_FR_responses',candidate=j+1,count=len(bank['vectors']))
    R=torch.stack(responses,1);Rp=torch.stack(independent,1)
    threshold=max(1e-6,10*noise,.01*float(R.norm(dim=0).median()))
    valid=R.norm(dim=0)>threshold
    vectors=[v for v,k in zip(vectors,valid) if k];metadata=[m for m,k in zip(metadata,valid) if k]
    R,Rp=R[:,valid],Rp[:,valid]
    assert sum(m['kind']=='minibatch' for m in metadata)>=4,'Insufficient non-noise independent candidate minibatches'
    S,coeff=orth_dictionary(vectors,16);S=S.cuda();A=R@coeff.to(R.device)
    errors=[]
    for j in range(min(2,S.shape[1])):errors.append(float((rt.jvp(rt.theta_w,S[:,j])-A[:,j]).norm()))
    epsilon_A=max(errors);L,s,V=reachable(A,epsilon_A)
    retained=(L.T@R).square().sum(0)/R.square().sum(0).clamp_min(1e-30)
    # Finite differences and damping are calibrated on Vcal only.
    v=vectors[0].cuda();rv=rt.jvp(rt.theta_w,v,'Vcal');fd=[]
    for h in [.01,.003,.001]:
        with torch.no_grad():finite=(rt.phi(rt.theta_w+h*v,'Vcal')-rt.phi(rt.theta_w-h*v,'Vcal'))/(2*h)
        fd.append(dict(h=h,relative_error=float((finite-rv).norm()/rv.norm())))
    h=min(fd,key=lambda r:r['relative_error'])['h'];assert min(r['relative_error'] for r in fd)<.05,'JVP finite difference failed'
    # True JVP / VJP adjoint check at the same single-task anchor.
    a=torch.randn_like(R[:,0]);jt=rt.vjp(rt.theta_w,a)
    left=a@R[:,0];right=jt.double()@vectors[0].cuda().double()
    adj=float((left-right).abs()/torch.maximum(left.abs(),right.abs()).clamp_min(1e-20));assert adj<.02
    scans=[]
    for gamma in [1e-5,1e-4,1e-3,1e-2]:
        C,_=decoder(A,gamma);err=[]
        for j in range(min(4,R.shape[1])):
            co=C@(h*(L@(L.T@R[:,j])))
            with torch.no_grad():
                target=rt.phi(rt.theta_w+h*vectors[j].cuda(),'Vcal');actual=rt.phi(rt.theta_w+S@co.float(),'Vcal')
                denominator=float(rt.dist(qv,target));err.append(float(rt.dist(actual,target))/max(denominator,1e-10))
        scans.append(dict(gamma=gamma,error=float(np.median(err))))
    structure=choose_structure(R,Rp,L,metadata,rt.classes)
    result=dict(S=S.cpu(),A=A.cpu(),L=L.cpu(),R=R.cpu(),Rprime=Rp.cpu(),qw=qw.cpu(),
        q0=rt.phi(rt.theta0).detach().cpu(),noise=noise,epsilon_A=epsilon_A,singular_values=s.tolist(),
        retained=retained.tolist(),threshold=threshold,metadata=metadata,structure=structure,damping_scan=scans,
        adjoint_error=adj,finite_difference=fd,anchor='theta_w',dictionary_cap=16,
        expanded=False,expansion_reason='prototype has at most 16 early directions; no later experts are substituted')
    save(path,result);write(folder/'structure.json',{k:v for k,v in result.items() if not isinstance(v,torch.Tensor)})
    return result


def controls(geo,M,seed):
    L,R=geo['L'].cuda(),geo['R'].cuda()
    labels=np.array(next(r['labels'] for r in geo['structure']['candidates'] if r['M']==M))
    B=group_bases(L,R,labels);ranks=[U.shape[1] for U in B];total=sum(ranks)
    A=[rank_basis(L,[R],total)]
    rng=np.random.default_rng(seed+819);random=labels[rng.permutation(len(labels))]
    euclidean=parameter_labels(geo['_vectors'],M)
    def fixed_rank(label):
        groups=[]
        for m,rank in enumerate(ranks):groups.append(rank_basis(L,[R[:,label==m]],rank))
        return groups
    return dict(A=A,B=B,C=fixed_rank(random),D=fixed_rank(euclidean),F=B)


def train_suite(rt,folder,geo,bank,M,seed,gamma,steps,budget,mu=0.):
    S,A,L,qw=[geo[k].cuda() for k in ['S','A','L','qw']];C,_=decoder(A,gamma)
    # metadata preserves indices after near-zero rejection; candidate directions are identified by their source.
    valid_vectors=[]
    for meta in geo['metadata']:
        idx=next(i for i,m in enumerate(bank['metadata']) if m==meta);valid_vectors.append(bank['vectors'][idx])
    geo=dict(geo,_vectors=valid_vectors);initializers=controls(geo,M,seed)
    assert len({sum(U.shape[1] for U in initializers[k]) for k in ['A','B','C','D']})==1,'Control coordinate ranks differ'
    runs={};Mfolder=folder/f'M{M}_mu{mu:g}';Mfolder.mkdir(exist_ok=True)
    for label,groups in initializers.items():
        runs[label]=train(rt,S,A,C,L,qw,groups,seed,Mfolder/f'{label}.pt',steps,budget,
                          mu=mu if label in ['A','B','C','D'] else 0.,nu=0.,fixed=label=='F')
    common=min(r['history'][-1]['training_images'] for label,r in runs.items() if label in ['A','B','C','D'])
    chosen={label:select(r,common) for label,r in runs.items()}
    metrics={}
    for label,state in chosen.items():
        U,z=device_state(state,S.device);score=rt.scores(decode(rt.theta_w,S,C,U,z),'Xprime',retain=True)
        metrics[label]=dict(Vcal_ce=state['metrics']['ce'],Xprime_ce=float(score['ce'].mean()),Xprime_accuracy=float(score['acc'].mean()),
             endpoint_FR=float(rt.dist(rt.phi(rt.theta_w,'Xprime'),rt.phi(decode(rt.theta_w,S,C,U,z),'Xprime'))))
    result=dict(M=M,mu=mu,gamma=gamma,common_training_images=common,selected=chosen,metrics=metrics,
         total_coordinate_rank=sum(U.shape[1] for U in initializers['B']),source='early same-task candidates only')
    save(Mfolder/'comparison.pt',result);write(Mfolder/'comparison.json',{k:v for k,v in result.items() if k!='selected'})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/shangxueyuan/merging/tsv'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--task',default='DTD',choices=['DTD'])
    p.add_argument('--warmup',type=int,default=100,choices=[50,100,200]);p.add_argument('--steps',type=int,default=50)
    p.add_argument('--budget',type=int,default=8000);p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,23,37,53]);p.add_argument('--X-n',type=int,default=512)
    p.add_argument('--phase',choices=['initialization','development'],default='development')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=open(args.output/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert args.candidates>=4 and args.steps>0
    config=dict(design='single_task_internal_functional_groups',task=args.task,warmup=args.warmup,steps=args.steps,
        training_forward_images=args.budget,candidates=args.candidates,X_n=args.X_n,seeds=args.seeds,
        code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},spec_sha256=sha(Path(__file__).parent/'EXECUTION_SPEC.md'),
        phase=args.phase,source_scope='same task, <=warmup steps',EC_enabled=False,fiber_enabled=False,test_opened=False)
    cfg=args.output/'config.json'
    if cfg.exists():assert read(cfg)==config,'New source/config requires a new result directory'
    else:write(cfg,config)
    rt=TaskRuntime(args.root,args.output,args.task,args.X_n);geometries={};banks={};warms={}
    for seed in args.seeds:
        if args.phase=='development' and seed in [37,53]:continue
        folder=args.output/f'seed_{seed}';folder.mkdir(exist_ok=True)
        warm=warmup(rt,seed,folder,args.warmup);bank=candidates(rt,seed,folder,warm,args.candidates)
        geometries[seed]=geometry(rt,folder,bank);banks[seed]=bank;warms[seed]=warm
    if args.phase=='initialization':
        write(args.output/'initialization_complete.json',dict(seeds=args.seeds,task=args.task,test_opened=False))
        return
    assert 11 in geometries and 23 in geometries,'Development seeds 11 and 23 are required'
    gamma_rows=[]
    for gamma in [1e-5,1e-4,1e-3,1e-2]:
        gamma_rows.append((gamma,np.median([next(r['error'] for r in geometries[s]['damping_scan'] if r['gamma']==gamma) for s in [11,23]])))
    best=min(r[1] for r in gamma_rows);gamma=max(g for g,e in gamma_rows if e<=best*1.01+1e-12)
    allowed=set.intersection(*[set(r['M'] for r in geometries[s]['structure']['candidates'] if r['qualified']) for s in [11,23]])
    if 1 not in allowed:
        write(args.output/'development_report.json',dict(status='initialization_failed',
           reason='early_dictionary_coverage_or_reachable_rank_insufficient',test_opened=False,
           multi_group_hypothesis_supported=False,fiber_gate=False,eight_task_transfer_gate=False,EC_gate=False))
        rt.status('initialization_failed',reason='dictionary_coverage_or_rank');return
    # DTD prototype tests M=1 plus one geometrically qualified multi-group proposal.
    # Further group counts are separate development runs, not an unlimited model search.
    multi=[m for m in allowed if m>1]
    if multi:
        bestM=max(multi,key=lambda m:np.mean([next(r['silhouette']*r['stability'] for r in geometries[s]['structure']['candidates'] if r['M']==m) for s in [11,23]]))
        allowed={1,bestM}
    comparisons={};matching={}
    for M in sorted(allowed):
        frames=[]
        for seed in [11,23]:
            geo=geometries[seed];labels=np.array(next(r['labels'] for r in geo['structure']['candidates'] if r['M']==M))
            U=group_bases(geo['L'].cuda(),geo['R'].cuda(),labels)
            frames.append([transport(geo['qw'].cuda(),geo['q0'].cuda(),V,rt.classes) for V in U])
        matching[M]=match_groups(*frames)
        if M>1 and matching[M]['minimum_affinity']<.5:continue
        for mu in ([0.,1e-4] if M>1 else [0.]):
            for seed in [11,23]:
                rt.theta_w=(rt.theta0+warms[seed]['offset'].cuda()).detach()
                comparisons[(seed,M,mu)]=train_suite(rt,args.output/f'seed_{seed}',geometries[seed],banks[seed],M,seed,gamma,args.steps,args.budget,mu)
    choices=[]
    for M,mu in sorted({(M,mu) for s,M,mu in comparisons}):
        rows=[comparisons[(s,M,mu)] for s in [11,23]]
        choices.append(dict(M=M,mu=mu,CE=float(np.mean([r['metrics']['B']['Vcal_ce'] for r in rows]))))
    choice=min(choices,key=lambda r:(r['CE'],r['M'],r['mu']))
    # A multi-group interpretation must outperform the same-total-rank controls on BOTH development seeds.
    if choice['M']>1:
        beneficial=all(all(comparisons[(s,choice['M'],choice['mu'])]['metrics']['B']['Vcal_ce']+.002<comparisons[(s,choice['M'],choice['mu'])]['metrics'][c]['Vcal_ce'] for c in ['A','C','D']) for s in [11,23])
        if not beneficial:choice=next(r for r in choices if r['M']==1)
    frozen=dict(**choice,gamma=gamma,development_seeds=[11,23],independent_seeds=[37,53],test_used=False,
                matching={str(m):v for m,v in matching.items()},multi_group_interpretation=choice['M']>1)
    write(args.output/'development_choice.json',frozen)
    reports=[]
    for seed in args.seeds:
        if seed not in geometries:
            folder=args.output/f'seed_{seed}';folder.mkdir(exist_ok=True)
            warm=warmup(rt,seed,folder,args.warmup);bank=candidates(rt,seed,folder,warm,args.candidates)
            geometries[seed]=geometry(rt,folder,bank);banks[seed]=bank;warms[seed]=warm
        rt.theta_w=(rt.theta0+warms[seed]['offset'].cuda()).detach();geo=geometries[seed]
        init_record=next((r for r in geo['structure']['candidates'] if r['M']==choice['M']),None)
        if init_record is None:
            reports.append(dict(seed=seed,supported=False,reason='frozen_group_count_not_representable_at_independent_anchor'))
            continue
        key=(seed,choice['M'],choice['mu'])
        result=comparisons.get(key) or train_suite(rt,args.output/f'seed_{seed}',geo,banks[seed],choice['M'],seed,gamma,args.steps,args.budget,choice['mu'])
        S,A,L,qw=[geo[k].cuda() for k in ['S','A','L','qw']];C,_=decoder(A,gamma)
        folder=args.output/f'seed_{seed}'/f'M{choice["M"]}_mu{choice["mu"]:g}'
        necessity=ablations(rt,S,A,C,L,qw,result['selected']['B'],seed,folder,args.steps,args.budget)
        reference=adamw_reference(rt,seed,folder/'full_AdamW_reference.pt',result['common_training_images'])
        endpoint=endpoint_diagnostics(rt,S,C,result,reference,folder)
        group_trajectory(rt,S,C,load(folder/'B.pt'),folder)
        state=result['selected']['B'];U,z=device_state(state,S.device);main_score=rt.scores(decode(rt.theta_w,S,C,U,z),'Xprime')
        controls_CI={}
        for method in ['A','C','D']:
            Ug,zg=device_state(result['selected'][method],S.device)
            score=rt.scores(decode(rt.theta_w,S,C,Ug,zg),'Xprime')
            controls_CI[method]=paired_ci((score['ce']-main_score['ce']).numpy(),seed)
        # Independent structure replication is assessed after hyperparameters are frozen.
        current=[transport(geo['qw'].cuda(),geo['q0'].cuda(),V,rt.classes) for V in U]
        dev_geo=geometries[11];dev_labels=np.array(next(r['labels'] for r in dev_geo['structure']['candidates'] if r['M']==choice['M']))
        dev_U=group_bases(dev_geo['L'].cuda(),dev_geo['R'].cuda(),dev_labels)
        replication=match_groups([transport(dev_geo['qw'].cuda(),dev_geo['q0'].cuda(),V,rt.classes) for V in dev_U],current)
        final_overlap=overlaps(U);collapsed=any(min(r['projection_i_into_j'],r['projection_j_into_i'])>=.95 for r in final_overlap)
        supported=init_record['qualified'] and necessity['necessary'] and all(ci[0]>0 for ci in controls_CI.values()) and replication['minimum_affinity']>=.5 and not collapsed
        reports.append(dict(seed=seed,comparison=result['metrics'],necessity=necessity['necessary'],M=choice['M'],
              control_gain_CI=controls_CI,replication=replication,collapsed=collapsed,endpoint_diagnostics=endpoint,supported=supported))
    review=[r for r in reports if r['seed'] in [37,53]]
    passed=choice['M']>1 and len(review)==2 and all(r['supported'] for r in review)
    write(args.output/'development_report.json',dict(status='complete',reports=reports,selection=frozen,
        multi_group_hypothesis_supported=passed,fiber_gate=passed,eight_task_transfer_gate=passed,
        EC_gate=False,reason='EC requires eight separately validated task structures',test_opened=False,
        expert_endpoints_used_for_initialization=False,calls=rt.calls))
    rt.status('development_complete',multi_group_hypothesis_supported=passed)


if __name__=='__main__':main()
