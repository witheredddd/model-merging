"""Task-driven sufficiency, conditional substitution and signed compatibility."""
import argparse,hashlib,json,signal,time,sys,os,fcntl
from pathlib import Path
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator
from bootstrap import CORE,LEGACY,R3,SPACE,FLOW
from carrier import TrainingPositions
from io_state import atomic_json,digest
from geometry import FullFunctionalGeometry
from iterative import continuing_cg
from numerics import sphere_log
from math_core import class_weights,gram_modes,metric_stats

BUILD=['mean']+[f'{trig}_{freq}' for freq in range(1,8) for trig in ('sin','cos')]
UNSEEN=['mean','sin_8','cos_8','sin_9','cos_9']

class Experiment:
    def __init__(self,a):
        self.a=a;self.out=a.output;self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'WRITER.lock').open('ab');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.stop=False;signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True))
        self.started=time.monotonic();self.cache={};self.verified=set();self.baselines={};self.rows=[]
        sources={str(p):digest(p) for folder in (Path(__file__).parent,CORE,LEGACY,R3,SPACE) for p in folder.glob('*.py')}
        config=dict(schema='task-substructure-diagnostic-v1',sources=sources,
            inputs={str(p):digest(p) for p in (a.inventory,a.probes,a.coordinates,a.trace/'trace.json')},
            anchors=a.anchors,build=BUILD,unseen=UNSEEN,radii=[.0025,.01,.04],alphas=[.1,1.,10.],
            numerical_rtol=a.rtol,evaluation_batch_size=a.batch_size,dimension_cap=None,official_test_opened=False,
            numerical_bounds_assume_exact_fixed_operator_and_projector=True,floating_point_operator_error_not_enclosed=True,
            final_expert_training=False,EC_search=False,batch_size=a.batch_size,
            storage_policy='Preserve gradients/bases/solvers and exact control recipes; archive transient controls after each anchor')
        cfg=self.out/'CONFIG.json'
        if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Frozen configuration/source changed')
        atomic_json(cfg,config)
        self.positions=TrainingPositions(a.trace,json.loads(a.coordinates.read_text())['parameter_keys'],self.out)
        atomic_json(self.out/'ANCHOR_PROVENANCE.json',self.positions.verify_normal_recipe())
        import task_runtime,shutil
        runtime=self.out/'runtime';runtime.mkdir(exist_ok=True);shutil.copy2(a.probes,runtime/'probes.json')
        inventory=json.loads(a.inventory.read_text())['tasks'];inventory=[next(r for r in inventory if r['task']==t) for t in ('DTD','Cars')]
        old=task_runtime.TASKS;task_runtime.TASKS=('DTD','Cars')
        try:c=task_runtime.EightTaskContext(runtime,inventory)
        finally:task_runtime.TASKS=old
        for key,batches in list(c.batches.items()):
            c.batches[key]=[(x[i:i+a.batch_size],y[i:i+a.batch_size]) for x,y in batches for i in range(0,len(y),a.batch_size)]
        self.c=c;self.shapes=c.shapes;self.keys=c.keys
        self.ranges=[];offset=0
        for i,(name,n) in enumerate(zip(c.keys,c.sizes)):
            self.ranges.append(dict(index=i,name=name,start=offset,stop=offset+n,dimension=n));offset+=n
        atomic_json(self.out/'PARAMETER_REGIONS.json',self.ranges)
        point=self.positions.flat(self.index(a.anchors[0])).cuda().double()
        self.g=FullFunctionalGeometry(c,point,0,precision='float64');self.classes={}
        counts={}
        for task in self.g.tasks:
            logits,labels=self.g.c.baseline(task,0,point);self.classes[task]=logits.shape[1]
            # Class-stratified, disjoint selection and final evaluation subsets.
            x=torch.cat([x for x,y in self.g.c.batches[task,1]]);y=torch.cat([y for x,y in self.g.c.batches[task,1]])
            indices=[[],[]]
            for cls in y.unique().tolist():
                ids=(y==cls).nonzero().flatten().tolist()
                for k,i in enumerate(ids):indices[k%2].append(i)
            for probe,ids in zip((2,3),indices):
                self.g.c.batches[task,probe]=[(x[ids[i:i+a.batch_size]],y[ids[i:i+a.batch_size]]) for i in range(0,len(ids),a.batch_size)]
            counts[task]={str(p):sum(len(y) for x,y in self.g.c.batches[task,p]) for p in (0,2,3)}
            assert not set(indices[0])&set(indices[1])
        atomic_json(self.out/'DATA_SCOPE.json',dict(counts=counts,roles={'0':'basis construction and quadratic objective','2':'unseen demand control and validation selection','3':'final evaluation only'},
            training_side_development_probes=True,unseen_to_expert_training_not_claimed=True,official_test=False,
            selection_evaluation_disjoint=True,all_reported_demands_predeclared=True))
        self.point=point;self.anchor=a.anchors[0]

    def status(self,stage,**extra):
        snapshot=dict(stage=stage,anchor=self.anchor,elapsed_this_process=time.monotonic()-self.started,
            operator=self.g.operator.audit(),stop_requested=self.stop,process_id=os.getpid(),wall_time=time.time(),**extra)
        atomic_json(self.out/'status.json',snapshot)
        with (self.out/'TIMING_HISTORY.jsonl').open('a') as f:f.write(json.dumps(snapshot)+'\n')
        print(json.dumps(dict(stage=stage,anchor=self.anchor,**extra)),flush=True)
        if self.stop:raise InterruptedError()
        import shutil
        free=shutil.disk_usage(self.out).free
        if free<30*1024**3:
            atomic_json(self.out/'RESOURCE_LIMIT.json',dict(free_bytes=free,resumable=True,
                reason='Less than 30 GiB free; no further parameter files allocated',dimension_was_not_truncated=True))
            raise InterruptedError()
    def index(self,step):return next(i for i,r in enumerate(self.positions.rows) if r['step']==step)
    def set_anchor(self,step):
        self.anchor=step;self.point=self.positions.flat(self.index(step)).cuda().double()
        self.g.point=self.point;self.g.operator.point=self.point;self.g.c.anchor=self.point;self.baselines={}
    def load(self,ref,device='cuda'):
        path=self.out/ref['file']
        if ref['sha256'] not in self.verified:
            if digest(path)!=ref['sha256']:raise ValueError('Vector checksum mismatch')
            self.verified.add(ref['sha256'])
        return torch.load(path,map_location=device,mmap=True,weights_only=True)['vector']
    def save_vector(self,name,v):
        p=self.out/(name+'.pt');p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
        torch.save(dict(vector=v.detach().cpu().double()),tmp);h=digest(tmp);p=p.with_name(p.stem+'_'+h[:12]+'.pt')
        if p.exists():tmp.unlink()
        else:tmp.replace(p)
        self.verified.add(h);return dict(file=str(p.relative_to(self.out)),sha256=h,dimension=v.numel())
    def vector_cached(self,name,compute):
        meta=self.out/(name+'.json')
        if meta.exists():
            ref=json.loads(meta.read_text())
            if (self.out/ref['file']).exists():return ref
        ref=self.save_vector(name,compute());atomic_json(meta,ref);return ref
    def grad(self,task,probe,demand):
        def compute():
            started=time.monotonic();batches=self.g.c.batches[task,probe]
            weights=torch.tensor(class_weights(self.classes[task],demand),device='cuda')
            total=sum(float(weights[y].sum()) for x,y in batches);gradient=torch.zeros_like(self.point)
            for x,y in batches:
                with torch.enable_grad():
                    leaf=self.point.detach().requires_grad_(True);z=self.g.c.predict(leaf,x,task)
                    loss=(torch.nn.functional.cross_entropy(z,y,reduction='none')*weights[y]).sum()/total
                    gradient+=torch.autograd.grad(loss,leaf)[0].detach()
            with (self.out/'gradient_timing.jsonl').open('a') as f:f.write(json.dumps(dict(task=task,probe=probe,demand=demand,anchor=self.anchor,seconds=time.monotonic()-started))+'\n')
            return gradient
        return self.vector_cached(f'gradients/a{self.anchor}_{task}_p{probe}_{demand}',compute)
    def j(self,v,task='DTD',probe=0):
        self.g.probe=probe;self.g.output_dimensions[task]=sum(len(y) for x,y in self.g.c.batches[task,probe])*self.classes[task]
        return self.g.jvp(task,v)
    def jt(self,y):
        self.g.probe=0;self.g.output_dimensions['DTD']=len(y);return self.g.vjp('DTD',y)
    def combine(self,refs,coeff):
        v=torch.zeros_like(self.point)
        for r,c in zip(refs,coeff):
            if c:v+=float(c)*self.load(r)
        return v
    def dot_basis(self,refs,v):
        cpu=v.detach().cpu();return np.array([float(self.load(r,'cpu')@cpu) for r in refs])
    def project(self,v,refs):return self.combine(refs,self.dot_basis(refs,v))
    def build_basis(self):
        meta=self.out/f'bases/a{self.anchor}.json'
        if meta.exists():return json.loads(meta.read_text())
        refs=[]
        for name in BUILD:
            self.status('GRADIENT_CONSTRUCTION',demand=name);refs.append(self.grad('DTD',0,name))
        gram=np.array([[float(self.load(a,'cpu')@self.load(b,'cpu')) for b in refs] for a in refs])
        values,C,report=gram_modes(gram/len(BUILD));basis=[];basis_recipes=[]
        for j in range(C.shape[1]):
            v=self.combine(refs,C[:,j]/np.sqrt(len(BUILD)))
            passes=[]
            for _ in range(2):
                coefficients=[]
                for r in basis:
                    b=self.load(r);coefficient=float(b@v);coefficients.append(coefficient);v-=coefficient*b
                passes.append(coefficients)
            norm=float(v.norm())
            if not norm:
                report.setdefault('arithmetic_dependent_columns',[]).append(j);continue
            ref=self.save_vector(f'bases/a{self.anchor}_mode{j}',v/norm);basis.append(ref)
            basis_recipes.append(dict(file=ref['file'],input_coefficients=C[:,j]/np.sqrt(len(BUILD)),
                orthogonalization_passes=passes,normalization=norm))
        physical_gram=np.array([[float(self.load(a,'cpu')@self.load(b,'cpu')) for b in basis] for a in basis])
        info=dict(gradients=refs,basis=basis,mean_gradient=refs[0],singular_values_squared=values,
            weighted_gram=gram/len(BUILD),physical_basis_gram=physical_gram,basis_reconstruction=basis_recipes,
            max_orthogonality_error=float(np.max(np.abs(physical_gram-np.eye(len(basis))))),
            sample_demand_rank_bound=len(BUILD),not_intrinsic_dimension=True,**report)
        centering=np.eye(len(BUILD))-np.ones_like(gram)/len(BUILD)
        mean_removed=gram-np.outer(gram[:,0],gram[0,:])/gram[0,0]
        info.update(centered_family_spectrum=np.linalg.eigvalsh(centering@gram@centering/len(BUILD))[::-1],
            mean_CE_removed_spectrum=np.linalg.eigvalsh(mean_removed/len(BUILD))[::-1],
            mean_CE_energy_fraction_per_training_demand=gram[0,:]**2/(gram[0,0]*np.diag(gram)))
        atomic_json(meta,info);return json.loads(meta.read_text())
    def logits(self,point,task,probe):return self.g.c.baseline(task,probe,point)
    def evaluate(self,key,ref,scale,demand,metadata):
        dest=self.out/f'evaluations/{key}.json'
        if dest.exists():return json.loads(dest.read_text())
        case_started=time.monotonic();before=self.g.operator.audit();before=dict(before,seconds=dict(before['seconds']))
        self.status('FINITE_INTERVENTION',case=key)
        v=self.load(ref);delta=scale*v;point=self.point+delta;arrays={};metrics={}
        norm=float(delta.norm())
        rounded_point=point.float().double();rounded_delta=rounded_point-self.point
        rounding=dict(parameter_float32_changed_fraction=float((rounded_delta!=0).double().mean()),
            parameter_float32_step_norm=float(rounded_delta.norm()),
            parameter_float32_rounding_relative_error=float((rounded_point-point).norm())/norm if norm else None,
            actual_model_precision='float64',float32_metrics_evaluated=False)
        del rounded_point,rounded_delta
        for task in ('DTD','Cars'):
            for probe in (0,2,3):
                if (task,probe) not in self.baselines:self.baselines[task,probe]=self.logits(self.point,task,probe)
                base,labels=self.baselines[task,probe];z,y=self.logits(point,task,probe)
                assert torch.equal(y,labels)
                w=class_weights(self.classes[task],demand if task=='DTD' else 'mean')
                metric=metric_stats(z.numpy(),y.numpy(),w,base.numpy())
                response_file=self.out/'linear_responses'/f'a{self.anchor}_{ref["sha256"]}_{task}_p{probe}.npz'
                if response_file.exists():
                    with np.load(response_file) as saved:unit_response=saved['JVP']
                else:
                    unit_response=self.j(v,task,probe).cpu().numpy();response_file.parent.mkdir(exist_ok=True)
                    np.savez_compressed(response_file,JVP=unit_response)
                n=len(y);phi0=2*base.softmax(-1).sqrt().numpy()/np.sqrt(n);phi1=2*z.softmax(-1).sqrt().numpy()/np.sqrt(n)
                actual=sphere_log(phi0,phi1).reshape(-1);predicted=scale*unit_response
                den=float(np.linalg.norm(predicted));absolute=float(np.linalg.norm(actual-predicted))
                slope=-2*predicted.reshape(n,-1)[np.arange(n),y.numpy()]/np.maximum(phi0[np.arange(n),y.numpy()],np.finfo(float).tiny)
                sample_weights=w[y.numpy()]
                metric.update(actual_Log_norm=float(np.linalg.norm(actual)),predicted_response_norm=den,
                    absolute_Log_JVP_error=absolute,relative_Log_JVP_error=absolute/den if den else None,
                    first_order_weighted_CE=float(sample_weights@slope/sample_weights.sum()),
                    CE_linearization_error=metric['delta_weighted_CE']-float(sample_weights@slope/sample_weights.sum()))
                metrics[f'{task}_p{probe}']=metric
                arrays[f'{task}_p{probe}_predicted_response']=predicted;arrays[f'{task}_p{probe}_actual_Log']=actual
                arrays[f'{task}_p{probe}_logits']=z.numpy();arrays[f'{task}_p{probe}_labels']=y.numpy()
                arrays[f'{task}_p{probe}_baseline']=base.numpy()
        dest.parent.mkdir(exist_ok=True,parents=True);arr=dest.with_suffix('.npz');np.savez_compressed(arr,**arrays)
        after=self.g.operator.audit()
        r=dict(case=key,anchor=self.anchor,demand=demand,vector=ref,scale=scale,parameter_norm=norm,
            process_id=os.getpid(),case_wall_seconds=time.monotonic()-case_started,**rounding,
            case_JVP_calls=after['JVP_calls']-before['JVP_calls'],case_VJP_calls=after['VJP_calls']-before['VJP_calls'],
            case_J_host_seconds=after['seconds']['J']-before['seconds']['J'],case_JT_host_seconds=after['seconds']['JT']-before['seconds']['JT'],
            metrics=metrics,arrays=str(arr.relative_to(self.out)),array_sha256=digest(arr),
            physical_full_parameter_evaluation=True,parameter_state_recipe='theta = exact recorded anchor + scale * vector',
            official_test=False,accuracy_acceptance=False,**metadata)
        atomic_json(dest,r);return r
    def supports(self,gD,gC):
        records=[]
        for region in self.ranges:
            seg=slice(region['start'],region['stop']);gd=gD[seg];gc=gC[seg]
            records.append(dict(**region,DTD_gradient_energy=float(gd@gd),Cars_gradient_energy=float(gc@gc),
                signed_inner=float(gd@gc),euclidean_deletion=.5*float(gd@gd)))
        atomic_json(self.out/f'regions/a{self.anchor}.json',dict(regions=records,
            euclidean_deletion_is_gradient_energy_not_functional_necessity=True))
        top=max(records,key=lambda r:r['DTD_gradient_energy'])
        conflict=[r for r in records if r['signed_inner']<0]
        result={'top_native_region':[top['index']], 'signed_conflict_union':[r['index'] for r in conflict]}
        # Size-matched deterministic coordinate control, no data-driven support truncation.
        n=top['dimension'];gen=torch.Generator(device='cpu');gen.manual_seed(3301)
        start=int(torch.randint(len(gD)-n+1,(1,),generator=gen));result['matched_contiguous_control']={'start':start,'stop':start+n}
        first=self.a.anchors[0]
        if self.anchor!=first:
            original=json.loads((self.out/f'regions/a{first}_selected.json').read_text())['supports']
            result['frozen_top_region_from_anchor0']=original['top_native_region']
            result['frozen_conflict_union_from_anchor0']=original['signed_conflict_union']
        atomic_json(self.out/f'regions/a{self.anchor}_selected.json',dict(supports=result,selection_probe=0,
            top_native_region_is_diagnostic_not_discovered_group=True,conflict_union_by_sign_not_magnitude_cut=True,
            matched_control_is_not_architecture_equivalent=True))
        return result
    def projector(self,support=None,basis=None):
        if basis is not None:
            def apply(v):return v-self.project(v,basis)
        elif support is None:
            def apply(v):return v
        else:
            def apply(v):
                v=v.clone()
                if isinstance(support,dict):v[support['start']:support['stop']]=0
                else:
                    for i in support:
                        r=self.ranges[i];v[r['start']:r['stop']]=0
                return v
        return apply
    def solve(self,key,g,beta,P):
        meta=self.out/f'solves/{key}.json'
        if meta.exists():return json.loads(meta.read_text())
        self.status('FISHER_CONDITIONAL_SOLVE',case=key,beta=beta)
        folder=meta.parent;folder.mkdir(exist_ok=True);ck=folder/(key+'_CG.npz')
        pg=P(g);pg_norm=float(pg.norm())
        arithmetic=np.finfo(float).eps*(np.log2(self.g.dimension)+16)*256*max(float(g.norm()),np.finfo(float).tiny)
        if pg_norm<=arithmetic:
            ref=self.save_vector('solves/'+key+'_motion',torch.zeros_like(g))
            report=dict(vector=ref,beta=beta,lambda_value=1.,q=0.,gradient_dot=0.,parameter_cost=0.,prediction_cost=0.,
                relative_stationarity=None,absolute_stationarity=pg_norm,parameter_error_upper=pg_norm,
                objective_gap_upper=.5*pg_norm**2,functional_error_upper=pg_norm/np.sqrt(beta) if beta>0 else None,
                iterations=0,projected_gradient_numerically_unresolved=True,arithmetic_resolution=arithmetic,
                arithmetic_resolution_is_heuristic=True,
                regularization_penalty_is_not_real_CE=True,complement_is_full_parameter_domain=True,
                training_gradient_inside_constructed_span_can_make_this_trivial=True,operator_cost_cumulative=self.g.operator.audit())
            atomic_json(meta,report);return report
        rhs=self.j(pg).cpu().numpy();b_norm=float(np.linalg.norm(rhs));history=[]
        def matvec(y):return y+beta*self.j(P(self.jt(torch.tensor(y,device='cuda')))).cpu().numpy()
        A=LinearOperator((len(rhs),len(rhs)),matvec=matvec,dtype=float)
        state=None
        if ck.exists():
            with np.load(ck) as z:state={k:z[k].copy() for k in z.files}
        def checkpoint(x,s):
            temp=ck.with_suffix('.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,**s)
            temp.replace(ck)
            if self.stop:raise InterruptedError()
        while True:
            x,info,state=continuing_cg(A,rhs,rtol=self.a.rtol*.1,max_iterations=4,state=state,callback=checkpoint)
            if info<0:raise ArithmeticError('Nonpositive output curvature')
            v=-pg+beta*P(self.jt(torch.tensor(x,device='cuda')));Jv=self.j(v)
            residual=P(g+v+beta*self.jt(Jv));rn=float(residual.norm());gn=float(pg.norm())
            q=float(g@v)+.5*float(v@v)+.5*beta*float(Jv@Jv)
            history.append(dict(iterations=state['iterations'],fresh_parameter_stationarity=rn,
                relative_stationarity=rn/max(gn,np.finfo(float).tiny),objective=q,
                optimum_gap_upper=.5*rn*rn,output_recursive_relative=np.sqrt(float(state['rr']))/max(b_norm,np.finfo(float).tiny)))
            atomic_json(folder/(key+'_history.json'),history)
            self.status('FISHER_CONDITIONAL_SOLVE',case=key,iterations=int(state['iterations']),relative_stationarity=history[-1]['relative_stationarity'])
            if rn<=self.a.rtol*max(gn,np.finfo(float).tiny):break
            if info==0:
                # Recompute the true dual residual before another recurrence.
                rr=rhs-matvec(x);state=dict(x=x,r=rr,p=rr,rr=float(rr@rr),iterations=state['iterations'])
        ref=self.save_vector('solves/'+key+'_motion',v)
        np.savez_compressed(folder/(key+'_response.npz'),Jv=Jv.cpu().numpy())
        report=dict(vector=ref,beta=beta,lambda_value=1.,q=q,gradient_dot=float(g@v),parameter_cost=.5*float(v@v),
            prediction_cost=.5*beta*float(Jv@Jv),relative_stationarity=history[-1]['relative_stationarity'],
            absolute_stationarity=rn,parameter_error_upper=rn,objective_gap_upper=.5*rn*rn,
            functional_error_upper=rn/np.sqrt(beta) if beta>0 else None,iterations=int(state['iterations']),
            regularization_penalty_is_not_real_CE=True,complement_is_full_parameter_domain=True,
            operator_cost_cumulative=self.g.operator.audit())
        atomic_json(meta,report);return report
    def finite_sufficiency(self,anchor_basis,local_basis,fisher,rhos):
        for demand in UNSEEN:
            g=self.load(self.grad('DTD',2,demand));gn=float(g.norm())
            methods=[('full_gradient',None,None),('mean_gradient_1d',None,None)]
            families={'frozen_gradient':anchor_basis['basis'],'existing_fisher':fisher['basis']}
            if self.anchor!=self.a.anchors[0]:families['refreshed_gradient']=local_basis['basis']
            coefficients={}
            for family,refs in families.items():
                b=self.dot_basis(refs,g);coefficients[family]=b.tolist()
                for r in range(1,len(refs)+1):methods.append((f'{family}_rank{r}',refs[:r],b[:r]))
            atomic_json(self.out/f'projections/a{self.anchor}_{demand}.json',dict(full_gradient_norm=gn,coefficients=coefficients,
                rank_bound_is_demand_count=len(BUILD),mean_baseline_included=True))
            for name,refs,b in methods:
                if name=='full_gradient':v=-g
                elif name=='mean_gradient_1d':
                    mean=self.load(anchor_basis['mean_gradient']);mean/=mean.norm();v=-float(mean@g)*mean
                else:v=-self.combine(refs,b)
                n=float(v.norm())
                if not n:continue
                ref=self.vector_cached(f'controls/a{self.anchor}_{demand}_{name}',lambda v=v:v/n)
                for i,rho in enumerate(rhos):
                    signs=(1,-1) if name in ('full_gradient','mean_gradient_1d',f'frozen_gradient_rank{len(anchor_basis["basis"])}',f'existing_fisher_rank{len(fisher["basis"])}') else (1,)
                    for sign in signs:
                        self.evaluate(f'a{self.anchor}_{demand}_{name}_r{i}_s{sign}',ref,sign*rho,demand,
                            dict(family='sufficiency',method=name,radius_index=i,sign=sign,
                                first_order_DTD_selection=float(sign*rho*(g@(v/n))),
                                euclidean_proxy_at_executed_step=float(sign*rho*(g@(v/n)))+.5*rho*rho,
                                matched_parameter_radius=True,not_a_fixed_response_protection_curve=True))
    def fisher_baseline(self):
        meta=self.out/'bases/existing_fisher.json'
        if meta.exists():return json.loads(meta.read_text())
        src=self.a.flow_source;current=json.loads((src/'CURRENT.json').read_text())
        raw=(src/current['state']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=current['sha256']:raise ValueError('Old flow state hash')
        state=json.loads(raw);first=json.loads((src/state['transitions'][0]).read_text())
        original=torch.load(src/first['theta_k']['file'],map_location='cpu',mmap=True,weights_only=True)['vector']
        if not torch.equal(original,self.point.cpu()):raise ValueError('Fisher baseline anchor mismatch')
        count=len(json.loads((src/'baselines/a0.json').read_text())['learned']['parameter_gram']);refs=[]
        for i,r in enumerate(first['basis_k'][:count]):
            if digest(src/r['file'])!=r['sha256']:raise ValueError('Existing Fisher direction checksum')
            v=torch.load(src/r['file'],map_location='cpu',mmap=True,weights_only=True)['vector']
            refs.append(self.save_vector(f'bases/existing_fisher_{i}',v))
        report=dict(basis=refs,available_rank=count,origin=str(src/state['transitions'][0]),
            original_parameter_anchor_exact_match=True,rank_comparisons_only_where_available=True)
        atomic_json(meta,report);return report
    def euclidean_regions(self,supports,rhos):
        for demand in ('mean',UNSEEN[1]):
            g=self.load(self.grad('DTD',2,demand))
            for name,support in supports.items():
                P=self.projector(support);v=-P(g);n=float(v.norm())
                if not n:continue
                ref=self.vector_cached(f'controls/a{self.anchor}_{demand}_freeze_{name}',lambda:v/n)
                for i,rho in enumerate(rhos):
                    self.evaluate(f'a{self.anchor}_{demand}_freeze_{name}_r{i}',ref,rho,demand,
                        dict(family='euclidean_freeze',method=name,radius_index=i,
                            native_proxy_deletion=.5*(float(g@g)-n*n),euclidean_proxy_has_no_functional_substitution=True,
                            frozen_support=support,complement_dimension_unrestricted=True))
    def fisher_cases(self,basis,supports,rhos,kappa):
        jobs=[]
        for alpha in (.1,1.,10.):
            selected=[('full',None),('top_native_region',supports['top_native_region']),('signed_conflict_union',supports['signed_conflict_union']),('freeze_learned_motion',None)]
            selected.extend((k,supports[k]) for k in ('frozen_top_region_from_anchor0','frozen_conflict_union_from_anchor0') if k in supports)
            for name,support in selected:
                jobs.append(('mean',0,alpha,name,support))
        # A genuinely unseen demand, plus a matched physical-size control.
        for name in ('full','top_native_region','signed_conflict_union','freeze_learned_motion','matched_contiguous_control'):
            jobs.append((UNSEEN[1],2,1.,name,supports.get(name)))
        for demand,probe,alpha,name,support in jobs:
            g=self.load(self.grad('DTD',probe,demand));beta=alpha/kappa
            P=self.projector(support,basis['basis'] if name=='freeze_learned_motion' else None)
            key=f'a{self.anchor}_{demand}_p{probe}_alpha{alpha}_{name}'
            report=self.solve(key,g,beta,P);v=self.load(report['vector']);n=float(v.norm())
            for i,rho in enumerate(rhos):
                self.evaluate(key+f'_r{i}',report['vector'],rho/n if n else 0.,demand,
                    dict(family='fisher_freeze',method=name,radius_index=i,alpha=alpha,beta=beta,
                        solver_report=str(Path('solves')/(key+'.json')),direction_normalized_for_common_parameter_budget=True,
                        normalized_probe_is_not_constrained_quadratic_optimum=True,
                        zero_direction_not_normalized=not bool(n)))
                tau=rho/float(g.norm())
                self.evaluate(key+f'_native_r{i}',report['vector'],tau,demand,
                    dict(family='fisher_common_objective',method=name,radius_index=i,alpha=alpha,beta=beta,
                        physical_lambda=1/tau,physical_beta=beta/tau,physical_proxy_optimum=tau*report['q'],
                        physical_proxy_gap_upper=tau*report['objective_gap_upper'],radius_upper_bound=rho,
                        same_objective_and_budget_for_full_and_restricted=True,individual_direction_not_renormalized=True))
            if name!='full':
                fullkey=f'a{self.anchor}_{demand}_p{probe}_alpha{alpha}_full'
                full=json.loads((self.out/'solves'/f'{fullkey}.json').read_text());v0=self.load(full['vector'])
                naive=P(v0);Jn=self.j(naive);qn=float(g@naive)+.5*float(naive@naive)+.5*beta*float(Jn@Jn)
                e=v-v0;Je=self.j(e);schur=.5*float(e@e)+.5*beta*float(Je@Je)
                delta=report['q']-full['q'];gap0=full['objective_gap_upper'];gap1=report['objective_gap_upper']
                atomic_json(self.out/'deletions'/f'{key}.json',dict(full=full,restricted=report,
                    approximate_proxy_deletion=delta,certified_deletion_interval=[delta-gap1,delta+gap0],
                    schur_energy_from_solution_difference=schur,schur_identity_discrepancy=delta-schur,
                    naive_freeze_q=qn,compensation_proxy_gain=qn-report['q'],
                    compensation_parameter_norm=float((v-naive).norm()),full_norm=float(v0.norm()),restricted_norm=n,
                    physical_deletion_by_radius=[dict(radius=rho,physical_lambda=float(g.norm())/rho,
                        physical_beta=beta*float(g.norm())/rho,proxy_deletion=rho/float(g.norm())*delta,
                        certified_interval=[rho/float(g.norm())*(delta-gap1),rho/float(g.norm())*(delta+gap0)]) for rho in rhos],
                    finite_normalized_effects_are_separate=True,not_a_real_CE_loss=True))
                naive_ref=self.vector_cached('controls/'+key+'_naive',lambda:naive)
                nn=float(naive.norm())
                self.evaluate(key+'_naive_r1',naive_ref,rhos[1]/nn if nn else 0.,demand,
                    dict(family='naive_freeze',method=name,radius_index=1,alpha=alpha,
                        same_frozen_support=True,without_complement_reoptimization=True))
                self.evaluate(key+'_naive_native_r1',naive_ref,rhos[1]/float(g.norm()),demand,
                    dict(family='naive_freeze_common_objective',method=name,radius_index=1,alpha=alpha,
                        same_common_objective_as_reoptimized=True,without_complement_reoptimization=True))
                self.conditional_driver_curve(key,name,demand,g,beta,full,report,P,rhos[1])
    def conditional_driver_curve(self,key,name,demand,g,beta,full,restricted,P,rho):
        v0=self.load(full['vector']);vn=self.load(restricted['vector']);driver=v0-P(v0)
        amplitudes=[-1.,-.5,0.,.5,1.,1.5,2.]
        maximum=0.
        for t in amplitudes:
            maximum=max(maximum,float((t*v0+(1-t)*vn).norm()),float((t*driver+vn).norm()))
        tau=rho/max(maximum,np.finfo(float).tiny)
        for t in amplitudes:
            for compensated in (True,False):
                label='compensated' if compensated else 'fixed_complement'
                v=t*v0+(1-t)*vn if compensated else t*driver+vn
                ref=self.vector_cached(f'controls/{key}_driver{t}_{label}',lambda:v)
                residual_bound=abs(t)*full['absolute_stationarity']+abs(1-t)*restricted['absolute_stationarity']
                self.evaluate(f'{key}_driver{t}_{label}',ref,tau,demand,
                    dict(family='schur_conditional',method=name,radius_index=1,driver_amplitude=t,
                        compensated=compensated,full_solver_vector=full['vector'],restricted_solver_vector=restricted['vector'],
                        physical_lambda=1/tau,physical_beta=beta/tau,shared_radius_upper_bound=rho,
                        fixed_driver_identical_within_pair=True,full_complement_allowed=compensated,
                        transient_control_vector_archived_after_evaluation=True,
                        conditional_proxy_gap_upper=tau*.5*residual_bound**2 if compensated else None,
                        reconstruction='t * full + (1-t) * restricted' if compensated else 't * fixed_region(full) + restricted'))
                # The retained base solutions and affine recipe exactly specify the action.
                path=self.out/ref['file']
                if not path.resolve().is_relative_to(self.out.resolve()):raise ValueError('Unsafe temporary control path')
                if path.exists():path.unlink()
    def compatibility(self,basis,rhos):
        gD=self.load(self.grad('DTD',0,'mean'));gC=self.load(self.grad('Cars',0,'mean'))
        gd2=float(gD@gD);gc2=float(gC@gC);cross=float(gD@gC)
        atomic_json(self.out/f'compatibility/a{self.anchor}_gradient.json',dict(DTD_norm=np.sqrt(gd2),Cars_norm=np.sqrt(gc2),
            signed_inner=cross,cosine=cross/np.sqrt(gd2*gc2),Cars_slope_along_DTD_descent=-cross,
            negative_inner_means_first_order_conflict=cross<0))
        supports=json.loads((self.out/f'regions/a{self.anchor}_selected.json').read_text())['supports']
        domains={'full':self.projector(),**{f'freeze_{k}':self.projector(v) for k,v in supports.items()}}
        for domain,P in domains.items():
            pd=P(gD);pc=P(gC)
            K=np.array([[float(gD@pd),float(gD@pc)],[float(gC@pd),float(gC@pc)]])
            atomic_json(self.out/f'compatibility/a{self.anchor}_{domain}_linear_reachable.json',dict(gradient_gram=K,
                radius=rhos[1],squared_semiaxes=rhos[1]**2*np.maximum(np.linalg.eigvalsh(K),0),
                support_function='rho * sqrt(w^T gradient_gram w)',
                exact_first_order_budget_geometry=True,not_exact_finite_nonlinear_reachability=True))
            for weight in np.linspace(0,1,9):
                v=-P(weight*gD/np.sqrt(gd2)+(1-weight)*gC/np.sqrt(gc2));n=float(v.norm())
                if not n:continue
                ref=self.vector_cached(f'compatibility/a{self.anchor}_{domain}_w{weight:.3f}',lambda:v/n)
                self.evaluate(f'compat_a{self.anchor}_{domain}_w{weight:.3f}',ref,rhos[1],'mean',
                    dict(family='compatibility',method=domain,mixture_weight=float(weight),radius_index=1,
                        DTD_first_order=rhos[1]*float(gD@(v/n)),Cars_first_order=rhos[1]*float(gC@(v/n)),
                        attainable_set_is_sampled_not_full_nonlinear_Pareto=True))
    def archive_controls(self,step):
        """Only transient counterfactual control vectors; exact inputs/recipes remain."""
        records=[json.loads(p.read_text()) for p in (self.out/'evaluations').glob('*.json')]
        live={r['vector']['file']:r for r in records if r['anchor']==step}
        old_index=self.out/f'CONTROL_ARCHIVE_a{step}.json'
        archived=json.loads(old_index.read_text())['records'] if old_index.exists() else []
        for file,r in live.items():
            p=self.out/file
            if not p.resolve().is_relative_to(self.out.resolve()):raise ValueError('Unsafe archive path')
            if p.parts[-2] not in ('controls','compatibility','expert_baselines'):continue
            if p.exists():
                if digest(p)!=r['vector']['sha256']:raise ValueError('Control changed before archival')
                archived.append(dict(vector=r['vector'],case=r['case'],case_file='evaluations/'+r['case']+'.json',
                    storage='Reconstruct from preserved gradients/bases, frozen resources and exact source',
                    original_bytes=p.stat().st_size))
                p.unlink()
        atomic_json(self.out/f'CONTROL_ARCHIVE_a{step}.json',dict(records=archived,
            original_training_and_flow_data_untouched=True,construction_gradients_and_solver_vectors_retained=True,
            anchor0_basis_retained=True,later_bases_recipe_archived=True))
        if step!=self.a.anchors[0]:
            meta=json.loads((self.out/f'bases/a{step}.json').read_text())
            for ref in meta['basis']:
                path=self.out/ref['file']
                if not path.resolve().is_relative_to(self.out.resolve()):raise ValueError('Unsafe basis archive path')
                if path.exists():
                    if digest(path)!=ref['sha256']:raise ValueError('Basis changed before recipe archival')
                    path.unlink()
            atomic_json(self.out/f'BASIS_ARCHIVE_a{step}.json',dict(basis=meta['basis'],
                construction_gradients_retained=True,coefficients_and_both_projection_passes_retained=True,
                normalization_scalars_retained=True,reference_file=f'bases/a{step}.json'))
    def run(self):
        self.status('START');initial=None;fisher=self.fisher_baseline()
        first_meta=self.out/f'bases/a{self.a.anchors[0]}.json'
        if first_meta.exists():initial=json.loads(first_meta.read_text())
        for step in self.a.anchors:
            if (self.out/f'ANCHOR_COMPLETE_{step}.json').exists():continue
            self.set_anchor(step);self.status('ANCHOR_START');local=self.build_basis()
            if initial is None:initial=local
            g=self.load(self.grad('DTD',0,'mean'));gc=self.load(self.grad('Cars',0,'mean'))
            supports=self.supports(g,gc);Jg=self.j(g);kappa=float(Jg@Jg)/float(g@g)
            unit_speed=float(Jg.norm())/float(g.norm());rhos=[r/unit_speed for r in (.0025,.01,.04)]
            atomic_json(self.out/f'calibration_a{step}.json',dict(kappa=kappa,rhos=rhos,
                calibration_uses_construction_only=True,functional_radii_requested=[.0025,.01,.04],
                actual_other_direction_responses_not_equal=True,parameter_metric='I'))
            self.finite_sufficiency(initial,local,fisher,rhos)
            self.euclidean_regions(supports,rhos)
            if step in (self.a.anchors[0],25,100,216):self.fisher_cases(initial,supports,rhos,kappa)
            self.compatibility(initial,rhos)
            atomic_json(self.out/f'ANCHOR_COMPLETE_{step}.json',dict(anchor=step,completed=True))
            self.archive_controls(step)
        self.status('COMPLETE')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','inventory','probes','coordinates','trace','flow-source'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--anchors',nargs='+',type=int,default=[0,25]);p.add_argument('--rtol',type=float,default=.001)
    p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args()
    try:Experiment(a).run()
    except InterruptedError:atomic_json(a.output/'PAUSED.json',dict(time=time.time(),resumable=True))
    except Exception:
        import traceback
        atomic_json(a.output/'FAILURE.json',dict(traceback=traceback.format_exc()));raise
