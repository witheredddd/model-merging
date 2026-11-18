"""DTD-only, matrix-free horizontal tracking and real functional-flow samples."""
import argparse,json,os,signal,time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator
from bootstrap import CORE,LEGACY,R3,SPACE
from carrier import TrainingPositions
from io_state import Transactions,atomic_json,digest
from geometry import FullFunctionalGeometry
from iterative import continuing_cg
from numerics import sphere_log,functional_innovation,principal_geometry,align_rotation


class Flow:
    def __init__(self,a):
        self.a=a;self.out=a.output.resolve();self.out.mkdir(parents=True,exist_ok=True)
        import fcntl
        self.lock=(self.out/'WRITER.lock').open('a+b');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.stop=False
        signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True))
        signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))
        self.store=Transactions(self.out);self.codes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
        names=json.loads(a.coordinates.read_text())['parameter_keys'];self.positions=TrainingPositions(a.trace,names,self.out)
        if (self.out/'CURRENT.json').exists():
            self.s=self.store.load()
            if self.s['codes']!=self.codes:raise ValueError('Explicit code migration required')
            for p,h in self.s['inputs'].items():
                if digest(p)!=h:raise ValueError('Frozen input changed: '+p)
        else:
            runtime=self.out/'runtime';runtime.mkdir(exist_ok=True)
            import shutil
            shutil.copy2(a.probes,runtime/'probes.json')
            self.s=dict(schema='DTD-self-fiber-flow-1',generation=0,codes=self.codes,
                inputs={str(p):digest(p) for p in (a.probes,a.inventory,a.coordinates,a.trace/'trace.json')},
                original_training_proof=self.positions.verify_normal_recipe(),
                anchor_cursor=0,phase='DISCOVER',basis=[],frames={},paths=[],transitions=[],solvers={},
                controls=[],path_cursor=0,path_step=0,stage='RUNNING',
                next_candidate=0,task='DTD',parameter_metric='I',other_task_protection=False,
                CE_accuracy_acceptance=False,candidate_block_is_not_intrinsic_dimension=a.candidate_block,
                path_steps_are_sampling_budget=a.path_steps,manifold_claim=False)
            self.save()
        if self.s['phase']=='PILOT_COMPLETE':return
        self.load_context();self.s['stage']='RUNNING';self.save()

    def save(self,cg=None,key=None):
        self.store.save(self.s,cg,key)
        atomic_json(self.out/'status.json',dict(phase=self.s['phase'],stage=self.s['stage'],
            anchor_step=self.a.anchors[self.s['anchor_cursor']] if self.s['anchor_cursor']<len(self.a.anchors) else None,
            basis_size=len(self.s['basis']),path_cursor=self.s['path_cursor'],path_step=self.s['path_step'],
            physical_transitions=len(self.s['transitions']),last=self.s.get('last'),
            other_task_protection=False,CE_accuracy_acceptance=False,manifold_claim=False))

    def load_context(self):
        import task_runtime
        inventory=json.loads(self.a.inventory.read_text())['tasks'];inventory=[r for r in inventory if r['task']=='DTD']
        if len(inventory)!=1:raise ValueError('DTD inventory missing')
        # Constructor's task list is local to this isolated process; it loads only
        # DTD images/head, never invokes another task's Fisher or protection.
        old=task_runtime.TASKS;task_runtime.TASKS=('DTD',)
        try:c=task_runtime.EightTaskContext(self.out/'runtime',inventory)
        finally:task_runtime.TASKS=old
        index=self.positions_index(self.a.anchors[self.s['anchor_cursor']])
        anchor=self.positions.prepare(index)
        point=torch.load(self.out/anchor['file'],map_location='cpu',mmap=True,weights_only=True)['full_visual_flat'].cuda().double()
        self.g=FullFunctionalGeometry(c,point,0,precision='float64')
        self.classes=self.g.output_dimensions['DTD']//sum(len(y) for _,y in self.g.c.batches['DTD',0])
        # Split the disjoint validation probe into two repeatability observations.
        for probe in (2,3,4):self.g.c.batches['DTD',probe]=self.g.c.batches['DTD',1][(probe-2)::3]
        if self.g.tasks!=('DTD',):raise ValueError('This experiment must be DTD only')
        self.s.setdefault('operator_sources',{str(p):digest(p) for folder in (CORE,LEGACY) for p in folder.glob('*.py')})
        for p,h in self.s['operator_sources'].items():
            if digest(p)!=h:raise ValueError('Operator source changed')
        if self.s.get('current_point'):self.set_point(self.load(self.s['current_point']))
        else:self.s['current_point']=self.vector('anchors/initial_'+str(index),point)

    def positions_index(self,step):return next(i for i,r in enumerate(self.positions.rows) if r['step']==step)
    def set_point(self,p):
        self.g.point=p.detach().to(self.g.device).double();self.g.operator.point=self.g.point;self.g.c.anchor=self.g.point
    def load(self,ref,device=None):
        p=self.out/ref['file']
        if digest(p)!=ref['sha256']:raise ValueError('Physical vector changed: '+str(p))
        return torch.load(p,map_location=device or self.g.device,mmap=True,weights_only=True)['vector'].double()
    def vector(self,name,v):
        if name.startswith(('basis/','aligned/','discovery/','tracking/')):name='numeric_scratch/'+name
        p=self.out/(name+'.pt');p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
        torch.save(dict(vector=v.detach().cpu()),tmp);h=digest(tmp);p=p.with_name(p.stem+'_'+h[:16]+p.suffix)
        if p.exists():tmp.unlink()
        else:tmp.replace(p)
        return dict(file=str(p.relative_to(self.out)),sha256=h,dtype=str(v.dtype),dimension=v.numel())

    def cleanup_numeric_scratch(self):
        live=set()
        def visit(obj):
            if isinstance(obj,dict):
                if 'file' in obj and 'sha256' in obj:live.add(obj['file'])
                for value in obj.values():visit(value)
            elif isinstance(obj,list):
                for value in obj:visit(value)
        visit(self.s)
        for p in (self.out/'flows').glob('*.json'):
            visit(json.loads(p.read_text()))
        scratch=(self.out/'numeric_scratch').resolve()
        if not scratch.exists():return
        for p in scratch.rglob('*.pt'):
            if str(p.relative_to(self.out)) not in live:
                assert p.resolve().is_relative_to(scratch)
                p.unlink() # Only unreferenced numerical scratch, never accepted flow states/bases.
    def j(self,v,probe=0):
        self.g.probe=probe
        self.g.output_dimensions['DTD']=sum(len(y) for _,y in self.g.c.batches['DTD',probe])*self.classes
        return self.g.tensor_to_numpy(self.g.jvp('DTD',v))
    def jt(self,y):
        self.g.probe=0;self.g.output_dimensions['DTD']=len(y)
        return self.g.vjp('DTD',self.g.numpy_to_tensor(y))
    def fisher(self,v):return self.jt(self.j(v,0))
    def phi(self,p,probe):
        logits,labels=self.g.c.baseline('DTD',probe,p);prob=logits.softmax(-1)
        return (2*prob.sqrt()/np.sqrt(len(labels))).numpy(),dict(CE=float(torch.nn.functional.cross_entropy(logits,labels)),Top1=float((logits.argmax(-1)==labels).double().mean()))

    def orthogonal(self,v,refs):
        for _ in range(2):
            for r in refs:
                b=self.load(r);v-=torch.dot(b,v)*b
        return v

    def project_horizontal(self,b,key):
        y=self.j(b);A=LinearOperator((len(y),len(y)),matvec=lambda x:self.j(self.jt(x)),dtype=float)
        cg=self.store.cg(self.s,key)
        while not self.stop:
            _,info,cg=continuing_cg(A,y,rtol=self.a.numerical_rtol,max_iterations=1,state=cg)
            self.save(cg,key)
            if info<0:raise ArithmeticError('Horizontal projection breakdown')
            if info==0:break
        if self.stop:raise InterruptedError()
        v=self.jt(cg['x']);fresh=y-self.j(v)
        return v,dict(method='Undamped output-dual minimum-norm horizontal lift',
            iterations=int(cg['iterations']),fresh_response_residual=float(np.linalg.norm(fresh)),
            vertical_parameter_norm=float((b-v).norm()),
            vertical_horizontal_cross=float(torch.dot(v,b-v)),parameter_error_bound=None,
            unknown_smallest_positive_singular_value=True)

    def discover(self):
        # Each candidate is in range(J^T), generated by task Fisher, not a stored
        # teacher update. Numerical refinements replace the same physical basis.
        if not self.s['basis']:
            refs=[]
            for i in range(self.a.candidate_block):
                gen=torch.Generator(device=self.g.device);gen.manual_seed(self.a.seed+i)
                q=torch.randn(self.g.dimension,device=self.g.device,dtype=torch.float64,generator=gen)
                v=self.orthogonal(self.fisher(q),refs);n=float(v.norm())
                if n:refs.append(self.vector(f'discovery/a{self.s["anchor_cursor"]}_initial{i}',v/n))
            self.s['basis']=refs;self.s['discovery_sweep']=0;self.save()
        old=self.s['basis'];new=[]
        for i,r in enumerate(old):
            v=self.orthogonal(self.fisher(self.load(r)),new);n=float(v.norm())
            if n:new.append(self.vector(f'discovery/a{self.s["anchor_cursor"]}_s{self.s.get("discovery_sweep",0)}_{i}',v/n))
        new,report=self.ritz(new,'discovery')
        self.s['basis']=new;self.s['discovery_sweep']=self.s.get('discovery_sweep',0)+1
        self.s['last']=report;self.save()
        self.cleanup_numeric_scratch()
        if report['max_relative_eigen_residual']<=self.a.numerical_rtol:
            validation=self.validate(new);self.s['anchor_basis']=new;self.s['anchor_point']=self.s['current_point']
            self.s['anchor_validation']=validation
            self.baselines()
            self.s['controls']=self.controls(len(new));self.s.update(phase='PATH',path_cursor=0,path_step=0);self.save()

    def ritz(self,refs,label,align_to=None):
        # Output-side Rayleigh matrix; never hold P-by-d on the GPU.
        responses=np.column_stack([self.j(self.load(r)) for r in refs]);H=responses.T@responses
        values,U=np.linalg.eigh((H+H.T)/2);order=np.argsort(values)[::-1]
        values=values[order];U=U[:,order];output=[];residuals=[]
        cutoff=np.finfo(float).eps*len(values)*max(1.,np.linalg.norm(H,2))
        for j in range(len(refs)):
            if values[j]<=cutoff:
                self.s.setdefault('inactive_observations',[]).append(dict(source=refs,ritz_value=float(values[j]),reason='Machine-unresolved nonpositive Fisher mode; observations retained'))
                continue
            v=sum((float(U[i,j])*self.load(refs[i]) for i in range(len(refs))),torch.zeros_like(self.g.point))
            Fv=self.fisher(v)
            residuals.append(float((Fv-values[j]*v).norm())/max(float(Fv.norm()),np.finfo(float).tiny))
            output.append(self.vector(f'basis/{label}_a{self.s["anchor_cursor"]}_p{self.s["path_cursor"]}_k{self.s["path_step"]}_s{self.s.get("discovery_sweep",0)}_{j}',v))
        alignment=None
        if align_to and output:
            cross=np.array([[float(self.load(a,'cpu')@self.load(b,'cpu')) for b in output] for a in align_to])
            alignment=principal_geometry(cross,len(align_to),len(output))
        if align_to and len(align_to)==len(output):
            rotation=align_rotation(cross)
            original=list(output);aligned=[]
            for j in range(len(output)):
                v=sum((float(rotation[i,j])*self.load(original[i]) for i in range(len(output))),torch.zeros_like(self.g.point))
                aligned.append(self.vector(f'aligned/{label}_a{self.s["anchor_cursor"]}_p{self.s["path_cursor"]}_k{self.s["path_step"]}_{j}',v))
            output=aligned
        report=dict(ritz_values=values.tolist(),relative_eigen_residuals=residuals,
            max_relative_eigen_residual=max(residuals) if residuals else 0.,alignment=alignment,
            aligned_columns_are_not_individual_eigenvectors=bool(alignment))
        with (self.out/'NUMERICAL_HISTORY.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        return output,report

    def validate(self,refs):
        Ys={probe:np.column_stack([self.j(self.load(r),probe) for r in refs]) for probe in (0,1,2,3,4)}
        report={str(p):dict(response_gram=Y.T@Y,rank=int(np.linalg.matrix_rank(Y)),
            norms=np.linalg.norm(Y,axis=0)) for p,Y in Ys.items()}
        report['parameter_gram']=np.array([[float(self.load(a,'cpu')@self.load(b,'cpu')) for b in refs] for a in refs])
        report['probe_roles']={'0':'Numerical discovery and step calibration','1':'Entire original validation pool; partially used for birth',
            '2':'Independent birth-repeatability batch A','3':'Independent birth-repeatability batch B','4':'Untouched by birth and acceptance; evaluation only'}
        self.g.probe=0
        atomic_json(self.out/f'validation/a{self.s["anchor_cursor"]}_p{self.s["path_cursor"]}_k{self.s["path_step"]}.json',report)
        return report

    def controls(self,d):
        if d<2:raise ArithmeticError('Fewer than two discovered directions; no two-direction flow pilot claim')
        controls=[]
        for i in range(2):
            for sign in (1,-1):
                a=np.zeros(d);a[i]=sign;controls.append(dict(id=f'axis{i}_{sign}',a=a.tolist(),kind='short_path'))
        for sign in (1,-1):
            a=np.zeros(d);a[0]=1/np.sqrt(2);a[1]=sign/np.sqrt(2);controls.append(dict(id=f'mixed_{sign}',a=a.tolist(),kind='short_path'))
        for seq in ([0,1,2,3],[1,0,3,2]):
            for radius_factor in (1.,.5,.25):
                actions=[]
                for i in seq:
                    a=np.zeros(d);a[i%2]=1 if i<2 else -1;actions.append(a.tolist())
                controls.append(dict(id='loop_'+''.join(map(str,seq))+f'_scale{radius_factor}',
                    basis_family='closed_loop_common',radius_factor=radius_factor,
                    sequence=actions,kind='matched_step_loop'))
        frozen=[dict(c,id=c['id']+'_frozen',tracking='frozen') for c in controls if c['kind']=='short_path']
        return [dict(c,tracking='online') for c in controls]+frozen

    def path(self):
        if self.s.get('pending_transition'):
            self.finish_transition(self.s['pending_transition']);return
        if self.s['path_cursor']>=len(self.s['controls']):self.next_anchor();return
        control=self.s['controls'][self.s['path_cursor']]
        if self.s['path_step']==0:
            self.set_point(self.load(self.s['anchor_point']));self.s['current_point']=self.s['anchor_point']
            starts=self.s.setdefault('control_start_bases',{})
            canonical=control.get('basis_family',control['id'].removesuffix('_frozen'))
            if control['tracking']=='online' and canonical not in starts:
                self.consider_innovation()
                starts[canonical]=list(self.s['anchor_basis'])
            self.s['basis']=list(starts.get(canonical,self.s['anchor_basis']))
            if self.s.get('active_path_id')!=control['id']:self.s.pop('loop_h',None)
            self.s['active_path_id']=control['id'];self.save()
        refs=self.s['basis'];point=self.g.point.clone()
        action=np.asarray(control['a'] if 'a' in control else control['sequence'][self.s['path_step']])
        if len(action)>len(refs):action=action[:len(refs)]
        action=np.pad(action,(0,len(refs)-len(action)))
        v=sum((float(c)*self.load(r) for c,r in zip(action,refs)),torch.zeros_like(point))
        pred=self.j(v,0);speed=float(np.linalg.norm(pred))
        if not speed:raise ArithmeticError('Zero functional speed, no fabricated motion')
        if control['kind']=='matched_step_loop':
            if not self.s.get('loop_common_h'):
                speeds=[np.linalg.norm(self.j(self.load(r))) for r in refs[:2]]
                self.s['loop_common_h']=self.a.functional_radius/max(speeds)
            initial_h=self.s['loop_common_h']*control['radius_factor']
        else:initial_h=self.a.functional_radius/speed
        h=self.s.get('loop_h',initial_h) if control['kind']=='matched_step_loop' else initial_h
        base,metrics=self.phi(point,0);repeat,_=self.phi(point,0)
        numerical_floor=float(np.linalg.norm(sphere_log(base,repeat)))
        independent_base,_=self.phi(point,4);independent_prediction=self.j(v,4);trials=[]
        while not self.stop:
            newpoint=point+h*v;newphi,newmetrics=self.phi(newpoint,0)
            actual=sphere_log(base,newphi).reshape(-1);error=float(np.linalg.norm(actual-h*pred)/max(np.linalg.norm(h*pred),np.finfo(float).tiny))
            halfphi,_=self.phi(point+.5*h*v,0)
            halfactual=sphere_log(base,halfphi).reshape(-1)
            halferror=float(np.linalg.norm(halfactual-.5*h*pred)/max(np.linalg.norm(.5*h*pred),np.finfo(float).tiny))
            trials.append(dict(step_size=h,relative_Log_JVP_error=error,half_step_error=halferror,
                numerical_observation_floor=numerical_floor,
                actual_functional_arclength=float(np.linalg.norm(actual)),predicted_functional_arclength=h*speed))
            if error<=self.a.local_error and halferror<=self.a.local_error:break
            h*=.5
            if not torch.any(point+h*v!=point):raise ArithmeticError('Step reached floating-point floor')
        if self.stop:raise InterruptedError()
        if control['kind']=='matched_step_loop':
            if 'loop_h' in self.s and h!=self.s['loop_h']:
                self.s['loop_h']=h;self.s['path_step']=0;self.s['current_point']=self.s['anchor_point'];self.set_point(self.load(self.s['anchor_point']));self.save();return
            self.s['loop_h']=h
        prefix=f'flows/a{self.s["anchor_cursor"]}_{control["id"]}_step{self.s["path_step"]}_{self.s["current_point"]["sha256"][:12]}_h{h:.12g}'
        newref=self.vector(prefix+'_theta',newpoint);delta=self.vector(prefix+'_delta',h*v)
        before_ref=self.s['current_point']
        arrays=self.out/(prefix+'_functional.npz');arrays.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(arrays,JVP=pred,actual_Log=actual,base_phi=base,next_phi=newphi)
        independent_next,_=self.phi(newpoint,4)
        independent_actual=sphere_log(independent_base,independent_next).reshape(-1)
        independent_error=float(np.linalg.norm(independent_actual-h*independent_prediction)/max(np.linalg.norm(h*independent_prediction),np.finfo(float).tiny))
        np.savez_compressed(self.out/(prefix+'_independent_functional.npz'),JVP=independent_prediction,actual_Log=independent_actual,
            base_phi=independent_base,next_phi=independent_next)
        pending=dict(prefix=prefix,theta_k=before_ref,theta_next=newref,delta_theta=delta,basis_k=refs,
            independent_functional_arrays=prefix+'_independent_functional.npz',independent_Log_JVP_error=independent_error,
            control=action.tolist(),accepted_step_size=h,trials=trials,metric_observers_before=metrics,
            metric_observers_after=newmetrics,functional_arrays=str(arrays.relative_to(self.out)))
        self.s['pending_transition']=pending;self.s['current_point']=newref;self.save()
        self.finish_transition(pending)

    def finish_transition(self,pending):
        control=self.s['controls'][self.s['path_cursor']];prefix=pending['prefix'];refs=pending['basis_k']
        newref=pending['theta_next'];newpoint=self.load(newref);self.set_point(newpoint)
        h=pending['accepted_step_size'];error=pending['trials'][-1]['relative_Log_JVP_error']
        relocated=[];certificates=[]
        if control['tracking']=='frozen':updated=refs;tracking=dict(baseline='Frozen physical anchor basis; no online geometry update')
        else:
            if pending.get('tracking_refs'):relocated=pending['tracking_refs'];certificates=pending['horizontal_certificates']
            else:
                for i,r in enumerate(refs):
                    q,cert=self.project_horizontal(self.load(r),prefix+f'_horizontal{i}')
                    q=self.orthogonal(q,relocated);n=float(q.norm())
                    cert['vertical_recipe']=dict(old_direction=r,theta=self.s['current_point'],
                        dual_solver=self.s['solvers'][prefix+f'_horizontal{i}'],
                        definition='w = old_direction - J_D(theta)^T dual_x; before parameter orthogonalization')
                    if n:relocated.append(self.vector('tracking/'+prefix.replace('/','_')+f'_horizontal_basis{i}',q/n));certificates.append(cert)
                pending.update(tracking_refs=relocated,horizontal_certificates=certificates,tracking_sweep=0);self.save()
            while not self.stop:
                updated,tracking=self.ritz(relocated,prefix.replace('/','_'),align_to=refs)
                if tracking['max_relative_eigen_residual']<=self.a.numerical_rtol:break
                transformed=[]
                for i,r in enumerate(updated):
                    q=self.orthogonal(self.fisher(self.load(r)),transformed);n=float(q.norm())
                    if n:transformed.append(self.vector(f'tracking/{prefix.replace("/","_")}_{i}',q/n))
                relocated=transformed;pending['tracking_refs']=relocated;pending['tracking_sweep']+=1
                self.save();self.cleanup_numeric_scratch()
            if self.stop:raise InterruptedError()
        # Existing directions are updated, not appended as numerical refinements.
        if pending.get('validated_next_basis'):
            updated=pending['validated_next_basis'];tracking=pending['validated_tracking']
        else:
            if control['tracking']=='online':
                self.consider_innovation(updated,allow_birth=control['kind']!='matched_step_loop')
            pending['validated_next_basis']=updated;pending['validated_tracking']=tracking;self.save()
        validation=self.validate(updated)
        cross=np.array([[float(self.load(a,'cpu')@self.load(b,'cpu')) for b in updated] for a in refs])
        tracking['actual_next_space_geometry']=principal_geometry(cross,len(refs),len(updated))
        vertical_effect=self.observe_vertical_effect(certificates,updated,prefix,h) if certificates else None
        record=dict(anchor_id=self.a.anchors[self.s['anchor_cursor']],path_id=control['id'],step=self.s['path_step'],
            **{k:v for k,v in pending.items() if k not in ('tracking_refs','horizontal_certificates')},basis_next=updated,tracking=tracking,
            horizontal_relocation_certificates=certificates,validation=validation,
            vertical_finite_effect=vertical_effect,
            CE_accuracy_acceptance=False,actual_parameter_writeback=True,original_inputs_untouched=True)
        atomic_json(self.out/(prefix+'.json'),record)
        self.s['transitions'].append(str(Path(prefix+'.json')));self.s.update(current_point=newref,basis=updated)
        self.s['path_step']+=1;self.s['last']=dict(control=control['id'],step=self.s['path_step'],functional_error=error)
        limit=len(control.get('sequence',[])) or self.a.path_steps
        if self.s['path_step']>=limit:
            if control['kind']=='matched_step_loop':
                closure=newpoint-self.load(self.s['anchor_point']);projected=self.orthogonal(closure.clone(),self.s['anchor_basis'])
                atomic_json(self.out/(prefix+'_closure.json'),dict(closure_norm=float(closure.norm()),
                    outside_anchor_space_norm=float(projected.norm()),step_size=h,
                    normalized_area_residual=float(projected.norm())/(h*h),
                    finite_loop_is_not_a_Frobenius_proof=True))
            self.s['paths'].append(dict(anchor=self.a.anchors[self.s['anchor_cursor']],control=control['id'],accepted_steps=self.s['path_step']))
            self.s['path_cursor']+=1;self.s['path_step']=0
        self.s.pop('pending_transition',None)
        record['operator_cost_cumulative']=self.g.operator.audit()
        atomic_json(self.out/(prefix+'.json'),record)
        self.save();self.cleanup_numeric_scratch()

    def observe_vertical_effect(self,certificates,updated,prefix,h):
        index=len(self.s['transitions'])%len(certificates);certificate=certificates[index]
        recipe=certificate['vertical_recipe'];old=self.load(recipe['old_direction'])
        p=self.out/recipe['dual_solver']['file']
        if digest(p)!=recipe['dual_solver']['sha256']:raise ValueError('Vertical recipe solver changed')
        with np.load(p) as z:x=z['x'].copy()
        vertical=old-self.jt(x);norm=float(vertical.norm())
        if not norm:return dict(index=index,norm=0.,exact_zero=True,recipe=recipe)
        arithmetic=np.finfo(float).eps*(np.log2(self.g.dimension)+16)*(float(old.norm())+float((old-vertical).norm()))
        if norm<=arithmetic:return dict(index=index,vertical_norm=norm,recipe=recipe,
            arithmetic_resolution=arithmetic,finite_probe_not_applied=True,reason='Numerically unresolved vertical residual retained without normalization')
        check=old-self.jt(x);reconstruction=float((vertical-check).norm())/norm
        point=self.g.point.clone();base,_=self.phi(point,0);before=self.j(self.load(updated[0]))
        try:
            self.set_point(point+h*vertical/norm);after_phi,_=self.phi(self.g.point,0)
            after=self.j(self.load(updated[0]));finite=sphere_log(base,after_phi).reshape(-1)
        finally:self.set_point(point)
        np.savez_compressed(self.out/(prefix+'_vertical_effect.npz'),actual_Log=finite,
            before_direction_response=before,after_direction_response=after)
        return dict(index=index,vertical_norm=norm,step_size=h,
            step_policy='Inherit accepted horizontal parameter step for a separate diagnostic',
            recipe=recipe,recipe_reconstruction_relative_error=reconstruction,
            functional_change_norm=float(np.linalg.norm(finite)),
            subsequent_direction_response_change=float(np.linalg.norm(after-before)),
            does_not_change_accepted_path=True,is_not_a_new_sensitive_mode=True,
            arrays=prefix+'_vertical_effect.npz')

    def next_anchor(self):
        self.s['anchor_cursor']+=1
        if self.s['anchor_cursor']>=len(self.a.anchors):self.s.update(phase='PILOT_COMPLETE',stage='COMPLETE');self.save();return
        index=self.positions_index(self.a.anchors[self.s['anchor_cursor']]);ref=self.positions.prepare(index)
        p=torch.load(self.out/ref['file'],map_location='cpu',mmap=True,weights_only=True)['full_visual_flat'].cuda().double()
        self.set_point(p);self.s.pop('loop_common_h',None);self.s.update(phase='DISCOVER',basis=[],path_cursor=0,path_step=0,control_start_bases={},current_point=self.vector(f'anchors/initial_{index}',p));self.save()

    def consider_innovation(self,space=None,allow_birth=True):
        """A separate Fisher candidate, never an eigensolver/curvature residual."""
        serial=self.s['next_candidate'];self.s['next_candidate']+=1
        gen=torch.Generator(device=self.g.device);gen.manual_seed(self.a.seed+1000+serial)
        random=torch.randn(self.g.dimension,device=self.g.device,dtype=torch.float64,generator=gen)
        space=self.s['anchor_basis'] if space is None else space
        original=self.fisher(random);candidate=self.orthogonal(original.clone(),space);n=float(candidate.norm())
        metadata=dict(source='Separate task Fisher Krylov probe',numerical_refinement_is_not_birth=True,
            original_parameter_norm=float(original.norm()),parameter_innovation_norm=n)
        if n:
            candidate/=n;ref=self.vector(f'innovations/a{self.s["anchor_cursor"]}_proposal{serial}',candidate)
            observations={}
            for probe in (0,2,3):
                existing=np.column_stack([self.j(self.load(r),probe) for r in space])
                observations[str(probe)]=functional_innovation(existing,self.j(candidate,probe),self.classes)
            supported=all(o['independent'] for o in observations.values());admitted=supported and allow_birth
            metadata.update(vector=ref,functional_observations=observations,admitted=admitted,
                is_not_intrinsic_dimension_estimate=True,finite_vertical_effect_not_yet_assessed=True,
                constant_rank_loop_test=not allow_birth,constant_rank_assumption_questioned=supported and not allow_birth,
                theta=self.s['current_point'])
            if admitted:space.append(ref)
        else:metadata.update(admitted=False,reason='Exact parameter dependence')
        self.g.probe=0;atomic_json(self.out/f'innovations/a{self.s["anchor_cursor"]}_proposal{serial}.json',metadata);self.save()

    def baselines(self):
        rows=[]
        for i in range(self.a.candidate_block):
            gen=torch.Generator(device=self.g.device);gen.manual_seed(self.a.seed+i)
            q=torch.randn(self.g.dimension,device=self.g.device,dtype=torch.float64,generator=gen);q/=q.norm()
            rows.append(dict(kind='isotropic_parameter_baseline',seed=self.a.seed+i,
                sensitivities={str(p):float(np.linalg.norm(self.j(q,p))**2) for p in (0,4)}))
        if self.a.wls_source:
            old=Transactions(self.a.wls_source).load()
            for r in old.get('adaptive',{}).get('frozen_bank',[]):
                array=np.load(self.a.wls_source/r['motion_file'],mmap_mode='r')
                q=self.g.numpy_to_tensor(np.array(array));q/=q.norm()
                rows.append(dict(kind='prior_WLS_coordinate_DTD_only_evaluation',id=r['id'],
                    sensitivities={str(p):float(np.linalg.norm(self.j(q,p))**2) for p in (0,4)}))
        atomic_json(self.out/f'baselines/a{self.s["anchor_cursor"]}.json',dict(rows=rows,
            learned=self.s.get('anchor_validation'),equal_path_step_budget=self.a.path_steps,
            learning_cost_not_assumed_equal=True));self.g.probe=0

    def run(self):
        try:
            while not self.stop and self.s['phase']!='PILOT_COMPLETE':
                if self.s['phase']=='DISCOVER':self.discover()
                else:self.path()
            if self.stop:raise InterruptedError()
        except InterruptedError:self.s['stage']='PAUSED';self.save()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('output','trace','coordinates','inventory','probes'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--anchors',type=int,nargs='+',default=[0,50]);p.add_argument('--candidate-block',type=int,default=4)
    p.add_argument('--path-steps',type=int,default=4);p.add_argument('--seed',type=int,default=43)
    p.add_argument('--numerical-rtol',type=float,default=.001);p.add_argument('--functional-radius',type=float,default=.01)
    p.add_argument('--local-error',type=float,default=.05)
    p.add_argument('--wls-source',type=Path);args=p.parse_args()
    try:Flow(args).run()
    except Exception:
        import traceback
        atomic_json(args.output/'FAILURE.json',dict(traceback=traceback.format_exc()));raise
