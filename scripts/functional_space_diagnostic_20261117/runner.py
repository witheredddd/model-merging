"""Checkpointed numerical refinement, shared full-motion space, paired positions.

No CE optimization, trajectory writeback, fixed iteration limit, or mode-count cap.
Old single-tensor protected lifts remain a baseline, not the discovery support.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import signal
import time
import numpy as np
import torch
from bootstrap import R3,SPACE
from carrier import TrainingPositions
from space_worker import ObservedOutputWLS
from task_runtime import load_eight_geometry
from operators import OutputDualWLS,vnorm
from control_math import independent_column,quadratic_solve
from io_state import Transactions,atomic_json,digest
from diagnostics import error_budget,measurement,compare,projection,condition_grid


class Runner:
    def __init__(self,args):
        self.args=args;self.out=args.output;self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'WRITER.lock').open('a+b')
        import fcntl
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.stop=False
        signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stop',True))
        signal.signal(signal.SIGINT,lambda *_:setattr(self,'stop',True))
        self.store=Transactions(self.out);self.old=Transactions(args.old).load()
        self.protocol=json.loads((args.source/'PROTOCOL.json').read_text())
        self.codes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
        names=json.loads(args.coordinates.read_text())['parameter_keys']
        self.positions=TrainingPositions(args.trace,names,self.out)
        self.scales=json.loads((args.source/'SCALES.json').read_text())['scales']
        with np.load(args.source/'PRECONDITIONER.npz') as z:self.diagonals={t:z[t].copy() for t in z.files}
        if (self.out/'CURRENT.json').exists():
            self.state=self.store.load()
            if self.state['codes']!=self.codes:raise ValueError('Explicit source migration required')
            if self.state['relative_error_budget']!=args.relative_error_budget:raise ValueError('Budget changed')
            if self.state['gamma']!=args.gamma:raise ValueError('WLS weights changed')
            self.verify_sources()
        else:self.initialize()
        self.g=None;self.loaded_index=None;self.change_position(self.state['position'])
        if not self.state.get('functional_demand_catalog_registered'):
            classes={t:sorted(set(int(label) for _,labels in self.g.c.batches[t,0]
                for label in labels.detach().cpu().tolist())) for t in self.g.tasks}
            self.state['conditions']=condition_grid(self.g.tasks,classes)
            self.state['functional_demand_catalog_registered']=True
            self.state['functional_demand_catalog_size']=len(self.state['conditions'])
        self.bank_cache={};self.demand_cache={}
        self.state['run_state']='RUNNING';self.save()

    def initialize(self):
        if self.old['run_state']!='PAUSED':raise ValueError('Old observation must be paused')
        (self.out/'runtime').mkdir(exist_ok=True)
        shutil.copy2(self.args.old/'runtime/probes.json',self.out/'runtime/probes.json')
        # Immutable audit inventory; originals and old candidates are never overwritten.
        proof=self.positions.verify_normal_recipe()
        for i in range(len(self.positions.rows)):self.positions.raw(i)
        source_files=[self.args.source/'PROTOCOL.json',self.args.source/'SCALES.json',
            self.args.source/'PRECONDITIONER.npz',self.args.trace/'trace.json',self.args.coordinates,
            self.out/'runtime/probes.json',Path(self.protocol['inventory'])]
        code_dependencies=[p for folder in (R3,SPACE) for p in folder.glob('*.py')]
        self.state=dict(schema='space-diagnostic-1',codes=self.codes,generation=0,solvers={},
            run_state='RUNNING',stage='PILOT_REFERENCE',position=0,turn=0,bank=[],observations=[],
            source_hashes={str(p):digest(p) for p in source_files+code_dependencies},
            relative_error_budget=self.args.relative_error_budget,gamma=self.args.gamma,checkpoint_quantum=self.args.quantum,
            quantum_is_not_solve_limit=True,ordinary_training_proof=proof,
            original_positions=[dict(row) for row in self.positions.rows],
            imported_old_state=dict(file=str(self.args.old/'CURRENT.json'),snapshot=self.old),
            conditions=condition_grid(self.protocol['tasks']),condition_index=0,
            new_training_updates=0,CE_or_accuracy_acceptance=False,
            bank_semantics='Shared conditional-motion coordinates; not asserted functional groups or intrinsic dimension',
            numerical_floor_events=[])
        self.save()

    def verify_sources(self):
        for file,sha in self.state['source_hashes'].items():
            if digest(file)!=sha:raise ValueError('Frozen input/source changed: '+file)
        for i in range(len(self.positions.rows)):self.positions.raw(i)

    def save(self,cg=None,key=None):
        self.store.save(self.state,cg,key)
        atomic_json(self.out/'status.json',{k:self.state.get(k) for k in
            ('stage','run_state','position','condition_index','last_solver','last_quality','last_budget','new_training_updates','CE_or_accuracy_acceptance')}
            |dict(bank_columns=len(self.state['bank']),observations=len(self.state['observations']),
                actual_training_step=self.positions.rows[self.state['position']]['step']))

    @contextmanager
    def timed(self,label):
        if self.g and self.g.device.type=='cuda':torch.cuda.synchronize()
        start=time.monotonic()
        try:yield
        finally:
            if self.g and self.g.device.type=='cuda':torch.cuda.synchronize()
            with (self.out/'TIMING.jsonl').open('a') as f:f.write(json.dumps(dict(label=label,seconds=time.monotonic()-start,position=self.state['position']))+'\n')

    def change_position(self,index):
        if self.loaded_index==index:return
        anchor=self.positions.prepare(index)
        with self.timed('load_actual_recorded_position'):
            self.g,self.identity=load_eight_geometry(self.out/'runtime',self.protocol['inventory'],self.out/anchor['file'])
        expected=self.protocol.get('source',{})
        for k in ('parameter_dimension','inventory_sha256','probe_sha256'):
            if k in expected and self.identity[k]!=expected[k]:raise ValueError('Geometry changed: '+k)
        self.state['position']=index;self.state['anchor']=anchor;self.loaded_index=index
        self.bank_cache={};self.demand_cache={}

    def settings(self,target):
        weights={t:1. for t in self.g.tasks if t!=target}
        return weights,self.args.gamma*sum(self.scales[t] for t in weights)

    def demand(self,condition,index=None):
        index=self.state['position'] if index is None else index
        key=(index,condition['id'])
        path=self.out/'demands'/f'p{index}_{condition["id"]}.npz'
        if path.exists():
            with np.load(path) as z:return z['y'].copy()
        with self.timed('fixed_functional_demand'):
            if condition['family']=='normal_response':
                left,right=self.positions.direction_interval(index)
                d=self.positions.flat(right).double()-self.positions.flat(left).double()
                norm=float(d.norm());direction=self.g.numpy_to_tensor(d.numpy())
                direction=direction/norm if norm else torch.zeros_like(direction)
            else:
                expert=self.g.c.experts[self.g.tasks.index(condition['target'])].to(self.g.device)
                direction=expert-self.g.point;norm=vnorm(direction)
                direction=direction/norm if norm else torch.zeros_like(direction)
            y=self.g.tensor_to_numpy(self.g.jvp(condition['target'],direction))
            if 'class_label' in condition:
                labels=torch.cat([labels.detach().cpu() for _,labels in self.g.c.batches[condition['target'],self.g.probe]])
                if len(y)%len(labels):raise ValueError('Functional output/sample layout differs')
                # A task-linked request on the samples belonging to this class;
                # mask entire probability-tangent rows, not arbitrary parameters.
                y=(y.reshape(len(labels),-1)*(labels.numpy()==condition['class_label'])[:,None]).reshape(-1)
        path.parent.mkdir(exist_ok=True);np.savez(path,y=y)
        return y

    def import_cg(self,key,oldkey):
        if key in self.state['solvers']:return
        cg=Transactions(self.args.old).cg(self.old,oldkey)
        if cg is None:raise ValueError('Missing old solver: '+oldkey)
        self.save(cg,key)

    def refine(self,session,key,desired,compensation=False):
        """One saved batch, then fresh residual observation; caller repeats same key."""
        cg=self.store.cg(self.state,key)
        measured=(self.out/'solves'/key/'LATEST.json')
        if not measured.exists():
            # Observe the imported four-step point before any further iteration.
            if cg is None:
                _,cg=session.advance(None,0,0.)
                self.save(cg,key)
        else:
            old=json.loads(measured.read_text())
            if old['budget']['ready']:
                return old
            for _ in range(self.args.quantum):
                if self.stop:return None
                with self.timed('CG_iteration'):
                    info,cg=session.advance(cg,1,0.)
                self.state['last_solver']=dict(key=key,iterations=int(cg['iterations']),solver_info=int(info))
                self.save(cg,key)
                if info<0:raise ArithmeticError('Nonpositive CG curvature; no structure update')
                if info==0:break
        if self.stop:return None
        with self.timed('fresh_reference_and_certificate'):
            motion=session.materialize(cg);cert=session.residual_certificate(cg)
        self.state['last_solver']=dict(key=key,iterations=int(cg['iterations']),observation_uses_fresh_residual=True)
        norm=vnorm(motion);scale=float(np.linalg.norm(np.concatenate(list(desired.values()))))
        budget=error_budget(cert,norm,scale,self.args.relative_error_budget)
        responses=session.observed_responses if isinstance(session,ObservedOutputWLS) else None
        folder=measured.parent;folder.mkdir(parents=True,exist_ok=True)
        file=folder/f'iteration{int(cg["iterations"]):08d}.pt'
        if not file.exists():
            temp=file.with_suffix('.tmp');torch.save(dict(motion=motion.cpu()),temp);temp.replace(file)
        if responses is not None:np.savez(file.with_suffix('.npz'),**responses)
        previous=json.loads(measured.read_text()) if measured.exists() else None
        change=None
        if previous and previous['motion_file']!=str(file.relative_to(self.out)):
            oldmotion=torch.load(self.out/previous['motion_file'],map_location=self.g.device,mmap=True,weights_only=True)['motion']
            change=vnorm(motion-oldmotion)/max(norm,np.finfo(float).tiny)
        report=dict(key=key,iterations=int(cg['iterations']),certificate=cert,budget=budget,
            parameter_norm=norm,relative_motion_change=change,
            motion_file=str(file.relative_to(self.out)),response_file=str(file.with_suffix('.npz').relative_to(self.out)) if responses is not None else None)
        if responses is not None:
            nonzero=[t for t,d in desired.items() if np.linalg.norm(d)>0]
            if not compensation and len(nonzero)==1:
                target=nonzero[0];weights={t:w for t,w in session.weights.items() if t!=target}
                report['functional_measurement']=measurement(responses,norm*norm,target,desired[target],weights,session.rho)
            if previous and previous.get('response_file'):
                with np.load(self.out/previous['response_file']) as z:
                    report['per_task_functional_change']={t:float(np.linalg.norm(responses[t]-z[t])/max(scale,np.finfo(float).tiny)) for t in responses}
        if previous and report['iterations']==previous['iterations'] and not budget['ready']:
            self.state['run_state']='NUMERICAL_FLOOR';self.save()
            raise ArithmeticError('Recursive CG stopped but fresh error budget is unresolved; no structure update')
        if previous and cert['primal_normal_residual_norm']>=previous['certificate']['primal_normal_residual_norm']*.99 and change is not None and change<np.finfo(float).eps*100:
            self.state['numerical_floor_events'].append(report)
            if not budget['ready']:
                self.state['run_state']='NUMERICAL_FLOOR';self.save()
                raise ArithmeticError('Numerical floor before requested error budget; candidate retained')
        atomic_json(folder/f'iteration{int(cg["iterations"]):08d}.json',report);atomic_json(measured,report)
        self.state.update(last_quality=cert,last_budget=budget);self.save()
        return report if budget['ready'] else None

    def full_reference(self,condition,key,oldkey=None,y=None):
        target=condition['target'];weights,rho=self.settings(target)
        y=self.demand(condition) if y is None else y
        desired={t:np.zeros(self.g.output_dimensions[t]) for t in weights};desired[target]=y
        session=ObservedOutputWLS(self.g,dict(weights,**{target:1.}),rho,desired,self.diagonals)
        if oldkey:self.import_cg(key,oldkey)
        if oldkey and not self.state.get('original_rhs_verified'):
            # Independently recover original b from its first ordinary CG state.
            for f in sorted((self.args.old/'transactions').glob('*/state.json')):
                s=json.loads(f.read_text());r=s.get('solvers',{}).get(oldkey)
                if not r:continue
                with np.load(self.args.old/r['file']) as z:
                    if int(z['iterations'])!=1:continue
                    u=z['p']-z['r'];b=u/(float(u@u)/float(z['rr']));break
            mismatch=float(np.linalg.norm(b-session.b)/np.linalg.norm(session.b))
            if mismatch>1e-11:raise ValueError('Refinement demand differs from original')
            self.state['original_rhs_verified']=dict(relative_difference=mismatch);self.save()
        return self.refine(session,key,desired)

    def load_motion(self,ref,device=None):
        return torch.load(self.out/ref['motion_file'],map_location=device or self.g.device,mmap=True,weights_only=True)['motion']

    def bank_data(self):
        vectors=[];responses={t:[] for t in self.g.tasks}
        for entry in self.state['bank']:
            # CPU mmap storage avoids turning GPU memory into a dimension cap.
            vectors.append(self.load_motion(entry,device='cpu'))
            file=self.out/'bank_responses'/f'p{self.state["position"]}_probe{self.g.probe}_{entry["id"]}.npz'
            if not file.exists():
                with self.timed('all_shared_directions_new_Jacobian'):
                    v=vectors[-1].to(self.g.device);ys={t:self.g.tensor_to_numpy(self.g.jvp(t,v)) for t in self.g.tasks}
                file.parent.mkdir(exist_ok=True);np.savez(file,**ys)
            with np.load(file) as z:
                for t in self.g.tasks:responses[t].append(z[t].copy())
        G=np.array([[float(a@b) for b in vectors] for a in vectors]).reshape(len(vectors),len(vectors))
        Y={t:np.column_stack(rows) if rows else np.empty((self.g.output_dimensions[t],0)) for t,rows in responses.items()}
        return vectors,Y,G

    def evaluate(self,condition,ref,label):
        v=self.load_motion(ref);vectors,Y,G=self.bank_data();y=self.demand(condition)
        weights,rho=self.settings(condition['target'])
        with np.load(self.out/ref['response_file']) as z:full={t:z[t].copy() for t in z.files}
        report=compare(Y,G,condition['target'],y,weights,rho,full,float(v@v),ref['certificate'],
            gain=1/(1+(ref['certificate']['parameter_error_bound']/max(vnorm(v),np.finfo(float).tiny))))
        report['Euclidean_projection']=projection(G,np.array([float(b@v.cpu()) for b in vectors]),float(v@v))
        report.update(condition=condition,position=self.state['position'],basis_ids=[e['id'] for e in self.state['bank']],
            all_targets_share_the_same_bank=True,not_a_functional_group_or_low_dimension_claim=True)
        file=self.out/'evaluations'/f'{label}.json';atomic_json(file,report)
        return report

    def add_reference(self,ref,condition):
        v=self.load_motion(ref,device='cpu');n=vnorm(v)
        vectors,_,G=self.bank_data();columns=[b.numpy() for b in vectors]
        # Discover a missing full-parameter component, rather than selecting a
        # tensor or storing the original full solution as another sample.
        residual=v.clone();coefficients=np.zeros(len(vectors))
        for _ in range(2):
            rhs=np.array([float(b@residual) for b in vectors])
            step=quadratic_solve(G,rhs) if len(vectors) else np.empty(0)
            coefficients+=step
            for b,z in zip(vectors,step):residual-=float(z)*b
        residual_norm=vnorm(residual)
        if not residual_norm:
            self.state.setdefault('extension_observations',[]).append(dict(condition=condition,
                position=self.state['position'],reason='Exact zero full-space innovation'))
            self.save();return
        unit=residual/residual_norm
        independent,rank=independent_column(columns,unit.numpy())
        # Avoid turning unresolved numerical innovations into learned coordinates.
        lower=float(rank.get('residual_norm_lower',rank.get('relative_residual') or 0.))
        parent_error=sum(abs(z)*e.get('coordinate_error_bound',0.) for z,e in zip(coefficients,self.state['bank']))
        uncertainty=ref['certificate']['parameter_error_bound']+parent_error
        resolved=residual_norm>uncertainty and lower>uncertainty/residual_norm
        if independent and resolved:
            folder=self.out/'bank';folder.mkdir(exist_ok=True);bid=f'b{len(self.state["bank"]):06d}'
            file=folder/(bid+'.pt');torch.save(dict(motion=unit.cpu()),file)
            self.state['bank'].append(dict(id=bid,motion_file=str(file.relative_to(self.out)),
                kind='full_conditional_residual',condition=condition,source_position=self.state['position'],
                fixed_demand_file=str((self.out/'demands'/f'p{self.state["position"]}_{condition["id"]}.npz').relative_to(self.out)),
                original_norm=n,residual_norm=residual_norm,
                parent_ids=[e['id'] for e in self.state['bank']],projection_coefficients=coefficients,
                coordinate_error_bound=uncertainty/residual_norm,
                source_reference=ref,rank=rank,
                independent_functional_validation_required=True))
        self.state.setdefault('extension_observations',[]).append(dict(condition=condition,position=self.state['position'],
            independent=bool(independent),numerically_resolved=bool(resolved),rank=rank,
            residual_norm=residual_norm,numerical_innovation_error_bound=uncertainty))
        self.save()

    def summarize_joint_coordinates(self):
        vectors,Y,G=self.bank_data()
        # Continuous coordinate loadings; native tensors are readout regions,
        # never driver masks or automatic functional-group memberships.
        regions={};offset=0
        for name,size in zip(self.g.c.keys,self.g.c.sizes):
            regions[name]=[float(v[offset:offset+size]@v[offset:offset+size]) for v in vectors]
            offset+=size
        atomic_json(self.out/'shared_structure'/f'p{self.state["position"]}.json',dict(
            basis_ids=[b['id'] for b in self.state['bank']],coordinate_gram=G,
            task_functional_cross_terms={t:ys.T@ys for t,ys in Y.items()},
            continuous_parameter_region_energy=regions,
            regions_are_readouts_not_discovery_support_constraints=True,
            sampled_rank_is_not_intrinsic_dimension=True,
            low_dimension_requires_independent_demand_and_position_evidence=True))

    def pilot_compensation(self,index):
        self.change_position(index);target='DTD';weights,rho=self.settings(target)
        oldaid='a0_DTD_normal_response_gamma0.1_tensor3_b0'
        meta=json.loads((self.args.old/'graph_actions'/oldaid/'MODE_SOURCE.json').read_text())
        segment=slice(meta['driver_start'],meta['driver_stop']);local=np.load(self.args.old/meta['driver_file'])
        driver=torch.zeros_like(self.g.point);driver[segment]=self.g.numpy_to_tensor(local)
        raw_file=self.out/f'raw_driver_p{index}.npz'
        if raw_file.exists():
            with np.load(raw_file) as z:raw={t:z[t].copy() for t in z.files}
        else:
            with self.timed('fixed_driver_task_responses'):
                raw={t:self.g.tensor_to_numpy(self.g.jvp(t,driver)) for t in self.g.tasks}
            np.savez(raw_file,**raw)
        desired={t:raw[t] for t in weights};key=f'pilot_T_p{index}'
        session=ObservedOutputWLS(self.g,weights,rho,desired,self.diagonals,segment)
        if index==0:
            with np.load(self.args.old/'graph_actions'/oldaid/'protected_driver_response.npz') as z:
                if any(not np.allclose(raw[t],z[t],rtol=1e-11,atol=1e-13) for t in weights):
                    raise ValueError('Protected-driver demand changed')
            self.import_cg(key,'compensation_'+oldaid)
        ref=self.refine(session,key,desired,compensation=True)
        if ref is None:return None
        comp=-self.load_motion(ref);lift=driver+comp
        with np.load(self.out/ref['response_file']) as z:
            response={t:raw[t]-z[t] for t in weights}
        response[target]=self.g.tensor_to_numpy(self.g.jvp(target,lift))
        folder=self.out/'pilot_compensation';folder.mkdir(exist_ok=True)
        file=folder/f'p{index}.pt';torch.save(dict(motion=lift.cpu()),file)
        np.savez(file.with_suffix('.npz'),**response)
        y=self.demand(dict(id='DTD_normal_response',target='DTD',family='normal_response'))
        costs=dict(driver=measurement(raw,float(driver@driver),target,y,weights,rho),
            lifted=measurement(response,float(lift@lift),target,y,weights,rho),
            target_response_difference_norm=float(np.linalg.norm(response[target]-raw[target])),
            target_response_difference=response[target]-raw[target],
            per_task_compensation_cross_terms={t:float(2*raw[t]@(response[t]-raw[t])) for t in weights},
            certificate=ref['certificate'],budget=ref['budget'],
            driver_support_is_a_baseline_only=True)
        atomic_json(folder/f'p{index}.json',costs)
        return dict(id='protected_baseline',kind='protected_baseline',motion_file=str(file.relative_to(self.out)),
            original_norm=1.,source_reference=ref,condition=dict(id='DTD_normal_response',target='DTD',family='normal_response'))

    def transport(self):
        """Relocalize EVERY shared column; old demand is fixed per source column."""
        i=self.state.get('transport_index',0)
        if i>=len(self.state['bank']):return True
        entry=self.state['bank'][i]
        if entry['kind']=='protected_baseline':
            new=self.pilot_compensation(self.state['position'])
            if new is None:return False
        else:
            with np.load(self.out/entry['fixed_demand_file']) as z:y=z['y'].copy()
            key=f'relocalize_p{self.state["position"]}_{entry["id"]}'
            ref=self.full_reference(entry['condition'],key,y=y)
            if ref is None:return False
            # Apply the same residual coordinate recipe at the new position.
            # Rebuild parents first; no independent normalization changes units.
            v=self.load_motion(ref)
            parents={e['id']:e for e in self.state['transport_bank']}
            for pid,z in zip(entry['parent_ids'],entry['projection_coefficients']):
                v-=float(z)*self.load_motion(parents[pid])
            v/=entry['residual_norm']
            folder=self.out/'relocalized';folder.mkdir(exist_ok=True);file=folder/(key+'.pt')
            torch.save(dict(motion=v.cpu()),file);new=dict(entry,motion_file=str(file.relative_to(self.out)),source_reference=ref)
        self.state.setdefault('transport_bank',[]).append(dict(entry,**{k:new[k] for k in ('motion_file','source_reference')}))
        self.state['transport_index']=i+1;self.save();return False

    def pilot_cross(self):
        condition=self.state['conditions'][self.state.get('cross_condition_index',0)]
        ref=self.full_reference(condition,'cross_p1_'+condition['id'])
        if ref is None:return
        frozen=self.evaluate(condition,ref,'cross_p1_'+condition['id']+'_frozen_all')
        original=self.state['bank'];self.state['bank']=self.state['transport_bank']
        # Distinct response cache IDs are mandatory for frozen/relocalized columns.
        self.state['bank']=[dict(e,id=e['id']+'_relocalized') for e in self.state['bank']]
        try:rebuilt=self.evaluate(condition,ref,'cross_p1_'+condition['id']+'_relocalized_all')
        finally:self.state['bank']=original
        atomic_json(self.out/'cross_position'/f'{condition["id"]}.json',dict(
            position=self.state['position'],same_new_demand=True,frozen_all_columns=frozen,relocalized_all_columns=rebuilt,
            comparison_scope='Full current bank; protected seed uses T, conditional coordinates use joint-WLS relocalization',
            fixed_source_demands_used_for_relocalization=True,new_demand_used_only_for_evaluation=True))
        self.state['cross_condition_index']=self.state.get('cross_condition_index',0)+1
        if self.state['cross_condition_index']>=len(self.state['conditions']):
            self.state.update(stage='STUDY_SELECT',condition_index=0)
        self.save()

    def validate_position(self):
        # Probe 1 is disjoint from learning observations. Never add its reference
        # movements to the bank, nor use its results to choose a dimension.
        ci=self.state.get('validation_index',0)
        if ci>=len(self.g.tasks):
            self.g.probe=0
            if self.state['position']==0:
                self.state.update(stage='PILOT_TRANSPORT',position=1,transport_index=0,
                    transport_bank=[],cross_condition_index=0)
                self.save();self.change_position(1)
            elif self.state['position']+1>=len(self.positions.rows):
                self.state.update(stage='INPUT_EXHAUSTED',run_state='COMPLETE');self.save()
            else:
                self.change_position(self.state['position']+1)
                self.state.update(stage='STUDY_SELECT',condition_index=0);self.save()
            return
        target=self.g.tasks[ci];self.g.probe=1
        # Distinct demand and response cache namespace for held-out samples.
        normal=dict(id=f'{target}_normal_response_heldout',target=target,family='normal_response')
        expert=dict(id=f'{target}_expert_response_heldout',target=target,family='expert_response')
        y=.37*self.demand(normal)+.63*self.demand(expert)
        condition=dict(id=f'{target}_heldout_mixture',target=target,family='heldout')
        demand_file=self.out/'demands'/f'p{self.state["position"]}_{condition["id"]}.npz'
        if not demand_file.exists():np.savez(demand_file,y=y)
        key=f'heldout_p{self.state["position"]}_{target}'
        ref=self.full_reference(condition,key,y=y)
        if ref:
            # Cache names depend on probe, so held-out responses can never reuse
            # learning-window response matrices.
            self.evaluate(condition,ref,key)
            self.state['validation_index']=ci+1;self.save()
        self.g.probe=0

    def step(self):
        stage=self.state['stage'];condition=dict(id='DTD_normal_response',target='DTD',family='normal_response')
        if stage=='PILOT_REFERENCE':
            ref=self.full_reference(condition,'pilot_reference','functional_reference_a0_DTD_normal_response_gamma0.1')
            if ref:
                self.state.update(pilot_reference=ref,stage='PILOT_COMPENSATION');self.save()
        elif stage=='PILOT_COMPENSATION':
            entry=self.pilot_compensation(0)
            if entry:
                self.state['bank']=[entry] # Historical one-mode diagnostic only.
                self.evaluate(condition,self.state['pilot_reference'],'pilot_refined_single_mode')
                self.state.update(legacy_baseline=entry,bank=[],stage='STUDY_SELECT',condition_index=0)
                self.save()
        elif stage=='PILOT_TRANSPORT':
            if self.transport():self.state['stage']='PILOT_CROSS';self.save()
        elif stage=='PILOT_CROSS':self.pilot_cross()
        elif stage=='VALIDATE_POSITION':self.validate_position()
        elif stage=='STUDY_SELECT':
            ci=self.state['condition_index'];pi=self.state['position']
            if ci>=len(self.state['conditions']):
                self.summarize_joint_coordinates()
                self.state.update(stage='VALIDATE_POSITION',validation_index=0);self.save();return
            condition=self.state['conditions'][ci];key=f'study_p{pi}_{condition["id"]}'
            # Existing shared space is evaluated first, before full solve/enrichment.
            if self.state.get('study_key')!=key:
                vectors,Y,G=self.bank_data();y=self.demand(condition);weights,rho=self.settings(condition['target'])
                from control_math import restricted_problem
                initial=(restricted_problem(Y,G,condition['target'],y,weights,rho,problem='weighted_ls') if len(vectors)
                    else measurement({t:np.zeros(self.g.output_dimensions[t]) for t in self.g.tasks},0.,condition['target'],y,weights,rho))
                atomic_json(self.out/'preexisting'/f'{key}.json',dict(condition=condition,position=pi,
                    basis_ids=[b['id'] for b in self.state['bank']],undamped=initial))
                self.state['study_key']=key;self.save()
            ref=(self.state['pilot_reference'] if pi==0 and ci==0 else self.full_reference(condition,key))
            if ref:
                before=self.evaluate(condition,ref,key+'_before')
                self.add_reference(ref,condition)
                after=self.evaluate(condition,ref,key+'_after')
                self.state['observations'].append(dict(position=pi,condition=condition,
                    before=f'evaluations/{key}_before.json',after=f'evaluations/{key}_after.json'))
                self.state['condition_index']+=1;self.save()
        else:raise ValueError(stage)

    def run(self):
        while not self.stop and self.state['stage']!='INPUT_EXHAUSTED':self.step()
        if self.stop:self.state['run_state']='PAUSED';self.save()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('old','source','trace','coordinates','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--quantum',type=int,default=4);p.add_argument('--relative-error-budget',type=float,default=.001)
    p.add_argument('--gamma',type=float,default=.1);a=p.parse_args()
    if a.quantum<1 or not 0<a.relative_error_budget<1:raise ValueError('Invalid numerical policy')
    try:Runner(a).run()
    except Exception:
        import traceback
        atomic_json(a.output/'FAILURE.json',dict(traceback=traceback.format_exc()));raise
