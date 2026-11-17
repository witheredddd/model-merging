"""Construct conditional functional modes along an existing normal-training path.

Training positions are the carrier only. Desired task responses induce new full
conditional motions and protected graph modes. Accuracy/CE never accept/reject
the modes, never choose a new carrier point, and never trigger training updates.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from bootstrap import R3
from worker import Engine,Paused
from io_state import Transactions,atomic_json,digest,plain,choose_condition
from task_runtime import load_eight_geometry
from operators import OutputDualWLS,vnorm
from control_math import CandidateMotion,canonical,independent_column,graphwise_modes,actuate,quadratic_solve,restricted_problem
from finite_control import probabilities
from carrier import TrainingPositions


class ObservedOutputWLS(OutputDualWLS):
    """Retain responses from the existing operator observation, without new JVPs."""
    def apply(self,x):
        self.calls+=1;q=self.scale*x;image=self.forward(self.transpose(q))
        self.observed_responses={};offset=0
        for task,n in zip(self.tasks,self.lengths):
            self.observed_responses[task]=image[offset:offset+n]/(self.rho*np.sqrt(self.weights[task]));offset+=n
        return self.scale*(q+image/self.rho)


class SpaceEngine(Engine):
    def __init__(self,out,source,trace,coordinates,steps=4,gamma=.1):
        self.trace_path=Path(trace);self.coordinates=Path(coordinates);self.gamma=gamma
        self.space_codes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
        names=json.loads(self.coordinates.read_text())['parameter_keys']
        self.positions=TrainingPositions(trace,names,Path(out))
        if (Path(out)/'CURRENT.json').exists():
            s=Transactions(out).load()
            if s['space_source_hashes']!=self.space_codes:raise ValueError('Explicit space-source migration required')
            if s['carrier_manifest_sha256']!=self.positions.manifest_sha256:raise ValueError('Carrier manifest changed')
        super().__init__(out,source,steps)
        if self.state['anchor']['source_step']!=self.positions.rows[self.state['anchor_serial']]['step']:
            raise ValueError('Carrier position identity mismatch')
        self.carrier_direction_cache=None

    def initialize(self):
        (self.out/'runtime').mkdir(exist_ok=True)
        shutil.copy2(self.source/'runtime/probes.json',self.out/'runtime/probes.json')
        current=json.loads((self.source/'CURRENT.json').read_text());legacy=json.loads((self.source/current['state']).read_text())
        if legacy['phase']!='PAUSED':raise ValueError('Old experiment must remain paused')
        tasks=list(self.protocol['tasks'])
        conditions=[dict(id=f'{t}_{family}_gamma{self.gamma}',target=t,gamma=self.gamma,source_family=family,
            last_turn=-1,seconds=0.,visits=0) for family in ('normal_response','expert_response') for t in tasks]
        self.state=dict(schema='functional-space-1',source_hashes=self.codes,space_source_hashes=self.space_codes,
            generation=0,run_state='RUNNING',stage='SELECT',source_protocol_sha256=digest(self.source/'PROTOCOL.json'),
            source_current=current,carrier_manifest=str(self.positions.manifest),carrier_manifest_sha256=self.positions.manifest_sha256,
            ordinary_training_proof=self.positions.verify_normal_recipe(),
            carrier_position_count=len(self.positions.rows),anchor_serial=0,anchor=self.positions.prepare(0),
            turn=0,conditions=conditions,graphs={},assets={},solvers={},updates=[],carrier_visits=[],space_assessments=[],
            legacy_observations=dict(directory=str(self.source),role='Historical candidates; not functional groups'),
            method=dict(research_object='Task-demand-induced conditional motions and protected drive/compensation modes',
                carrier='Actual recorded normal-learning positions; no new CE optimization or synthetic interpolation',
                full_reference_problem='Explicit PDF14 joint weighted functional fit on all parameters',
                desired_functional_response_sources=['Observed normal-training task response','Existing expert-motion task response'],
                target_weight=1.,parameter_metric='I baseline',gamma=self.gamma,
                CE_and_Top1_are_read_only=True,accuracy_change_required=False,CE_descent_required=False,
                counterfactual_mode_model_is_not_written_to_carrier=True,
                temporal_motion_PCA_is_not_the_learning_objective=True,normal_update_reconstruction_is_auxiliary=True,
                numerical_work_quantum=self.steps,work_quantum_is_not_dimension_or_total_solve_limit=True))
        self.store.save(self.state)

    def save(self,cg=None,key=None):
        self.store.save(self.state,cg,key)
        atomic_json(self.out/'status.json',dict(stage=self.state['stage'],run_state=self.state['run_state'],
            condition=self.state.get('condition'),carrier_index=self.state['anchor_serial'],
            actual_training_step=self.state['anchor'].get('source_step'),
            normal_training_positions=self.state['carrier_position_count'],
            assessments=len(self.state['space_assessments']),
            new_training_updates=0,CE_Top1_acceptance_gate=False,last_solver=self.state.get('last_solver'),
            last_quality=self.state.get('last_quality')))

    def training_direction(self):
        index=self.state['anchor_serial']
        if self.carrier_direction_cache is None or self.carrier_direction_cache[0]!=index:
            left,right=self.positions.direction_interval(index)
            delta=self.positions.flat(right).double()-self.positions.flat(left).double()
            norm=float(delta.norm())
            if norm==0:raise ValueError('Recorded positions have an exactly zero motion')
            self.carrier_direction_cache=(index,self.g.numpy_to_tensor(delta.numpy()),norm,left,right)
        return self.carrier_direction_cache[1:]

    def demand(self):
        key=(self.state['anchor_serial'],self.state['condition'])
        if self.demand_cache is None or self.demand_cache[0]!=key:
            task=self.condition()['target'];family=self.condition()['source_family']
            with self.timed('functional_demand_from_fixed_resources'):
                if family=='normal_response':
                    delta,norm,left,right=self.training_direction();reference=delta/norm
                    info=dict(source='Current JVP of an actual normal-training motion direction',
                        interval_steps=[self.positions.rows[left]['step'],self.positions.rows[right]['step']],
                        original_parameter_motion_norm=norm)
                else:
                    expert=self.g.c.experts[self.g.tasks.index(task)].to(self.g.device)
                    reference=expert-self.g.point;norm=vnorm(reference)
                    if not norm:raise ValueError('Exactly zero expert reference motion')
                    reference=reference/norm
                    info=dict(source='Existing trained expert parameter motion as a functional reference',normalization=norm)
                y=self.g.jvp(task,reference)
                # Orientation is demand alignment. This covector is not called CE.
                orientation=-self.g.vjp(task,y)
            info.update(evaluated_training_step=self.state['anchor']['source_step'],
                role='Defines task response only; parameters are not replayed as the learned mode',
                CE_optimization=False,problem='weighted_ls')
            self.demand_cache=(key,y,orientation,info)
            atomic_json(self.out/'demands'/f'a{key[0]}_{key[1]}.json',info)
        return self.demand_cache[1:]

    def reference(self):
        target,weights,rho=self.settings();y,_,info=self.demand();all_weights=dict(weights);all_weights[target]=1.
        desired={t:np.zeros(self.g.output_dimensions[t]) for t in self.g.tasks};desired[target]=self.g.tensor_to_numpy(y)
        session=ObservedOutputWLS(self.g,all_weights,rho,desired,self.diagonals)
        key=f'functional_reference_a{self.state["anchor_serial"]}_{self.state["condition"]}'
        cg=self.advance(session,key)
        with self.timed('full_conditional_observation'):
            motion=session.materialize(cg);cert=session.residual_certificate(cg)
        norm=vnorm(motion);confidence=norm/(norm+cert['parameter_error_bound']) if norm else 0.
        self.state['last_quality']=dict(cert,reference_confidence=confidence,conditional_optimum_claim=False)
        candidate=CandidateMotion(key,str(self.state['anchor_serial']),cert['primal_normal_residual_norm'],cert['parameter_error_bound'],cert)
        self.state['candidate_motion']=dict(type='CandidateMotion',**asdict(candidate),functional_mode_claim=False)
        folder=self.out/'references';folder.mkdir(exist_ok=True);reference_file=folder/(key+'.pt')
        temp=reference_file.with_suffix('.tmp');torch.save(dict(motion=motion.detach().cpu()),temp);temp.replace(reference_file)
        self.state['reference_file']=str(reference_file.relative_to(self.out))
        response_file=reference_file.with_suffix('.npz');np.savez(response_file,**session.observed_responses)
        self.state['reference_response_file']=str(response_file.relative_to(self.out))
        choices=[];offset=0
        for index,(name,size) in enumerate(zip(self.g.c.keys,self.g.c.sizes)):
            segment=slice(offset,offset+size);offset+=size
            energy=float(motion[segment].square().sum());gid=f'{self.state["condition"]}_tensor{index}'
            old=self.state['graphs'].get(gid,{})
            priority=(self.state['turn']-old.get('last_turn',-1)+1)*np.sqrt(energy)/max(1,old.get('visits',0))
            choices.append((priority,gid,name,segment,energy))
        _,gid,name,segment,energy=max(choices,key=lambda x:x[0])
        if energy==0:raise ValueError('Exactly zero induced functional motion; no random replacement')
        graph=self.state['graphs'].setdefault(gid,dict(id=gid,condition=self.state['condition'],driver_name=name,
            start=segment.start,stop=segment.stop,bases=[],observations=[],visits=0))
        graph['last_turn']=self.state['turn'];graph['visits']+=1
        unit,rawnorm=canonical(self.g.tensor_to_numpy(motion[segment]));columns=[np.load(self.out/b['driver_file']) for b in graph['bases']]
        independent,rank=independent_column(columns,unit)
        graph['observations'].append(dict(source=key,quality=cert,rank=rank,independent=bool(independent),
            original_normal_training_motion_is_not_a_functional_group=True))
        if independent:
            residual=unit.copy()
            if columns:
                Q=np.column_stack(columns)
                for _ in range(2):residual-=Q@np.linalg.lstsq(Q.T@Q,Q.T@residual,rcond=None)[0]
            residual,n=canonical(residual);folder=self.out/'drivers'/gid;folder.mkdir(parents=True,exist_ok=True)
            file=folder/f'{len(columns):08d}.npy';np.save(file,residual)
            graph['bases'].append(dict(driver_file=str(file.relative_to(self.out)),source_anchor=self.state['anchor_serial'],
                source=key,reference_confidence=confidence,numerical_residual_amplification=1/max(n,np.finfo(float).tiny)))
        queue=[dict(graph=g['id'],basis=i) for g in self.state['graphs'].values() if g['condition']==self.state['condition'] for i in range(len(g['bases']))]
        self.state.update(current_graph=gid,build_queue=queue,build_index=0,stage='BUILD')
        self.condition()['uncertainty']=cert['parameter_error_bound']/max(norm,np.finfo(float).tiny)
        self.condition()['information']=(rank.get('relative_residual') or 0.)*confidence
        self.save()

    def select(self):
        # Every observed position receives a task-conditioned construction; source
        # families and targets remain eligible without any performance gate.
        index=choose_condition(self.state['conditions'],self.state['turn']);c=self.state['conditions'][index]
        self.state.update(condition_index=index,condition=c['id'],stage='REFERENCE',condition_compute_seconds=0.,reuse_attempt=False)
        self.save()

    def control(self):
        target,weights,rho=self.settings();y,orientation,info=self.demand()
        assets=[self.state['assets'][self.asset_key(e)] for e in self.state['build_queue']]
        vectors=[self.cache.get(self.out/a['motion_file']) for a in assets]
        G=np.array([[float(a@b) for b in vectors] for a in vectors]);Y={t:[] for t in self.g.tasks}
        for a in assets:
            with np.load(self.out/a['response_file']) as z:
                for t in self.g.tasks:Y[t].append(z[t].copy())
        Y={t:np.column_stack(columns) for t,columns in Y.items()}
        covector=np.array([float(orientation@v) for v in vectors])
        errors=np.array([a['certificate']['parameter_error_bound'] for a in assets]);q=self.state['last_quality']['reference_confidence']
        with self.timed('conditional_spectral_space'):
            modes,provenance=graphwise_modes(Y,G,target,weights,rho,covector,errors,[a['graph'] for a in assets],q,[a['reference_confidence'] for a in assets])
            result=actuate(Y,G,target,self.g.tensor_to_numpy(y),weights,rho,modes,1.,q)
        coefficients=result['applied_base_coefficients'];constructed=torch.zeros_like(self.g.point)
        for v,z in zip(vectors,coefficients):constructed+=float(z)*v
        reference=self.cache.get(self.out/self.state['reference_file'])
        rhs=np.array([float(v@reference) for v in vectors]);projection=quadratic_solve(G,rhs)
        reconstructed=torch.zeros_like(reference)
        for v,z in zip(vectors,projection):reconstructed+=float(z)*v
        # The target is the induced full conditional movement, not fitting an
        # arbitrary trajectory basis. Training-motion projection is auxiliary.
        report=dict(carrier_index=self.state['anchor_serial'],training_step=self.state['anchor']['source_step'],
            condition=self.state['condition'],graph_actions=[a['id'] for a in assets],
            spectral_axes=modes.axes,eigenvalues=modes.eigenvalues,demand_alignment=-modes.CE_slopes,
            uncertainty_confidence=modes.confidence,mode_graph_provenance=provenance,
            control=result,conditional_motion_projection_coefficients=projection,
            conditional_parameter_reconstruction_relative_error=vnorm(reference-reconstructed)/max(vnorm(reference),np.finfo(float).tiny),
            achieved_functional_control={t:dict(response_norm=float(np.linalg.norm(Y[t]@coefficients)),
                target_relative_error=(float(np.linalg.norm(Y[t]@coefficients-self.g.tensor_to_numpy(y))/max(vnorm(y),np.finfo(float).tiny)) if t==target else None)) for t in self.g.tasks},
            CE_or_accuracy_acceptance=False,carrier_is_not_changed_by_control=True,
            motion_is_induced_by_functional_problem=True)
        with np.load(self.out/self.state['reference_response_file']) as ref:
            report['conditional_functional_reconstruction']={t:dict(
                projection_relative_error=float(np.linalg.norm(Y[t]@projection-ref[t])/max(np.linalg.norm(ref[t]),np.finfo(float).tiny)),
                full_reference_response_energy=float(ref[t]@ref[t]),
                controlled_response_energy=float(np.linalg.norm(Y[t]@coefficients)**2)) for t in self.g.tasks}
        with self.timed('counterfactual_functional_observation_no_acceptance'):
            before=probabilities(self.g,self.g.point);after=probabilities(self.g,self.g.point+constructed)
        report['counterfactual']={t:dict(CE_change=after[t]['CE']-before[t]['CE'],Top1_change=after[t]['Top1']-before[t]['Top1'],
            finite_response_energy=float((after[t]['phi']-before[t]['phi']).square().sum())) for t in self.g.tasks}
        report['counterfactual']['role']='Read-only observation; not a training or acceptance decision'
        path=self.out/'space'/f'a{self.state["anchor_serial"]}_{self.state["condition"]}.json';atomic_json(path,report)
        self.state['space_assessments'].append(dict(file=str(path.relative_to(self.out)),training_step=report['training_step'],condition=self.state['condition']))
        self.state['previous_action']=self.asset_key(self.state['build_queue'][-1]);self.state['reanchor_entry']=self.state['build_queue'][-1]
        self.state['previous_control_file']=str(path.relative_to(self.out))
        self.advance_carrier()

    def advance_carrier(self):
        index=self.state['anchor_serial']
        if index+1>=len(self.positions.rows):
            self.finish_condition();self.state['stage']='INPUT_EXHAUSTED';self.state['run_state']='INPUT_EXHAUSTED';self.save();return
        new=self.positions.prepare(index+1)
        self.state['carrier_visits'].append(dict(from_training_step=self.state['anchor']['source_step'],to_training_step=new['source_step'],
            source='actual recorded normal-learning trajectory',new_optimization=False))
        self.state.update(anchor=new,anchor_serial=index+1,stage='REANCHOR');self.save()
        self.cache.clear();self.carrier_direction_cache=None;self.demand_cache=None
        with self.timed('load_next_actual_training_position'):
            self.g,self.identity=load_eight_geometry(self.out/'runtime',self.protocol['inventory'],self.out/new['file'])
        self.state['stage']='REBUILD';self.save()

    def rebuild(self):
        entry=self.state['reanchor_entry'];old=self.state['assets'][self.state['previous_action']]
        new_id=self.build(entry);new=self.state['assets'][new_id]
        a=self.cache.get(self.out/old['motion_file']);b=self.cache.get(self.out/new['motion_file'])
        target,weights,rho=self.settings();y,_,_=self.demand();y_np=self.g.tensor_to_numpy(y)
        # Old and updated modes are evaluated using the same NEW Jacobian,
        # same NEW demand and same protection weights. Old cached responses
        # are not carried across parameter positions.
        with self.timed('cross_position_frozen_vs_rebuilt'):
            Ya={t:self.g.tensor_to_numpy(self.g.jvp(t,a))[:,None] for t in self.g.tasks}
        with np.load(self.out/new['response_file']) as z:Yb={t:z[t][:,None] for t in self.g.tasks}
        frozen=restricted_problem(Ya,np.array([[float(a@a)]]),target,y_np,weights,rho,problem='weighted_ls')
        retuned=restricted_problem(Yb,np.array([[float(b@b)]]),target,y_np,weights,rho,problem='weighted_ls')
        error=old['certificate']['parameter_error_bound']+new['certificate']['parameter_error_bound'];change=vnorm(a-b)
        event=dict(condition=self.state['condition'],old_action=old['id'],new_action=new_id,
            actual_training_step=self.state['anchor']['source_step'],carrier_source='normal training',
            lift_change_norm=change,combined_compensation_error_bound=error,change_resolved_beyond_error=change>error,
            frozen_mode_at_new_position=frozen,rebuilt_mode_at_new_position=retuned,
            no_CE_or_accuracy_gate=True,operator_recomputed_at_actual_training_position=True,
            mode_columns_for_this_comparison=1,comparison_does_not_claim_complete_graph_or_intrinsic_dimension=True)
        self.state.setdefault('geometry_updates',[]).append(event);atomic_json(self.out/'LAST_GEOMETRY_UPDATE.json',event)
        self.finish_condition()

    def run(self):
        try:
            while not self.stop:
                stage=self.state['stage']
                if stage=='INPUT_EXHAUSTED':return
                if stage=='SELECT':self.select()
                elif stage=='REFERENCE':self.reference()
                elif stage=='BUILD':self.build_phase()
                elif stage=='CONTROL':self.control()
                elif stage in ('REANCHOR','REBUILD'):self.rebuild()
                else:raise ValueError(stage)
            raise Paused()
        except Paused:self.state['run_state']='PAUSED';self.save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--coordinates',type=Path,required=True)
    p.add_argument('--work-quantum',type=int,default=4);p.add_argument('--gamma',type=float,default=.1);args=p.parse_args()
    try:SpaceEngine(args.output,args.source,args.trace,args.coordinates,args.work_quantum,args.gamma).run()
    except Exception:
        import traceback
        atomic_json(args.output/'FAILURE.json',dict(traceback=traceback.format_exc(),resume='CURRENT.json'))
        raise
