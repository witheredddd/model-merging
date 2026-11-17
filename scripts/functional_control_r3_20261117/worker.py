"""Discrete protected-graph learning and actual mode-controlled anchor updates.

No candidate Krylov direction is labelled a functional group. No mode is used
as a CG correction. All learned actions are full-complement T applications.
"""
import argparse
from dataclasses import asdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from bootstrap import CORE,LEGACY
from task_runtime import load_eight_geometry
from control_math import CandidateMotion,canonical,independent_column,graphwise_modes,actuate
from operators import OutputDualWLS,ParameterSupportSchur,ce_cotangent_target,vnorm
from cache import MotionCache
from io_state import Transactions,atomic_json,digest,plain,choose_condition
from finite_control import line_search


class Paused(Exception):pass


class Engine:
    def __init__(self,out,source,steps):
        self.out=Path(out);self.source=Path(source);self.store=Transactions(self.out);self.steps=steps
        self.writer_lock=(self.out/'WRITER.lock').open('a+b')
        if os.name=='nt':
            import msvcrt
            self.writer_lock.write(b'0');self.writer_lock.flush();self.writer_lock.seek(0)
            msvcrt.locking(self.writer_lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(self.writer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.stop=False;signal.signal(signal.SIGTERM,self.pause);signal.signal(signal.SIGINT,self.pause)
        self.protocol=json.loads((self.source/'PROTOCOL.json').read_text())
        self.codes={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
        if (self.out/'CURRENT.json').exists():
            self.state=self.store.load()
            if self.state['source_hashes']!=self.codes:raise ValueError('Explicit code migration required')
        else:self.initialize()
        s=self.state;anchor=s.get('anchor')
        checkpoint=self.out/anchor['file'] if anchor else Path(self.protocol['checkpoint'])
        if anchor and digest(checkpoint)!=anchor['sha256']:raise ValueError('Anchor checksum mismatch')
        self.g,self.identity=load_eight_geometry(self.out/'runtime',self.protocol['inventory'],checkpoint)
        with np.load(self.source/'PRECONDITIONER.npz') as z:self.diagonals={t:z[t].copy() for t in self.g.tasks}
        self.scales=json.loads((self.source/'SCALES.json').read_text())['scales']
        free,_=torch.cuda.mem_get_info() if self.g.device.type=='cuda' else (2**30,2**30)
        self.cache=MotionCache(self.g.device,int(free*.5));self.demand_cache=None
        s['run_state']='RUNNING';self.save()

    def pause(self,*_):self.stop=True

    def initialize(self):
        (self.out/'runtime').mkdir(exist_ok=True)
        shutil.copy2(self.source/'runtime'/'probes.json',self.out/'runtime'/'probes.json')
        ref=json.loads((self.source/'CURRENT.json').read_text());legacy=json.loads((self.source/ref['state']).read_text())
        if legacy['phase']!='PAUSED':raise ValueError('Preserve original experiment at a committed pause first')
        tasks=list(self.protocol['tasks'])
        if 'SUN397' in tasks:tasks=tasks[tasks.index('SUN397'):]+tasks[:tasks.index('SUN397')]
        conditions=[dict(id=f'{t}_gamma{gamma}',target=t,gamma=gamma,last_turn=-1,seconds=0.,visits=0) for gamma in self.protocol['gammas'] for t in tasks]
        self.state=dict(schema=3,source_hashes=self.codes,generation=0,run_state='RUNNING',stage='SELECT',
            source_protocol_sha256=digest(self.source/'PROTOCOL.json'),source_current=ref,
            anchor_serial=0,anchor=None,turn=0,conditions=conditions,graphs={},assets={},solvers={},updates=[],
            legacy_observations=dict(count=len(legacy['provisional_state']['modes']),
                role='Archived incomplete full-space motions, not imported as functional groups',
                directory=str(self.source),state=ref['state']),
            method=dict(full_reference_problem='PDF14 weighted_ls explicitly selected; PDF9 exact primitive remains separate',
                target_family='CE cotangent; J^T y=-grad_CE/scale; not historical expert or CE-unit-parameter JVP',
                parameter_metric='I explicit baseline',target_weight=1.,
                regularization='Fixed initial spectral-unit gamma family, not a newly learned intrinsic scale',
                numerical_work_quantum=self.steps,work_quantum_is_not_total_iteration_or_dimension_cap=True,
                full_complement=True,automatic_Krylov_Galerkin_feedback=False,
                certification_controls_optimality_claims_only=True))
        self.store.save(self.state)

    def save(self,cg=None,key=None):
        self.store.save(self.state,cg,key)
        atomic_json(self.out/'status.json',dict(stage=self.state['stage'],run_state=self.state['run_state'],
            condition=self.state.get('condition'),anchor_serial=self.state['anchor_serial'],
            committed_model_updates=len(self.state['updates']),graph_observations=sum(len(x['bases']) for x in self.state['graphs'].values()),
            last_solver=self.state.get('last_solver'),last_quality=self.state.get('last_quality'),
            source_Krylov_columns_are_functional_modes=False))

    @contextmanager
    def timed(self,label):
        if self.g.device.type=='cuda':torch.cuda.synchronize()
        start=time.monotonic()
        try:yield
        finally:
            if self.g.device.type=='cuda':torch.cuda.synchronize()
            elapsed=time.monotonic()-start
            self.state['condition_compute_seconds']=self.state.get('condition_compute_seconds',0.)+elapsed
            self.state.setdefault('timing',{}).setdefault(label,[]).append(elapsed)
            with (self.out/'TIMING.jsonl').open('a') as f:f.write(json.dumps(dict(stage=label,seconds=elapsed,anchor=self.state['anchor_serial'],condition=self.state.get('condition')))+'\n')

    def condition(self):return self.state['conditions'][self.state['condition_index']]

    def settings(self):
        c=self.condition();weights={t:1. for t in self.g.tasks if t!=c['target']}
        return c['target'],weights,c['gamma']*sum(self.scales[t] for t in weights)

    def demand(self):
        key=(self.state['anchor_serial'],self.state['condition'])
        if self.demand_cache is None or self.demand_cache[0]!=key:
            with self.timed('CE_demand'):
                y,gradient,info=ce_cotangent_target(self.g,self.condition()['target'])
            self.demand_cache=(key,y,gradient,info)
            atomic_json(self.out/'demands'/f'a{key[0]}_{key[1]}.json',info)
        return self.demand_cache[1:]

    def advance(self,session,key):
        cg=self.store.cg(self.state,key)
        for _ in range(self.steps):
            if self.stop:raise Paused()
            with self.timed('output_dual_linear_iteration'):
                info,cg=session.advance(cg,1,1e-5)
            self.state['last_solver']=dict(key=key,iterations=int(cg['iterations']),solver_info=int(info))
            self.save(cg,key)
            if info==0:break
            if info<0:raise ArithmeticError('Full-space positive solver broke down; no structure conclusion')
        return cg

    def select(self):
        index=choose_condition(self.state['conditions'],self.state['turn'])
        c=self.state['conditions'][index]
        self.state.update(condition_index=index,condition=c['id'],stage='REFERENCE',condition_started=time.time(),condition_compute_seconds=0.)
        queue=[]
        for graph in self.state['graphs'].values():
            if graph['condition']==c['id']:
                queue.extend(dict(graph=graph['id'],basis=i) for i in range(len(graph['bases'])))
        if queue:
            # Existing functional modes get the first opportunity to control the
            # real model. Rebuild T at the new anchor, without another full solve.
            self.state.update(stage='BUILD',build_queue=queue,build_index=0,reuse_attempt=True,
                last_quality=dict(reference_confidence=1.,reuse_existing_graph=True,
                    old_anchor_reference_error_not_reinterpreted_as_current_error=True))
        else:self.state['reuse_attempt']=False
        self.save()

    def reference(self):
        target,weights,rho=self.settings();y,gradient,info=self.demand()
        all_weights=dict(weights);all_weights[target]=1.
        desired={t:np.zeros(self.g.output_dimensions[t]) for t in self.g.tasks};desired[target]=self.g.tensor_to_numpy(y)
        session=OutputDualWLS(self.g,all_weights,rho,desired,self.diagonals)
        key=f'reference_a{self.state["anchor_serial"]}_{self.state["condition"]}'
        cg=self.advance(session,key)
        with self.timed('reference_observation_reused_for_structure'):
            motion=session.materialize(cg);cert=session.residual_certificate(cg)
        norm=vnorm(motion);confidence=norm/(norm+cert['parameter_error_bound']) if norm else 0.
        candidate=CandidateMotion(key,str(self.state['anchor_serial']),cert['primal_normal_residual_norm'],cert['parameter_error_bound'],cert)
        self.state['candidate_motion']=dict(type='CandidateMotion',**asdict(candidate),
            full_parameter_dimension=self.g.dimension,representation_solver_key=key,
            functional_mode_claim=False)
        self.state['last_quality']=dict(cert,reference_confidence=confidence,conditional_optimum_claim=False)
        # Driver ordering is post-full-space representation, not a solver support restriction.
        slices=[];offset=0
        for index,(name,size) in enumerate(zip(self.g.c.keys,self.g.c.sizes)):
            segment=slice(offset,offset+size);offset+=size
            energy=float(motion[segment].square().sum());gid=f'{self.state["condition"]}_tensor{index}'
            old=self.state['graphs'].get(gid,{})
            age=self.state['turn']-old.get('last_turn',-1)+1
            score=age*np.sqrt(energy)/max(1,old.get('visits',0))
            slices.append((score,gid,name,segment,energy))
        _,gid,name,segment,energy=max(slices,key=lambda row:row[0])
        if energy==0:raise ArithmeticError('Zero reference motion; no random replacement')
        graph=self.state['graphs'].setdefault(gid,dict(id=gid,condition=self.state['condition'],driver_name=name,
            start=segment.start,stop=segment.stop,bases=[],observations=[],visits=0))
        graph['last_turn']=self.state['turn'];graph['visits']+=1
        raw=self.g.tensor_to_numpy(motion[segment]);unit,rawnorm=canonical(raw)
        columns=[np.load(self.out/b['driver_file']) for b in graph['bases']]
        independent,rank=independent_column(columns,unit)
        observation=dict(source=key,iterations=int(cg['iterations']),rank=rank,quality=cert,
            reference_confidence=confidence,raw_driver_norm=rawnorm,independent=bool(independent))
        graph['observations'].append(observation)
        if independent:
            residual=unit.copy()
            if columns:
                Q=np.column_stack(columns)
                for _ in range(2):residual-=Q@np.linalg.lstsq(Q.T@Q,Q.T@residual,rcond=None)[0]
            residual,n=canonical(residual)
            folder=self.out/'drivers'/gid;folder.mkdir(parents=True,exist_ok=True)
            file=folder/f'{len(columns):08d}.npy';np.save(file,residual)
            graph['bases'].append(dict(driver_file=str(file.relative_to(self.out)),
                source_anchor=self.state['anchor_serial'],source=key,reference_confidence=confidence,
                numerical_residual_amplification=1/max(n,np.finfo(float).tiny)))
        self.state['current_graph']=gid
        queue=[]
        for g in self.state['graphs'].values():
            if g['condition']==self.state['condition']:
                for i in range(len(g['bases'])):queue.append(dict(graph=g['id'],basis=i))
        self.state.update(build_queue=queue,build_index=0,stage='BUILD')
        self.condition()['uncertainty']=cert['parameter_error_bound']/max(norm,np.finfo(float).tiny)
        self.condition()['information']=rank['relative_residual']*confidence
        del motion
        self.save()

    def asset_key(self,entry):return f'a{self.state["anchor_serial"]}_{entry["graph"]}_b{entry["basis"]}'

    def build(self,entry):
        aid=self.asset_key(entry)
        if aid in self.state['assets']:return aid
        graph=self.state['graphs'][entry['graph']];basis=graph['bases'][entry['basis']]
        target,weights,rho=self.settings();segment=slice(graph['start'],graph['stop'])
        local=np.load(self.out/basis['driver_file']);driver=torch.zeros_like(self.g.point)
        driver[segment]=self.g.numpy_to_tensor(local)
        operator=ParameterSupportSchur(self.g,segment,weights,rho,diagonals=self.diagonals)
        folder=self.out/'graph_actions'/aid;folder.mkdir(parents=True,exist_ok=True)
        desired_file=folder/'protected_driver_response.npz'
        if desired_file.exists():
            with np.load(desired_file) as z:desired={t:z[t].copy() for t in z.files}
            session=OutputDualWLS(self.g,weights,rho,desired,self.diagonals,segment)
        else:
            with self.timed('driver_response_once'):
                session=operator.session(driver)
            offset=0;desired={}
            for t,n in zip(session.tasks,session.lengths):
                desired[t]=session.d[offset:offset+n]/np.sqrt(weights[t]);offset+=n
            np.savez(desired_file,**desired)
        cg=self.advance(session,'compensation_'+aid)
        with self.timed('protected_graph_lift_and_certificate'):
            motion,certificate=operator.lift(driver,session,cg)
            responses={t:self.g.tensor_to_numpy(self.g.jvp(t,motion)) for t in self.g.tasks}
        temp=folder/'motion.tmp';torch.save(dict(motion=motion.detach().cpu()),temp);temp.replace(folder/'motion.pt')
        np.savez(folder/'responses.npz',**responses)
        meta=dict(id=aid,graph=entry['graph'],basis=entry['basis'],anchor_serial=self.state['anchor_serial'],
            driver_name=graph['driver_name'],driver_start=segment.start,driver_stop=segment.stop,
            driver_file=basis['driver_file'],motion_file=str((folder/'motion.pt').relative_to(self.out)),
            response_file=str((folder/'responses.npz').relative_to(self.out)),certificate=certificate,
            reference_confidence=basis['reference_confidence']/max(1.,basis['numerical_residual_amplification']),solver_iterations=int(cg['iterations']),
            T_operator='-(F_P,NN+rho I)^-1 F_P,NM, full complement; matrix-free action',
            sampled_graph_column=True,entire_graph_explored_claim=False)
        effective=sum(weights[t]*float(responses[t]@responses[t]) for t in weights)+rho*float(motion.square().sum())
        uncompensated=sum(weights[t]*float(desired[t]@desired[t]) for t in weights)+rho*float(driver.square().sum())
        meta['parameter_support_schur_energy']=dict(lifted_cost=effective,uncompensated_driver_cost=uncompensated,
            compensation_cost_reduction=uncompensated-effective,
            approximate_T_stationarity_bound=certificate['compensation_stationarity_norm'])
        atomic_json(folder/'MODE_SOURCE.json',meta);self.state['assets'][aid]=meta
        self.save();return aid

    def build_phase(self):
        if self.state['build_index']<len(self.state['build_queue']):
            self.build(self.state['build_queue'][self.state['build_index']]);self.state['build_index']+=1;self.save()
        else:self.state['stage']='CONTROL';self.save()

    def control(self):
        target,weights,rho=self.settings();y,gradient,info=self.demand()
        assets=[self.state['assets'][self.asset_key(e)] for e in self.state['build_queue']]
        vectors=[self.cache.get(self.out/a['motion_file']) for a in assets]
        G=np.array([[float(torch.dot(a,b)) for b in vectors] for a in vectors])
        Y={t:[] for t in self.g.tasks}
        for a in assets:
            with np.load(self.out/a['response_file']) as z:
                for t in self.g.tasks:Y[t].append(z[t].copy())
        Y={t:np.column_stack(columns) for t,columns in Y.items()}
        slopes=np.array([float(torch.dot(gradient,v)) for v in vectors])
        errors=np.array([a['certificate']['parameter_error_bound'] for a in assets])
        reference_confidence=self.state['last_quality']['reference_confidence']
        with self.timed('selectivity_constructs_actual_control_modes'):
            modes,provenance=graphwise_modes(Y,G,target,weights,rho,slopes,errors,[a['graph'] for a in assets],reference_confidence,[a['reference_confidence'] for a in assets])
            result=actuate(Y,G,target,self.g.tensor_to_numpy(y),weights,rho,modes,1.,reference_confidence)
        motion=torch.zeros_like(self.g.point)
        for v,c in zip(vectors,result['applied_base_coefficients']):motion+=float(c)*v
        actual_slope=float(torch.dot(gradient,motion))
        mode_report=dict(graph_actions=[a['id'] for a in assets],spectral_axes=modes.axes,
            eigenvalues=modes.eigenvalues,CE_slopes=modes.CE_slopes,confidence=modes.confidence,
            parameter_error_bounds=modes.parameter_error_bounds,mode_graph_provenance=provenance,
            control=result,actual_CE_slope=actual_slope,anchor_serial=self.state['anchor_serial'],
            functional_mode_definition='Each b_j=L_graph(j) u_j; mode amplitudes may be coupled by WLS',
            conditional_optimality_claim=False)
        path=self.out/'controls'/f'a{self.state["anchor_serial"]}_{self.state["condition"]}.json'
        atomic_json(path,mode_report)
        with self.timed('real_model_finite_control'):
            point,finite=line_search(self.g,motion,y,target,weights,rho,1.,actual_slope)
        atomic_json(path.with_name(path.stem+'_FINITE.json'),finite)
        if point is None:
            self.state.setdefault('control_failures',[]).append(dict(condition=self.state['condition'],anchor=self.state['anchor_serial'],finite=finite))
            # A failed existing-mode control returns to the complete-space
            # discovery problem. The observations and graph bases are retained.
            if self.state.get('reuse_attempt'):
                self.state['reuse_attempt']=False;self.state['stage']='REFERENCE';self.save()
            else:self.finish_condition()
            return
        old_serial=self.state['anchor_serial'];new=self.store.anchor_file(point,old_serial+1,
            dict(accepted_mode_control=str(path.relative_to(self.out)),source_anchor=old_serial))
        accepted_alpha=float(finite['alpha'])
        self.state['updates'].append(dict(old_anchor=old_serial,new_anchor=new,
            condition=self.state['condition'],mode_control=str(path.relative_to(self.out)),finite=finite,
            motion_norm=accepted_alpha*vnorm(motion),line_search_alpha=accepted_alpha,
            applied_amplitudes=accepted_alpha*result['applied_spectral_coefficients'],
            mode_count=len(modes.eigenvalues),cache_loads=self.cache.loads,cache_hits=self.cache.hits))
        # Commit anchor reference before rebuilding; recovery never reapplies motion.
        self.state.update(anchor=new,anchor_serial=old_serial+1,stage='REANCHOR',
            reanchor_entry=self.state['build_queue'][-1],previous_action=self.asset_key(self.state['build_queue'][-1]))
        self.save();self.cache.clear();self.demand_cache=None
        with self.timed('new_anchor_model_and_geometry'):
            self.g,self.identity=load_eight_geometry(self.out/'runtime',self.protocol['inventory'],self.out/new['file'])
        self.state['stage']='REBUILD';self.save()

    def rebuild(self):
        entry=self.state['reanchor_entry'];old=self.state['assets'][self.state['previous_action']]
        new_id=self.build(entry);new=self.state['assets'][new_id]
        before=self.cache.get(self.out/old['motion_file']);after=self.cache.get(self.out/new['motion_file'])
        event=dict(condition=self.state['condition'],old_action=old['id'],new_action=new_id,
            real_anchor_serial=self.state['anchor_serial'],lift_change_norm=vnorm(after-before),
            new_compensation_stationarity=new['certificate']['compensation_stationarity_norm'],
            operator_recomputed_at_new_anchor=True,continuous_ODE_claim=False)
        self.state.setdefault('geometry_updates',[]).append(event)
        atomic_json(self.out/'LAST_GEOMETRY_UPDATE.json',event);self.finish_condition()

    def finish_condition(self):
        c=self.condition();elapsed=self.state.get('condition_compute_seconds',0.)
        c.update(last_turn=self.state['turn'],visits=c['visits']+1,seconds=elapsed)
        self.state['turn']+=1;self.state['stage']='SELECT';self.save()

    def run(self):
        try:
            while not self.stop:
                stage=self.state['stage']
                if stage=='SELECT':self.select()
                elif stage=='REFERENCE':self.reference()
                elif stage=='BUILD':self.build_phase()
                elif stage=='CONTROL':self.control()
                elif stage in ('REANCHOR','REBUILD'):self.state['stage']='REBUILD';self.rebuild()
                else:raise ValueError('Unknown stage '+stage)
            raise Paused()
        except Paused:
            self.state['run_state']='PAUSED';self.save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--work-quantum',type=int,default=4)
    args=parser.parse_args()
    try:Engine(args.output,args.source,args.work_quantum).run()
    except Exception:
        import traceback
        atomic_json(args.output/'FAILURE.json',dict(traceback=traceback.format_exc(),resume='CURRENT.json'))
        raise
