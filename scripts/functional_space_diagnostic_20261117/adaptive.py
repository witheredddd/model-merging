"""Interleaved functional-space learning with evidence-specific refinement.

Old numerical baselines are optional historical evidence. No relative-error
budget, tensor support, number of modes, or complete class catalogue blocks work.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from bootstrap import CORE,LEGACY,R3,SPACE
from runner import Runner
from io_state import atomic_json,digest
from operators import vnorm
from space_worker import ObservedOutputWLS
from diagnostics import compare,measurement,projection
from functional_update import functional_signal,refinement_question
from streaming import DiskAlgebra


class AdaptiveRunner(Runner):
    def __init__(self,args):
        super().__init__(args)
        self.disk=DiskAlgebra(self.out,args.block_elements)
        if 'adaptive' not in self.state:self.upgrade_state()
        self.fingerprint_dependencies()
        self.state['stage']='ADAPTIVE';self.save()

    def upgrade_state(self):
        bank=[self.disk.convert(e) for e in self.state['bank']]
        old_cursor=min(self.state.get('condition_index',0),2*len(self.g.tasks))
        self.state['adaptive']=dict(turn=0,role_cursor=0,column_serial=0,position_states={},jobs={},
            frozen_bank=list(bank),accumulated_bank=list(bank),shared_sources={e['id']:e['condition'] for e in bank},
            old_baseline_is_not_a_prerequisite=True,relative_error_budget_is_not_admission=True)
        self.state['bank']=bank
        ps=self.position_state(self.state['position']);ps['bank']=bank;ps['global_cursor']=old_cursor
        ps['seen_sources']=[e['id'] for e in bank]
        self.state['schema']='adaptive-functional-space-2'
        self.save()

    def fingerprint_dependencies(self):
        hashes=self.state['source_hashes']
        files=set(p.resolve() for folder in (CORE,LEGACY,R3,SPACE,Path(__file__).parent) for p in folder.glob('*.py'))
        project=Path('/home/shangxueyuan/merging/tsv')
        for module in list(sys.modules.values()):
            path=getattr(module,'__file__',None)
            if path:
                path=Path(path).resolve()
                if path.suffix=='.py' and path.is_relative_to(project) and path.is_file():files.add(path)
        for p in files:
            current=digest(p)
            if str(p) in hashes and hashes[str(p)]!=current:raise ValueError('Imported dependency changed: '+str(p))
            hashes[str(p)]=current
        self.state['dependency_scope']='CORE/LEGACY/R3/SPACE and imported project Python modules; third-party versions recorded separately'
        import scipy
        self.state['runtime_versions']=dict(torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__)

    def position_state(self,index):
        states=self.state['adaptive']['position_states']
        return states.setdefault(str(index),dict(bank=[],global_cursor=0,class_cursor=2*len(self.g.tasks),
            validation_cursor=0,seen_sources=[],visit=0))

    def save(self,cg=None,key=None):
        if 'adaptive' in self.state:
            self.position_state(self.state['position'])['bank']=list(self.state['bank'])
        super().save(cg,key)
        if 'adaptive' in self.state:
            a=self.state['adaptive'];ps=self.position_state(self.state['position'])
            atomic_json(self.out/'ADAPTIVE_STATUS.json',dict(position=self.state['position'],
                actual_training_step=self.positions.rows[self.state['position']]['step'],turn=a['turn'],
                role=self.state.get('current_role'),condition=self.state.get('current_condition'),
                active_columns=len(self.state['bank']),frozen_columns=len(a['frozen_bank']),
                historical_columns=len(a['accumulated_bank']),global_conditions_observed=ps['global_cursor'],
                total_conditions=len(self.state['conditions']),last_functional_update=self.state.get('last_functional_update'),
                single_condition_cannot_block=True,old_baseline_is_not_a_prerequisite=True))

    def load_motion(self,ref,device=None):
        if Path(ref['motion_file']).suffix!='.npy':return super().load_motion(ref,device)
        # One full vector may be transferred for a JVP; the bank is never stacked.
        array=np.load(self.out/ref['motion_file'],mmap_mode='r')
        return torch.from_numpy(np.array(array)).to(device or self.g.device)

    def bank_data(self):
        bank=self.state['bank'];label=getattr(self,'bank_tag',f'active_p{self.state["position"]}')
        with self.timed('stream_incremental_gram'):
            G,E=self.disk.gram(bank,label)
        responses={t:[] for t in self.g.tasks}
        for entry in bank:
            file=self.response_path(entry)
            if not file.exists():
                with self.timed('basis_task_JVPs'):
                    v=self.load_motion(entry);ys={t:self.g.tensor_to_numpy(self.g.jvp(t,v)) for t in self.g.tasks}
                file.parent.mkdir(exist_ok=True);np.savez(file,**ys)
            with np.load(file) as z:
                for t in self.g.tasks:responses[t].append(z[t].copy())
        Y={t:np.column_stack(rows) if rows else np.empty((self.g.output_dimensions[t],0)) for t,rows in responses.items()}
        return bank,Y,G

    def response_path(self,entry):
        return self.out/'bank_responses'/f'p{self.state["position"]}_probe{self.g.probe}_{entry["id"]}.npz'

    @contextmanager
    def using_bank(self,bank,tag):
        old=self.state['bank'];oldtag=getattr(self,'bank_tag',None)
        self.state['bank']=bank;self.bank_tag=tag
        try:yield
        finally:
            self.state['bank']=old
            if oldtag is None:del self.bank_tag
            else:self.bank_tag=oldtag

    def evaluate(self,condition,ref,label):
        bank,Y,G=self.bank_data();y=self.demand(condition);weights,rho=self.settings(condition['target'])
        v=self.disk.open(ref);norm2=self.disk.dot(v,v)[0]
        with np.load(self.out/ref['response_file']) as z:full={t:z[t].copy() for t in z.files}
        report=compare(Y,G,condition['target'],y,weights,rho,full,norm2,ref['certificate'])
        rhs=np.array([self.disk.dot(self.disk.open(e),v)[0] for e in bank])
        report['Euclidean_projection']=projection(G,rhs,norm2)
        report.update(position=self.state['position'],probe=self.g.probe,condition=condition,
            basis_ids=[e['id'] for e in bank],reference_is_finite_candidate=True,
            fixed_percentage_numerical_admission=False)
        atomic_json(self.out/'adaptive_evaluations'/(label+'.json'),report)
        return report

    def observe_batch(self,condition,key,refining=False):
        target=condition['target'];weights,rho=self.settings(target);y=self.demand(condition)
        desired={t:np.zeros(self.g.output_dimensions[t]) for t in weights};desired[target]=y
        session=ObservedOutputWLS(self.g,dict(weights,**{target:1.}),rho,desired,self.diagonals)
        folder=self.out/'solves'/key;latest=folder/'LATEST.json'
        if latest.exists() and not refining:return json.loads(latest.read_text())
        cg=self.store.cg(self.state,key)
        for _ in range(self.args.quantum):
            if self.stop:return None
            with self.timed('CG_iteration'):
                info,cg=session.advance(cg,1,0.)
            self.state['last_solver']=dict(key=key,iterations=int(cg['iterations']),solver_info=int(info))
            self.save(cg,key)
            if info<0:raise ArithmeticError('Nonpositive curvature in '+key)
            if info==0:break
        if self.stop:return None
        with self.timed('fresh_reference_and_certificate'):
            v=session.materialize(cg);cert=session.residual_certificate(cg)
        folder.mkdir(exist_ok=True,parents=True);file=folder/f'iteration{int(cg["iterations"]):08d}.pt'
        if not file.exists():
            temp=file.with_suffix('.tmp');torch.save(dict(motion=v.cpu()),temp);temp.replace(file)
        np.savez(file.with_suffix('.npz'),**session.observed_responses)
        previous=json.loads(latest.read_text()) if latest.exists() else None
        floor=bool(previous and previous['iterations']==int(cg['iterations']))
        ref=dict(key=key,iterations=int(cg['iterations']),certificate=cert,parameter_norm=vnorm(v),
            motion_file=str(file.relative_to(self.out)),response_file=str(file.with_suffix('.npz').relative_to(self.out)),
            numerical_floor=floor,ready_is_not_required_to_observe_or_learn=True,
            functional_measurement=measurement(session.observed_responses,float(v@v),target,y,weights,rho))
        atomic_json(file.with_suffix('.json'),ref);atomic_json(latest,ref)
        self.state.update(last_quality=cert,last_budget=dict(policy='Evidence-specific refinement; no unified relative-error gate'))
        self.save();return ref

    def functional_extension(self,condition,ref,job):
        recorded=self.state.get('current_adaptive_job',{}).get('extension_done')
        if recorded:return json.loads((self.out/recorded).read_text())
        bank,Y,G=self.bank_data();v=self.disk.open(ref)
        label=f'candidate_p{self.state["position"]}_t{self.state["adaptive"]["turn"]}'
        with self.timed('stream_full_parameter_innovation'):
            innovation=self.disk.innovation(bank,v,Path('adaptive_candidates')/(label+'.npy'),
                getattr(self,'bank_tag',f'active_p{self.state["position"]}'))
        if not innovation['numerically_nonzero']:
            result=dict(added=False,reason='No machine-resolved missing parameter component',innovation=innovation,
                question=dict(refinement_priority=1.,missing_control_resolved=False,source_innovation_resolved=False))
            file=Path('functional_updates')/(label+'.json');atomic_json(self.out/file,result)
            self.state['current_adaptive_job']['extension_done']=str(file);self.save();return result
        entry=dict(id=label,motion_file=innovation['motion_file'],motion_sha256=innovation['motion_sha256'])
        with self.timed('candidate_functional_JVPs'):
            q=self.load_motion(entry);newY={t:self.g.tensor_to_numpy(self.g.jvp(t,q)) for t in self.g.tasks}
        cross=np.array([self.disk.dot(self.disk.open(e),self.disk.open(entry))[0] for e in bank])
        norm2=self.disk.dot(self.disk.open(entry),self.disk.open(entry))[0]
        target=condition['target'];weights,rho=self.settings(target);y=self.demand(condition)
        signal=functional_signal(Y,G,target,y,weights,rho,newY,cross,norm2)
        question=refinement_question(signal,ref['certificate'],innovation['norm'])
        confidence=(max(0.,signal['joint_objective_improvement'])/(max(0.,signal['joint_objective_improvement'])+
            ref['certificate']['objective_gap_upper_bound']+np.finfo(float).tiny))*innovation['norm']/(innovation['norm']+
            ref['certificate']['parameter_error_bound']+np.finfo(float).tiny)
        added=signal['functionally_resolved']
        if added:
            a=self.state['adaptive'];entry.update(kind='functionally_tested_full_residual',condition=condition,
                source_position=self.state['position'],source_reference=ref,job=job,
                coordinate_error_bound=ref['certificate']['parameter_error_bound']/innovation['norm'],
                confidence=confidence,provisional=not(question['missing_control_resolved'] and question['source_innovation_resolved']),
                is_not_an_asserted_functional_group=True)
            self.state['bank'].append(entry);a['accumulated_bank'].append(entry);a['shared_sources'][entry['id']]=condition
            self.position_state(self.state['position'])['seen_sources'].append(entry['id'])
            file=self.response_path(entry);file.parent.mkdir(exist_ok=True);np.savez(file,**newY)
        result=dict(added=bool(added),innovation=innovation,functional_signal=signal,question=question,
            source_confidence=confidence,independent_numerical_rank_is_not_sufficient_for_birth=True)
        self.state['last_functional_update']=dict(added=bool(added),gain=signal['joint_objective_improvement'],
            numerical_roundoff=signal['numerical_comparison_roundoff'],source_confidence=confidence)
        file=Path('functional_updates')/(label+'.json');atomic_json(self.out/file,result)
        self.state['current_adaptive_job']['extension_done']=str(file)
        self.save();return result

    def choose_job(self):
        a=self.state['adaptive'];ps=self.position_state(self.state['position']);tasks=len(self.g.tasks)
        # Workstream fairness is a compute schedule, never a dimension rule.
        roles=('learn','relocate','validate','refine','coupling')
        for _ in roles:
            role=roles[a['role_cursor']%len(roles)];a['role_cursor']+=1
            if role=='learn':
                if ps['global_cursor']<2*tasks:
                    i=ps['global_cursor'];ps['global_cursor']+=1
                else:
                    i=ps['class_cursor']
                    if i>=len(self.state['conditions']):continue
                    ps['class_cursor']+=1
                c=self.state['conditions'][i]
                return dict(role=role,condition=c,key=f'study_p{self.state["position"]}_{c["id"]}',refining=False)
            if role=='relocate':
                unseen=[sid for sid in a['shared_sources'] if sid not in ps['seen_sources']]
                if not unseen:continue
                sid=unseen[0];ps['seen_sources'].append(sid);c=a['shared_sources'][sid]
                # Crucially regenerate at the NEW anchor. Never load an old y0.
                return dict(role=role,condition=c,key=f'regenerated_p{self.state["position"]}_{c["id"]}',
                    source=sid,refining=False,geometry_policy='Same functional demand generator at current position')
            if role=='validate':
                if ps['validation_cursor']>=len(self.state['conditions']):continue
                ci=ps['validation_cursor'];ps['validation_cursor']+=1
                c=dict(self.state['conditions'][ci]);c['id']+='_independent_probe'
                return dict(role=role,condition=c,key=f'independent_p{self.state["position"]}_{c["id"]}',refining=False)
            if role=='refine':
                jobs=[j for j in a['jobs'].values() if j['position']==self.state['position'] and j.get('needs_refinement') and not j.get('numerical_floor')]
                if not jobs:continue
                j=max(jobs,key=lambda j:(a['turn']-j.get('last_turn',-1)+1)*(1+j.get('priority',1.))/max(1,j.get('seconds',1.)))
                return dict(j['job'],role='refine',refining=True)
            if role=='coupling' and ps.get('coupling'):
                return dict(role='coupling',condition=ps['coupling']['condition'],key='regional_coupling')
        return None

    def advance_visit(self):
        a=self.state['adaptive'];ps=self.position_state(self.state['position']);tasks=len(self.g.tasks)
        # Initial global family at every position precedes the large class sweep.
        if ps['global_cursor']<2*tasks:return
        if ps.get('visit_start_class') is None:ps['visit_start_class']=ps['class_cursor']
        initial=not ps.get('initial_global_visit_done')
        if not initial and ps['class_cursor']<len(self.state['conditions']) and ps['class_cursor']-ps['visit_start_class']<tasks:return
        ps['initial_global_visit_done']=True;ps['visit']+=1;ps['visit_start_class']=None
        ps['bank']=list(self.state['bank']);previous=self.state['position']
        index=(previous+1)%len(self.positions.rows)
        self.change_position(index);new=self.position_state(index)
        self.state['bank']=list(new['bank'])
        new['evolution_history']=new.get('evolution_history',[])+[dict(from_position=previous,turn=a['turn'])]
        self.save()

    def start_coupling(self,condition,report):
        ps=self.position_state(self.state['position'])
        if ps.get('coupling') or ps.get('completed_coupling'):return
        z=np.asarray(report['undamped_coefficients'])
        if not len(z) or not np.any(z):return
        path=Path('regional_controls')/f'p{self.state["position"]}.npy'
        with self.timed('stream_control_for_regional_diagnostic'):
            self.disk.write_combination(self.state['bank'],z,path)
        ps['coupling']=dict(motion_file=str(path),condition=condition,region_index=0,
            output_folder=f'regional_coupling/p{self.state["position"]}',probe=0)
        self.save()

    def coupling_step(self):
        from region_coupling import observe_region
        ps=self.position_state(self.state['position']);item=ps['coupling']
        with self.timed('parameter_region_functional_diagnostic'):
            done=observe_region(self.g,self.disk,self.out,item,self.demand(item['condition']),self.settings(item['condition']['target']))
        if done:ps['completed_coupling']=item;del ps['coupling']
        self.save()

    def step(self):
        job=self.state.get('current_adaptive_job')
        if job is None:
            self.advance_visit();job=self.choose_job()
            if job is None:
                for _ in range(len(self.positions.rows)-1):
                    index=(self.state['position']+1)%len(self.positions.rows)
                    self.change_position(index);self.state['bank']=list(self.position_state(index)['bank'])
                    job=self.choose_job()
                    if job is not None:break
                if job is None:self.state['run_state']='COVERAGE_EXHAUSTED';self.stop=True;return
            self.state['current_adaptive_job']=job
        role=job['role'];condition=job['condition'];self.g.probe=1 if role=='validate' else 0
        self.state.update(current_role=role,current_condition=condition['id']);self.save()
        started=time.monotonic()
        if role=='coupling':self.coupling_step()
        else:
            try:ref=self.observe_batch(condition,job['key'],job['refining'])
            except ArithmeticError as error:
                # A numerical issue affects this condition, not the other tasks.
                self.state['adaptive']['jobs'][job['key']]=dict(job=job,position=self.state['position'],
                    numerical_floor=True,error=str(error),needs_refinement=False)
                ref=None
            if self.stop:return
            if ref:
                turn=self.state['adaptive']['turn'];label=f'p{self.state["position"]}_t{turn}_{role}'
                before=self.evaluate(condition,ref,label+'_before')
                if role=='validate':
                    for name,bank in (('frozen',self.state['adaptive']['frozen_bank']),('accumulated',self.state['adaptive']['accumulated_bank'])):
                        with self.using_bank(bank,f'{name}_p{self.state["position"]}'):
                            self.evaluate(condition,ref,label+'_'+name)
                    result=None
                else:
                    result=self.functional_extension(condition,ref,job['key'])
                    after=self.evaluate(condition,ref,label+'_after');self.start_coupling(condition,after)
                    self.state['observations'].append(dict(position=self.state['position'],condition=condition,
                        role=role,before='adaptive_evaluations/'+label+'_before.json',after='adaptive_evaluations/'+label+'_after.json'))
                question=result['question'] if result else dict(refinement_priority=0.,missing_control_resolved=True,source_innovation_resolved=True)
                self.state['adaptive']['jobs'][job['key']]=dict(job=job,position=self.state['position'],
                    needs_refinement=not(question['missing_control_resolved'] and question['source_innovation_resolved']),
                    priority=question['refinement_priority'],numerical_floor=ref.get('numerical_floor',False),
                    last_turn=turn,seconds=time.monotonic()-started)
                if job.get('source'):
                    atomic_json(self.out/'regeneration'/f'p{self.state["position"]}_{job["source"]}.json',dict(
                        source=job['source'],condition=condition,reference=ref,geometry_policy=job['geometry_policy'],
                        functional_demand_regenerated_at_new_anchor=True,old_tangent_vector_reused=False))
                if role!='validate':
                    ps=self.position_state(self.state['position']);sources=self.state['adaptive']['shared_sources']
                    for sid,c in sources.items():
                        if c['id']==condition['id'] and sid not in ps['seen_sources']:ps['seen_sources'].append(sid)
        self.g.probe=0;self.state['adaptive']['turn']+=1
        self.state.pop('current_adaptive_job',None)
        self.state['last_job_wall_seconds']=time.monotonic()-started
        atomic_json(self.out/'STREAM_IO.json',self.disk.stats);self.save()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('old','source','trace','coordinates','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--quantum',type=int,default=4);p.add_argument('--relative-error-budget',type=float,default=.001)
    p.add_argument('--gamma',type=float,default=.1);p.add_argument('--block-elements',type=int,default=262144)
    args=p.parse_args()
    try:AdaptiveRunner(args).run()
    except Exception:
        import traceback
        atomic_json(args.output/'FAILURE.json',dict(traceback=traceback.format_exc()));raise
