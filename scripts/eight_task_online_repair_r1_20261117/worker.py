"""Resumable fair conditional-demand engine. No wall-clock stop or claimed block learner."""
import argparse,json,signal,time
from pathlib import Path
import numpy as np
from bootstrap import CORE,LEGACY
from checkpoint import Store,WriterLock,atomic_json,digest
from scheduler import take,finish
from schur_session import SchurSession
from task_runtime import load_eight_geometry
from geometry import EuclideanMetric,FullFunctionalGeometry
from conditional_controls import ConditionalControls,plain
import couplings
from finite_monitor import evaluate
from provisional import ProvisionalControls,focused_take
from coarse import correct


def target_response(g,spec):
    task=spec['task']
    if spec['source']=='CE':
        y,_,_,info=g.ce_unit_target(task);return y,info
    if spec['source']=='class_CE':
        grad=g.class_ce_gradient(task,spec['label']);m=float(grad.norm())
        return (g.jvp(task,-grad/m) if m else None),dict(defined=bool(m),label=spec['label'])
    expert=g.c.experts[g.tasks.index(task)].to(device=g.device,dtype=g.point.dtype)
    motion=expert-g.point;m=float(motion.norm())
    return (g.jvp(task,motion/m) if m else None),dict(defined=bool(m),source='Full expert-motion current-anchor JVP; not nonlinear expert Log')


def run(output):
    out=Path(output);store=Store(out)
    with WriterLock(out):
        state=store.load();protocol=json.loads((out/'PROTOCOL.json').read_text())
        if state['protocol_sha256']!=digest(out/'PROTOCOL.json'):raise ValueError('Protocol changed; explicit migration required')
        for name,expected in state['code_hashes'].items():
            if digest(name)!=expected:raise ValueError('Executed source changed; explicit migration required: '+name)
        for name,key in [('runtime/probes.json','probe_sha256'),('SCALES.json','scales_sha256'),('PRECONDITIONER.npz','preconditioner_sha256')]:
            if digest(out/name)!=protocol[key]:raise ValueError('Geometry/numerical cache differs: '+name)
        stop=[False]
        def pause(*_):stop[0]=True
        signal.signal(signal.SIGTERM,pause);signal.signal(signal.SIGINT,pause)
        state['phase']='RESTORING';store.save(state)
        atomic_json(out/'status.json',dict(stage='restoring_model_and_process_state'))
        g,source=load_eight_geometry(out/'runtime',protocol['inventory'],protocol['checkpoint'])
        old=protocol['source']
        for key in ('checkpoint_sha256','inventory_sha256','probe_sha256','parameter_dimension'):
            if source[key]!=old[key]:raise ValueError('Model/data identity differs: '+key)
        # Restore after model/data constructors have consumed any random numbers.
        state=store.load(restore_random_state=True)
        scales=json.loads((out/'SCALES.json').read_text())['scales']
        with np.load(out/'PRECONDITIONER.npz') as d:diagonals={t:d[t].copy() for t in g.tasks}
        scheduler=state['scheduler'];state['phase']='RUNNING'
        bank=ConditionalControls(g,out/'control_space',EuclideanMetric(),state.get('structural_state'))
        state['structural_state']=bank.state;geometries={0:g}
        exploring=ProvisionalControls(g,out/'provisional_space',EuclideanMetric(),state.get('provisional_state'))
        state['provisional_state']=exploring.state
        state['structure_status']='Online conditional control space; not discrete hard blocks'
        store.save(state)
        while not stop[0]:
            dispatch=focused_take(scheduler)
            if dispatch is None:
                state['phase']='INPUT_EXHAUSTED';store.save(state)
                atomic_json(out/'status.json',dict(stage='input_catalog_exhausted',structure_learned=False))
                return
            case_id=dispatch['case_id'];case=scheduler['cases'][case_id]
            probe=case['spec'].get('probe',0)
            if probe not in geometries:geometries[probe]=FullFunctionalGeometry(g.c,g.point,probe,precision='float64')
            current_g=geometries[probe]
            if 'start_iteration' not in dispatch:dispatch['start_iteration']=case['iterations']
            folder=out/'demands'/case_id;folder.mkdir(parents=True,exist_ok=True)
            state['phase']='RUNNING';store.save(state)
            atomic_json(out/'status.json',dict(stage='dispatch',case_id=case_id,family=case['spec']['source'],
                iterations=case['iterations'],control_count=exploring.size,certified_reference_controls=bank.size,
                structure_module='online_provisional_control_space'))
            if (folder/'response.npy').exists():y=g.numpy_to_tensor(np.load(folder/'response.npy'))
            else:
                y,info=target_response(current_g,case['spec'])
                atomic_json(folder/'DEMAND.json',dict(spec=case['spec'],info=info,
                    output_coordinate_id=current_g.output_coordinate_ids[case['spec']['task']],
                    anchor_sha256=source['checkpoint_sha256']))
                if y is None:
                    case['status']='undefined_zero_source';scheduler['inflight']=None;store.save(state);continue
                tmp=folder/'response.tmp'
                with tmp.open('wb') as f:np.save(f,g.tensor_to_numpy(y))
                tmp.replace(folder/'response.npy')
            if float(y.norm())==0:
                case['status']='zero_response';scheduler['inflight']=None;store.save(state);continue
            task=case['spec']['task'];weights={t:1. for t in g.tasks if t!=task}
            rho=case['gamma']*sum(scales[t] for t in weights)
            pd=np.concatenate([1+diagonals[t]/rho for t in weights]);td=diagonals[task]/rho;td=np.where(td>0,td,1.)
            session=SchurSession(current_g,task,weights,rho,y,pd,td)
            cg=store.load_cg(case)
            exploratory_restricted=exploring.solve(task,y,weights,rho,protocol['solver_rtol'],g=current_g)
            atomic_json(folder/'PROVISIONAL_CONTROL.json',plain(dict(exploratory_restricted,
                numerical_status='provisional',conditional_optimality_claim=False)))
            if cg is not None and exploring.size and not dispatch.get('coarse_done',False):
                cg,feedback=correct(session,cg,exploring)
                case['latest_structure_feedback']=plain(feedback)
                state.setdefault('structure_feedback_events',[]).append(dict(case_id=case_id,
                    cg_iteration=int(cg['iterations']),feedback=plain(feedback)))
                atomic_json(folder/'STRUCTURE_FEEDBACK.json',plain(feedback))
                dispatch['coarse_done']=True
                store.save(state,cg,case_id)
            elif not dispatch.get('coarse_done',False):
                dispatch['coarse_done']=True
                store.save(state)
            restricted=bank.solve(task,y,weights,rho,protocol['solver_rtol'],g=current_g)
            atomic_json(folder/f'CONTROL_BEFORE_v{bank.size:08d}.json',plain(restricted))
            if case['spec'].get('role')=='validation' and bank.size and case.get('finite_control_version')!=bank.size:
                atomic_json(folder/f'FINITE_CONTROL_v{bank.size:08d}.json',evaluate(current_g,bank.decode(restricted['z'])))
                case['finite_control_version']=bank.size
                # Validation never expands or selects the control space.
                store.save(state)
            if cg is None and bank.size:
                cg,seed=bank.seed(session,restricted['z'],protocol['solver_rtol'])
                case['control_seed']=plain(seed)
                if cg is not None:
                    case['basis_version_at_fullsolve_start']=bank.size
                    store.save(state,cg,case_id)
            elif cg is None and exploring.size:
                cg,seed=exploring.seed(session,exploratory_restricted['z'],protocol['solver_rtol'])
                case['provisional_initialization']=plain(seed)
                if cg is not None:store.save(state,cg,case_id)
            if state.get('preflight_pending') and cg is not None:
                fresh=session.b-session.apply(cg['x'])
                error=float(np.linalg.norm(fresh-cg['r'])/max(np.linalg.norm(session.b),np.finfo(float).tiny))
                atomic_json(out/'HANDOFF_CHECK.json',dict(case_id=case_id,iterations=int(cg['iterations']),
                    original_vs_restored_recursive_residual_difference=error,
                    requested_numeric_tolerance=protocol['solver_rtol'],
                    source_checkpoint=case['cg_checkpoint'],model_and_probes_hashes_match=True))
                if error>protocol['solver_rtol']:raise ArithmeticError('Restored true residual does not match the legacy CG state')
                state['preflight_pending']=False;store.save(state)
            elif cg is None and state.get('preflight_pending'):
                state['preflight_pending']=False;store.save(state)
            inner=case['inner_rtol'] or protocol['solver_rtol']
            def committed_iteration(x,current):
                case['iterations']=int(current['iterations']);case['status']='provisional'
                case['recursive_response_estimate']=session.response_residual_estimate(current)
                store.save(state,current,case_id)
                atomic_json(out/'status.json',dict(stage='joint_seven_condition_iteration',case_id=case_id,
                    task=task,family=case['spec']['source'],iterations=case['iterations'],
                    numerical_status='provisional',scheduler_dispatches=scheduler['dispatches'],
                    control_count=bank.size,structure_module='online_conditional_control_space'))
                atomic_json(out/'status.json',dict(stage='focused_fullspace_iteration',case_id=case_id,
                    task=task,iterations=case['iterations'],provisional_controls=exploring.size,
                    certified_reference_controls=bank.size,
                    actual_structure_corrections=sum(e['feedback']['applied'] for e in state.get('structure_feedback_events',[]))))
            already=(int(cg['iterations'])-dispatch['start_iteration']) if cg is not None else 0
            remaining=protocol['checkpoint_iterations']-already
            if remaining>0:
                _,info,cg=session.step(cg,inner,remaining,committed_iteration)
            else:
                # Crash after an iteration commit but before dispatch completion:
                # finish that same dispatch, without silently taking another step.
                info=0 if np.sqrt(float(cg['rr']))<=inner*np.linalg.norm(session.b) else 1
            case['visits']+=1
            # Every state is a provisional candidate independently of rtol.
            # It can be exactly materialized from dual weights and the locked model,
            # without accumulating one 658 MiB parameter column per observation.
            atomic_json(folder/'CANDIDATE.json',dict(case_id=case_id,iterations=int(cg['iterations']),
                status='provisional',representation=case['cg_checkpoint'],
                response_estimate=session.response_residual_estimate(cg),
                available_for_exploration=True,conditional_optimum_claim=False,
                parameter_dimension=g.dimension,online_provisional_learning_enabled=True,
                intrinsic_dimension_claim=False))
            audit_due=True # Original KKT audit at every focused checkpoint, no 80-step trigger.
            resolved=False
            if audit_due:
                audit=session.audit(cg)
                certified=all(audit[k] is not None and audit[k]<=protocol['solver_rtol'] for k in
                    ('response_relative_residual','stationarity_relative_residual'))
                case['audit']=audit;case['certified']=certified
                case['status']='certified' if certified else 'provisional'
                atomic_json(folder/'NUMERICAL_AUDIT.json',dict(audit,certified=certified,
                    solve_rtol=protocol['solver_rtol'],iterations=int(cg['iterations'])))
                resolved=certified
                if probe==0 and case['spec'].get('role','training')=='training':
                    middle,_=session.materialize(cg)
                    event=exploring.refresh(middle,session.full_motion_duals(cg),audit,case_id)
                    state['provisional_state']=exploring.state
                    atomic_json(folder/'PROVISIONAL_UPDATE.json',plain(dict(event,
                        numerical_status='provisional',original_KKT=audit,conditional_optimum_claim=False)))
                    # Consume already available intermediate motions from the
                    # other tasks, without starting more independent solves.
                    cursor=state.get('observation_cursor',0)
                    other_task=g.tasks[cursor%len(g.tasks)];state['observation_cursor']=cursor+1
                    observed=[c for c in scheduler['cases'].values() if c['id']!=case_id and
                        c['spec']['task']==other_task and c['iterations']>0 and
                        c['spec'].get('probe',0)==0 and c['spec'].get('role','training')=='training']
                    if observed:
                        other=observed[(cursor//len(g.tasks))%len(observed)]
                        other_y=g.numpy_to_tensor(np.load(out/'demands'/other['id']/'response.npy'))
                        other_weights={t:1. for t in g.tasks if t!=other_task}
                        other_rho=other['gamma']*sum(scales[t] for t in other_weights)
                        opd=np.concatenate([1+diagonals[t]/other_rho for t in other_weights])
                        otd=diagonals[other_task]/other_rho;otd=np.where(otd>0,otd,1.)
                        other_session=SchurSession(g,other_task,other_weights,other_rho,other_y,opd,otd)
                        other_cg=store.load_cg(other);other_v,_=other_session.materialize(other_cg)
                        quality=dict(other.get('audit') or other_session.response_residual_estimate(other_cg),
                            numerical_status='provisional',source_iterations=other['iterations'],
                            original_optimality_not_claimed=True)
                        exploring.refresh(other_v,other_session.full_motion_duals(other_cg),quality,other['id'])
                        state['provisional_state']=exploring.state
                if certified:
                    certificate=dict(audit,certified=True)
                    vstar,_=session.materialize(cg)
                    comparison=bank.compare(restricted,vstar,weights,rho,certificate,g=current_g)
                    atomic_json(folder/f'CONTROL_GAP_v{bank.size:08d}.json',plain(comparison))
                    if case['spec'].get('role','training')=='training':
                        event=bank.observe_certified(vstar,session.full_motion_duals(cg),certificate,
                            dict(case_id=case_id,iterations=int(cg['iterations']),probe=probe))
                        atomic_json(folder/'CONTROL_EXPANSION.json',plain(event))
                        if event['independent_at_machine_precision']:
                            couplings.enqueue(state,event['control_index'],g)
                        after=bank.solve(task,y,weights,rho,protocol['solver_rtol'])
                        atomic_json(folder/f'CONTROL_AFTER_v{bank.size:08d}.json',plain(after))
                    else:
                        atomic_json(folder/'FINITE_FULL_REFERENCE.json',evaluate(current_g,vstar))
                    atomic_json(folder/f'SELECTIVITY_v{bank.size:08d}.json',plain(bank.spectrum(task,weights,rho,g=current_g)))
                if info<0:
                    case['status']='numerical_breakdown';resolved=True
                elif info==0 and not certified:
                    worst=max(audit['response_relative_residual'],audit['stationarity_relative_residual'])
                    case['inner_rtol']=inner*protocol['solver_rtol']/worst
                    if case['inner_rtol']<np.finfo(float).eps:
                        case['status']='working_precision_exhausted';resolved=True
            if resolved:scheduler['inflight']=None
            else:scheduler['inflight']=dict(case_id=case_id,start_iteration=case['iterations'])
            store.save(state,cg,case_id)
            exploring.prune_materialization_cache()
            if couplings.step(state,bank,out):store.save(state)
        state['phase']='PAUSED';store.save(state)
        atomic_json(out/'PAUSED.json',dict(recoverable=True,checkpoint='CURRENT.json',phase='PAUSED',
            reason='External signal; no internal time cutoff',structure_learned=False))
        atomic_json(out/'status.json',dict(stage='paused',recoverable=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true',required=True)
    args=p.parse_args()
    try:run(args.output)
    except Exception:
        import traceback
        atomic_json(args.output/'FAILURE.json',dict(traceback=traceback.format_exc(),recover_from='CURRENT.json',
            no_structural_failure_conclusion=True))
        raise
