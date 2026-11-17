"""Frozen-structure A100 audit. Original production checkpoints are read only."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RUNTIME = HERE.parent/'eight_task_online_repair_r2_20261117'
SOURCE = Path('/media/shangxueyuan/data1/eight_task_online_repair_r2_20261117')
OUT = Path('/media/shangxueyuan/data1/coordination_audit_20261117')
sys.path.insert(0, str(RUNTIME))
from bootstrap import CORE
from checkpoint import Store, atomic_json, digest
from conditional_controls import ConditionalControls, plain
from provisional import ProvisionalControls
from schur_session import SchurSession
from coarse import correct
from geometry import EuclideanMetric, FullFunctionalGeometry
from task_runtime import load_eight_geometry
from finite_monitor import evaluate
from audit_common import Profiler

CASE = 'SUN397_expert_motion_JVP_gamma0.1'


def load_json(path):
    return json.loads(Path(path).read_text())


def load_state(path):
    return load_json(path)


def cg_from(state):
    case = state['scheduler']['cases'][CASE]
    ref = case['cg_checkpoint']
    path = SOURCE/ref['file']
    assert digest(path) == ref['sha256']
    with np.load(path) as z:
        return {k:z[k].copy() for k in z.files}


def relative(a, b):
    return float(np.linalg.norm(a)/max(np.linalg.norm(b), np.finfo(float).tiny))


def prepare():
    OUT.mkdir(exist_ok=True)
    current = load_json(SOURCE/'CURRENT.json')
    state = load_state(SOURCE/current['state'])
    assert state['phase'] == 'PAUSED', 'Producer must commit a graceful pause first'
    found = {}
    for path in sorted((SOURCE/'checkpoints').glob('*/state.json'), reverse=True):
        s = load_state(path)
        n = s['scheduler']['cases'][CASE]['iterations']
        count = len((s.get('provisional_state') or {}).get('modes', []))
        if n in (14, 15, 16) and str(n) not in found:
            found[str(n)] = str(path)
        if count == 29 and 'bank29' not in found:
            found['bank29'] = str(path)
        if all(k in found for k in ('14', '15', '16', 'bank29')):
            break
    initial_generation = load_json(SOURCE/'FEEDBACK_VALIDATION.json')['source_generation']
    found['bank25'] = str(SOURCE/'checkpoints'/f'{initial_generation:012d}'/'state.json')
    found['paused'] = str(SOURCE/current['state'])
    protocol = load_json(SOURCE/'PROTOCOL.json')
    hashes_match = {f:digest(f)==h for f,h in state['code_hashes'].items()}
    assert all(hashes_match.values())
    report = dict(stage='prepared', inputs=found, source_current=current,
                  source_protocol_sha256=digest(SOURCE/'PROTOCOL.json'),
                  runtime_hashes_verified=hashes_match,
                  SUN_structure_feedback_events=[e for e in state.get('structure_feedback_events', []) if e['case_id']==CASE],
                  automatic_structure_correction=protocol['automatic_structure_correction'],
                  production_paused=True, no_structure_learning_in_this_audit=True,
                  numerical_comparison_steps=1,
                  comparison_note='Same SUN397 expert-motion target, gamma .1, seven unit protection weights, identical step14 x/r/p for all three branches. Extra correction cost counted. No optimality claims.',
                  no_structure_means='Full-space CG without learned-subspace correction, retaining the historical CG state, not a zero parameter update.')
    # Frozen provenance, small metadata and executed source, without large model blobs.
    for name,path in found.items():
        atomic_json(OUT/(name+'_state.json'), load_state(path))
    atomic_json(OUT/'PROTOCOL_SOURCE.json', protocol)
    code = OUT/'executed_source';code.mkdir(exist_ok=True)
    for path in list(HERE.glob('*.py'))+list(RUNTIME.glob('*.py')):
        target = code/(('runtime_' if path.parent==RUNTIME else '')+path.name)
        shutil.copy2(path,target)
    atomic_json(code/'SOURCE_HASHES.json', {p.name:digest(p) for p in code.glob('*.py')})
    atomic_json(OUT/'REPORT.json', report)
    return report, protocol


def historical_timing(report):
    rows=[]
    for path in sorted((SOURCE/'checkpoints').glob('*/state.json')):
        if int(path.parent.name)<2975:
            continue
        state=load_state(path)
        rows.append((path.stat().st_mtime, state))
    dispatches=[]
    for i,(stamp,s) in enumerate(rows):
        inflight=s['scheduler'].get('inflight')
        if not inflight or not inflight.get('coarse_done'):
            continue
        cid=inflight['case_id'];case=s['scheduler']['cases'][cid]
        start_iteration=inflight['start_iteration']
        if case['iterations'] != start_iteration:
            continue
        # First committed coarse_done point for this dispatch.
        if i and rows[i-1][1]['scheduler'].get('inflight') == inflight:
            continue
        committed=None;finished=None
        for later,(when,q) in enumerate(rows[i+1:],i+1):
            c=q['scheduler']['cases'][cid]
            if c['iterations']>start_iteration and committed is None:
                committed=(when,q)
            if committed and q['scheduler'].get('inflight') is None:
                finished=(when,q);break
        if committed:
            dispatches.append(dict(case_id=cid,start_iteration=start_iteration,
                iteration_commit_seconds=committed[0]-stamp,
                post_iteration_seconds=(finished[0]-committed[0]) if finished else None,
                includes='Iteration commit includes Schur apply, CG vector arithmetic and checkpoint callback; post includes full KKT, materialize and both provisional refreshes plus persistence. Inferred from immutable checkpoint timestamps.',
                start_generation=s['generation'],iteration_generation=committed[1]['generation'],
                final_generation=finished[1]['generation'] if finished else None))
    report['historical_dispatch_timing']=dispatches


def instrument_bank(bank,prof):
    for name,label in [('vector','vector_materialize_or_read'),('duals','dual_read_and_hash'),
                       ('decode','parameter_decode'),('responses','response_cache_or_JVP'),
                       ('solve','restricted_solve'),('refresh','structure_refresh')]:
        prof.wrap(bank,name,label)


def profile_observe_method(prof):
    """Insert contexts around unchanged arithmetic regions in a diagnostic copy."""
    path=RUNTIME/'conditional_controls.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ConditionalControls')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_observe_motion')
    # Only the two-pass projection loop and Gram construction loop are timed
    # here; vector/cache/transfer children are subtracted in exclusive totals.
    for i,node in enumerate(method.body):
        if isinstance(node,ast.For) and ast.unparse(node.target)=='_':
            method.body[i]=ast.With(items=[ast.withitem(context_expr=ast.parse("prof.measure('orthogonalization')",mode='eval').body)],body=[node])
        elif isinstance(node,ast.For) and ast.unparse(node.target)=='j' and 'range(self.size)' in ast.unparse(node.iter):
            method.body[i]=ast.With(items=[ast.withitem(context_expr=ast.parse("prof.measure('parameter_gram')",mode='eval').body)],body=[node])
    ast.fix_missing_locations(tree)
    namespace=dict(prof=prof,__name__='profiled_conditional_controls')
    exec(compile(tree,str(path)+'[diagnostic contexts]','exec'),namespace)
    return namespace['ConditionalControls']._observe_motion


def run():
    report,protocol=prepare()
    historical_timing(report)
    prof=Profiler(lambda:torch.cuda.synchronize() if torch.cuda.is_initialized() else None)
    def save(stage):
        report['stage']=stage;report['timing']=prof.summary()
        atomic_json(OUT/'REPORT.json',plain(report))
        atomic_json(OUT/'TIMING_EVENTS.json',prof.records)
        print(stage,flush=True)
    save('loading_model')
    with prof.measure('model_and_geometry_load'):
        g,identity=load_eight_geometry(SOURCE/'runtime',protocol['inventory'],protocol['checkpoint'])
    report['geometry_identity']=identity
    for name,label in [('jvp','JVP'),('vjp','VJP'),('ce_gradient','CE_gradient')]:
        prof.wrap(g,name,label)
    weights={t:1. for t in g.tasks if t!='SUN397'}
    scales=load_json(SOURCE/'SCALES.json')['scales']
    rho=.1*sum(scales[t] for t in weights)
    with np.load(SOURCE/'PRECONDITIONER.npz') as z:
        pd=np.concatenate([1+z[t]/rho for t in weights]);td=np.where(z['SUN397']>0,z['SUN397']/rho,1.)
    y=g.numpy_to_tensor(np.load(SOURCE/'demands'/CASE/'response.npy'))
    session=SchurSession(g,'SUN397',weights,rho,y,pd,td)
    prof.wrap(session,'apply','Schur_apply')
    prof.wrap(session,'audit','full_KKT_audit')
    # LinearOperator captures a bound method at construction; update it to the
    # measured wrapper, retaining exactly the same operator arithmetic.
    from scipy.sparse.linalg import LinearOperator
    session.A=LinearOperator((len(session.b),len(session.b)),matvec=session.apply,dtype=np.float64)
    banks={}
    for count in (25,29):
        frozen=load_state(report['inputs']['bank'+str(count)])['provisional_state']
        assert len(frozen['modes'])==count
        # Read-only during comparison: all learning-window responses exist.
        bank=ProvisionalControls(g,SOURCE/'provisional_space',EuclideanMetric(),copy.deepcopy(frozen))
        instrument_bank(bank,prof);banks[count]=bank
    h14=load_state(report['inputs']['14']);cg14=cg_from(h14)
    stored15=cg_from(load_state(report['inputs']['15']))
    report['SUN_replay']={}
    prof.phase='SUN_step14'
    with prof.measure('true_residual'):
        true14=session.b-session.apply(cg14['x'])
    with prof.measure('full_audit'):
        report['SUN_replay']['step14_audit']=session.audit(cg14)
    report['SUN_replay']['step14_recursive_true_relative_difference']=relative(true14-cg14['r'],session.b)
    Ap14=None
    original_apply=session.apply
    # Capture the actual historical search image without an extra operator call.
    def capture(x):
        nonlocal Ap14
        result=original_apply(x)
        Ap14=result.copy()
        return result
    session.A=LinearOperator((len(session.b),len(session.b)),matvec=capture,dtype=np.float64)
    prof.phase='no_structure'
    start=time.perf_counter()
    with prof.measure('CG_one_step'):
        _,info,cg15=session.step(cg14,protocol['solver_rtol'],1,None)
    report['SUN_replay']['pure_CG_seconds']=time.perf_counter()-start
    session.A=LinearOperator((len(session.b),len(session.b)),matvec=session.apply,dtype=np.float64)
    curvature=float(cg14['p']@Ap14);alpha=float(cg14['rr'])/curvature
    beta=float(cg15['rr'])/float(cg14['rr'])
    report['SUN_replay'].update(pure_CG_matches_historical={k:relative(cg15[k]-stored15[k],stored15[k]) for k in ('x','r','p')},
        alpha=alpha,beta=beta,p14_curvature=curvature,
        p15_conjugacy_numerator=float(cg15['p']@Ap14),
        conjugacy_normalized_by_norms=float((cg15['p']@Ap14)/(np.linalg.norm(cg15['p'])*np.linalg.norm(Ap14))),
        direction_recurrence_relative_error=relative(cg15['p']-(cg15['r']+beta*cg14['p']),cg15['p']),
        iterate_recurrence_relative_error=relative(cg15['x']-(cg14['x']+alpha*cg14['p']),cg15['x']),
        actual_step_had_structure_correction=False,
        schur_quadratic_change=float(-.5*alpha*float(cg14['rr'])))
    branches={'no_structure':cg15}
    report['branches']={}
    def audit_branch(name,cg):
        prof.phase=name
        with prof.measure('true_residual'):
            fresh=session.b-session.apply(cg['x'])
        audit=session.audit(cg)
        result=report['branches'].setdefault(name,{})
        result.update(audit=audit,recursive_true_relative_difference=relative(fresh-cg['r'],session.b),
                      full_scaled_schur_relative_residual=relative(fresh,session.b),
                      full_unscaled_schur_residual_norm=float(np.linalg.norm(fresh/session.scale)),
                      recursive_residual_norm=float(np.linalg.norm(cg['r'])),
                      true_residual_norm=float(np.linalg.norm(fresh)),
                      protected_energy_sum=sum(audit['protected_response_energies'].values()))
        np.savez(OUT/(name+'_cg.npz'),**cg)
    audit_branch('no_structure',cg15)
    save('pure_CG_replayed')
    for count,bank in banks.items():
        name='structure'+str(count);prof.phase=name;start=time.perf_counter()
        with prof.measure('restricted_control_problem'):
            restricted=bank.solve('SUN397',y,weights,rho,protocol['solver_rtol'])
        with prof.measure('structure_correction'):
            corrected,event=correct(session,copy.deepcopy(cg14),bank)
        with prof.measure('CG_one_step'):
            _,_,final=session.step(corrected,protocol['solver_rtol'],1,None)
        report['branches'][name]=dict(restricted_control=plain(restricted),feedback=plain(event),
            solve_and_correction_and_step_seconds=time.perf_counter()-start,
            search_restarted=bool(event.get('applied')),
            corrected_search_equals_residual=relative(corrected['p']-corrected['r'],corrected['r']) if event.get('applied') else None)
        branches[name]=final
        audit_branch(name,final)
        save(name+'_replayed')
    # A subsequent actual checkpoint tests whether the target spike persists.
    prof.phase='historical_step16'
    cg16=cg_from(load_state(report['inputs']['16']))
    report['SUN_replay']['historical_step16_audit']=session.audit(cg16)
    save('SUN_pairing_complete')
    # Retrospective nesting check in physical Euclidean parameter geometry:
    # <B25_i,B29_j> = sum_t q25_ti^T (J_t B29_j), using archived responses.
    prof.phase='space_comparison'
    Y29=banks[29].responses();cross=np.zeros((25,29),dtype=np.longdouble)
    for i in range(25):
        q=banks[25].duals(i)
        for task in g.tasks:
            cross[i]+=q[task].astype(np.longdouble)@Y29[task].astype(np.longdouble)
    cross=np.asarray(cross,dtype=float)
    L25=np.linalg.cholesky(banks[25].gram);L29=np.linalg.cholesky(banks[29].gram)
    overlap=np.linalg.solve(L25,cross)@np.linalg.inv(L29.T)
    singular=np.linalg.svd(overlap,compute_uv=False)
    defect=banks[25].gram-cross@np.linalg.solve(banks[29].gram,cross.T)
    report['space_nesting']=dict(principal_cosines=singular.tolist(),
        old_mode_squared_projection_defects=np.diag(defect).tolist(),
        note='Dual/response inner products have reconstruction roundoff. Source refresh replaces modes; nesting must not be assumed. These are full physical parameter-space overlap diagnostics, not just target-space fit.')
    save('space_overlap_measured')
    # Evaluate actual post-step motions on the same held training-side window.
    prof.phase='finite_setup'
    with prof.measure('held_geometry'):
        held=FullFunctionalGeometry(g.c,g.point,1,precision='float64')
    baseline={t:held.c.baseline(t,held.probe,held.point) for t in held.tasks}
    for name,cg in branches.items():
        prof.phase=name+'_finite'
        with prof.measure('materialize_full_motion'):
            v,_=session.materialize(cg)
        with prof.measure('finite_evaluation'):
            report['branches'][name]['finite']=evaluate(held,v,baseline)
        del v
        save(name+'_finite_complete')
    # Replay maintenance in an isolated hard-linked cache. Only paths in this
    # diagnostic directory may be written or pruned; source files stay intact.
    prof.phase='structure_maintenance'
    scratch=OUT/'maintenance_scratch'
    shutil.copytree(SOURCE/'provisional_space',scratch,copy_function=os.link)
    paused=load_state(report['inputs']['paused'])
    pb=ProvisionalControls(g,scratch,EuclideanMetric(),copy.deepcopy(paused['provisional_state']))
    instrument_bank(pb,prof)
    pb._observe_motion=types.MethodType(profile_observe_method(prof),pb)
    latest=cg_from(paused)
    with prof.measure('full_KKT_before_refresh'):
        audit=session.audit(latest)
    with prof.measure('candidate_materialize'):
        v,_=session.materialize(latest)
    report['maintenance_profile']=dict(start_control_count=pb.size,source_iteration=int(latest['iterations']),
        cache_mode='Production materialization cache at graceful pause, hard-linked into isolated writable diagnostic directory; naturally missing entries are measured as reconstruction, not hidden warmup.')
    with prof.measure('primary_refresh'):
        event=pb.refresh(v,session.full_motion_duals(latest),audit,CASE)
    del v
    report['maintenance_profile']['primary_refresh_event']=plain(event)
    # Exact peer choice rule used by the production worker at the frozen cursor.
    cursor=paused.get('observation_cursor',0);other_task=g.tasks[cursor%len(g.tasks)]
    observed=[c for c in paused['scheduler']['cases'].values() if c['id']!=CASE and c['spec']['task']==other_task and c['iterations']>0 and c['spec'].get('probe',0)==0 and c['spec'].get('role','training')=='training']
    if observed:
        other=observed[(cursor//len(g.tasks))%len(observed)]
        other_y=g.numpy_to_tensor(np.load(SOURCE/'demands'/other['id']/'response.npy'))
        other_weights={t:1. for t in g.tasks if t!=other_task};other_rho=other['gamma']*sum(scales[t] for t in other_weights)
        with np.load(SOURCE/'PRECONDITIONER.npz') as z:
            opd=np.concatenate([1+z[t]/other_rho for t in other_weights]);otd=np.where(z[other_task]>0,z[other_task]/other_rho,1.)
        other_session=SchurSession(g,other_task,other_weights,other_rho,other_y,opd,otd)
        with np.load(SOURCE/other['cg_checkpoint']['file']) as z:
            ocg={k:z[k].copy() for k in z.files}
        with prof.measure('peer_candidate_materialize'):
            other_v,_=other_session.materialize(ocg)
        with prof.measure('peer_refresh'):
            peer_event=pb.refresh(other_v,other_session.full_motion_duals(ocg),other.get('audit') or other_session.response_residual_estimate(ocg),other['id'])
        del other_v
        report['maintenance_profile']['peer_case']=other['id']
        report['maintenance_profile']['peer_refresh_event']=plain(peer_event)
    with prof.measure('checkpoint_write'):
        checkpoint_state=copy.deepcopy(paused)
        Store(OUT/'checkpoint_write_measurement').save(checkpoint_state,latest,CASE)
    report['maintenance_profile']['final_control_count']=pb.size
    save('complete')


if __name__=='__main__':
    try:
        run()
    except Exception:
        import traceback
        OUT.mkdir(exist_ok=True)
        atomic_json(OUT/'FAILURE.json',dict(traceback=traceback.format_exc()))
        raise
