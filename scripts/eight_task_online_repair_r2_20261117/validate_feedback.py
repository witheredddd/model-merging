import json,time
from pathlib import Path
import numpy as np
from bootstrap import CORE
from checkpoint import Store,atomic_json
from task_runtime import load_eight_geometry
from geometry import EuclideanMetric,FullFunctionalGeometry
from provisional import ProvisionalControls
from schur_session import SchurSession
from coarse import correct
from finite_monitor import evaluate

out=Path('/media/shangxueyuan/data1/eight_task_online_repair_r2_20261117')
report=dict(stage='loading',conditional_optimality_claim=False,empirical_drift_is_not_a_rigorous_error_bound=True)
def save(stage):
    report['stage']=stage;atomic_json(out/'FEEDBACK_VALIDATION.json',report);print(stage,flush=True)
store=Store(out);state=store.load();protocol=json.loads((out/'PROTOCOL.json').read_text())
case=state['scheduler']['cases']['EuroSAT_CE_gamma0.1'];cg=store.load_cg(case)
report.update(source_generation=state['generation'],source_iteration=int(cg['iterations']),control_count=len(state['provisional_state']['modes']))
save('loading_model')
g,_=load_eight_geometry(out/'runtime',protocol['inventory'],protocol['checkpoint'])
bank=ProvisionalControls(g,out/'provisional_space',EuclideanMetric(),state['provisional_state'])
y=g.numpy_to_tensor(np.load(out/'demands'/case['id']/'response.npy'))
scales=json.loads((out/'SCALES.json').read_text())['scales'];d=np.load(out/'PRECONDITIONER.npz');weights={t:1. for t in g.tasks if t!='EuroSAT'};rho=case['gamma']*sum(scales[t] for t in weights)
pd=np.concatenate([1+d[t]/rho for t in weights]);td=np.where(d['EuroSAT']>0,d['EuroSAT']/rho,1.)
session=SchurSession(g,'EuroSAT',weights,rho,y,pd,td)
report['before']=session.audit(cg);save('before_audited')
t0=time.monotonic();updated,feedback=correct(session,cg,bank)
report['correction_seconds']=time.monotonic()-t0;report['feedback']=feedback
report['after']=session.audit(updated);save('correction_audited')
# Same initial checkpoint and five continuation steps; report extra correction
# cost explicitly. This single pair is evidence, not a general speedup claim.
results={};motions={}
for name,start in [('baseline',cg),('corrected',updated)]:
    t0=time.monotonic();_,info,final=session.step(start,protocol['solver_rtol'],5,None)
    results[name]=dict(audit=session.audit(final),seconds=time.monotonic()-t0,info=int(info),iterations=int(final['iterations']))
    motions[name]=session.materialize(final)[0]
    save(name+'_continued')
report['five_step_comparison']=results
before_v=session.materialize(cg)[0];after_v=session.materialize(updated)[0]
report['per_task_linear_response']={}
for task in g.tasks:
    base=g.tensor_to_numpy(g.jvp(task,before_v));new=g.tensor_to_numpy(g.jvp(task,after_v));continued=g.tensor_to_numpy(g.jvp(task,motions['corrected']))
    report['per_task_linear_response'][task]=dict(before_norm=float(np.linalg.norm(base)),after_norm=float(np.linalg.norm(new)),correction_response_norm=float(np.linalg.norm(new-base)),five_step_response_drift=float(np.linalg.norm(continued-new)))
report['mode_functional_metadata']=[dict(owner=m['source']['case_id'],dual_lift_relative_error=m['dual_lift_relative_error'],CE_signed_slopes=m['CE_signed_slopes']) for m in bank.state['modes']]
save('linear_response_measured')
# The independent training-side window measures finite functionality. No
# validation metric admits modes or chooses a step size.
held=FullFunctionalGeometry(g.c,g.point,1,precision='float64')
report['finite_before']=evaluate(held,before_v);save('finite_before_measured')
report['finite_after']=evaluate(held,after_v);save('complete')
