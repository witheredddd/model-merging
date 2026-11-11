"""Read-only finite-difference check after the bounded run; no parameter updates."""
from pathlib import Path
import sys,json,time
import torch
root=Path('/media/shangxueyuan/data1/fisher_rao_online_v5_DTD_controlled_seed11_20261112')
source=Path('/home/shangxueyuan/merging/tsv/scripts/fisher_rao_online_v5_3a57857')
sys.path.insert(0,str(source))
from clip_runtime import ClipTask
from state import ModelState,Group
from fr_math import distance
from structural_persistence import transport_columns
if not (root/'report.json').exists():raise RuntimeError('Training must finish before numerical verification')
qa=root/'posthoc_numerical';qa.mkdir(exist_ok=True)
started=time.monotonic();raw=ClipTask('/home/shangxueyuan/merging/tsv',qa,128,32)
stored=torch.load(root/'final_state.pt',map_location='cpu',weights_only=True,mmap=True)
if stored['base_sha256']!=raw.rt.input['base_sha256']:raise ValueError('Base asset hash changed')
device=raw.theta0.device
state=ModelState(raw.theta0,stored['S'].to(device),tuple(Group(g['name'],g['Z'].to(device),g['a'].to(device))
    for g in stored['groups']),stored['dictionary_version']).validate()
theta=state.parameters().detach();direction=state.S[:,0].detach()
before=raw.rt.calls.copy()
q0=raw.phi(raw.theta0).detach();q=raw.phi(theta).detach()
r0=raw.jvp(raw.theta0,direction).detach();r=raw.jvp(theta,direction).detach()
transported=transport_columns(q,q0,r.flatten()[:,None])[:,0].reshape_as(r0)
rows=[]
for epsilon in [.001,.003,.01]:
    plus=raw.phi(theta+epsilon*direction).detach();minus=raw.phi(theta-epsilon*direction).detach()
    numerical=(plus-minus)/(2*epsilon)
    rows.append(dict(epsilon=epsilon,relative_error=float((numerical-r).norm()/r.norm().clamp_min(1e-12)),
        cosine=float(torch.nn.functional.cosine_similarity(numerical.flatten(),r.flatten(),dim=0))))
primary=next(row for row in rows if row['epsilon']==.01)
result=dict(status='passed' if primary['relative_error']<=.02 and primary['cosine']>=.999 else 'numerical_check_failed',
    primary_epsilon=.01,predeclared_primary_relative_tolerance=.02,predeclared_primary_cosine=.999,
    finite_difference=rows,current_vs_base_FR=float(distance(q,q0)),
    ambient_relative_JVP_change=float((r-r0).norm()/r0.norm().clamp_min(1e-12)),
    transported_relative_JVP_change=float((transported-r0).norm()/r0.norm().clamp_min(1e-12)),
    dictionary_shape=list(state.S.shape),coordinate_count=state.total_rank,
    fixed_global_base=True,no_optimizer_updates=True,training_result_modified=False,
    calls={key:raw.rt.calls[key]-before[key] for key in before},seconds=time.monotonic()-started)
(qa/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result))
