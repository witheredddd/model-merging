"""Read-only final task JVP check. Primary epsilon/tolerances fixed in advance."""
from pathlib import Path
import sys,json,time
import torch
source=Path('/home/shangxueyuan/merging/tsv/scripts/fisher_rao_online_v5_98f36a1');sys.path.insert(0,str(source))
from clip_runtime import ClipTask
from checkpointing import load_model
root=Path('/home/shangxueyuan/merging/tsv');disk=Path('/media/shangxueyuan/data1')
for task in ('DTD','SUN397'):
    output=disk/f'fisher_rao_online_v5_{task}_medium_seed11_98f36a1_20261112'
    if len(json.loads((output/'coordinate_events.json').read_text()))!=128:raise RuntimeError('Wait for both tasks to finish')
for task in ('DTD','SUN397'):
    output=disk/f'fisher_rao_online_v5_{task}_medium_seed11_98f36a1_20261112';qa=output/'posthoc_jvp';qa.mkdir(exist_ok=True)
    raw=ClipTask(root,qa,128,32,task=task)
    saved=torch.load(output/'recovery_state.pt',map_location='cpu',weights_only=True,mmap=True)
    if saved['identity']['base_sha256']!=raw.rt.input['base_sha256'] or saved['identity']['head_sha256']!=raw.rt.input['head_sha256']:
        raise ValueError('Model asset mismatch')
    state=load_model(saved['state'],raw);point=state.parameters();direction=state.S[:,0]
    before=dict(raw.rt.calls);started=time.monotonic();exact=raw.jvp(point,direction);rows=[]
    for epsilon in (.001,.003,.01):
        numerical=(raw.phi(point+epsilon*direction)-raw.phi(point-epsilon*direction))/(2*epsilon)
        rows.append(dict(epsilon=epsilon,relative_error=float((numerical-exact).norm()/exact.norm().clamp_min(1e-12)),
            cosine=float(torch.nn.functional.cosine_similarity(numerical.flatten(),exact.flatten(),dim=0))))
    primary=next(row for row in rows if row['epsilon']==.003)
    result=dict(task=task,status='passed' if primary['relative_error']<=.02 and primary['cosine']>=.999 else 'failed',
        predeclared_primary_epsilon=.003,predeclared_relative_tolerance=.02,predeclared_cosine=.999,
        finite_difference=rows,probe_n=128,classes=raw.classes,
        calls={k:raw.rt.calls[k]-before[k] for k in before},seconds=time.monotonic()-started,
        optimizer_updates=0,trained_state_modified=False,scope='One final-state dictionary direction; not a whole-trajectory proof')
    (output/'final_jvp_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result))
    del state,point,direction,exact,saved,raw;torch.cuda.empty_cache()
