"""Read-only baseline on the SAME independent development IDs after training."""
from pathlib import Path
import json,sys,time
import torch
source=Path('/home/shangxueyuan/merging/tsv/scripts/fisher_rao_online_v5_98f36a1')
sys.path.insert(0,str(source))
from clip_runtime import ClipTask
root=Path('/home/shangxueyuan/merging/tsv');disk=Path('/media/shangxueyuan/data1')
for task in ('DTD','SUN397'):
    output=disk/f'fisher_rao_online_v5_{task}_medium_seed11_98f36a1_20261112'
    events=json.loads((output/'coordinate_events.json').read_text())
    if len(events)!=128:raise RuntimeError('Only run baseline diagnostic after both medium trajectories finish')
for task in ('DTD','SUN397'):
    output=disk/f'fisher_rao_online_v5_{task}_medium_seed11_98f36a1_20261112'
    original=json.loads((output/'assets.json').read_text());qa=output/'posthoc_base';qa.mkdir(exist_ok=True)
    start=time.monotonic();raw=ClipTask(root,qa,128,32,task=task)
    for key in ('base_sha256','head_sha256','split_manifest_sha256','split'):
        if raw.rt.input[key]!=original[key]:raise ValueError('Independent baseline asset/split mismatch')
    before=dict(raw.rt.calls);score=raw.generalization(raw.theta0)
    result=dict(task=task,base_independent=score,development_ids=raw.rt.split['Xprime'],
        calls={k:raw.rt.calls[k]-before[k] for k in before},seconds=time.monotonic()-start,
        trained_state_modified=False,optimizer_updates=0,official_test_opened=False,
        purpose='Posthoc untrained-base reference on same independent development sample IDs')
    (output/'independent_base_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result));del raw;torch.cuda.empty_cache()
