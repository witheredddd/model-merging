"""Eight-task resource/coordinate inventory; no expert-space or structural limits."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(os.environ.get('TSV_ROOT',Path(__file__).resolve().parents[2])).resolve()
COORDINATES=Path(os.environ.get('TSV_COORDINATES',
    '/media/shangxueyuan/data1/DTD_Cars_joint_tangent_20261114_r1/config.json'))
TASKS=['DTD','SUN397','Cars','EuroSAT','GTSRB','MNIST','RESISC45','SVHN']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)


def labels(ds):
    for key in ['targets','labels','_labels']:
        value=getattr(ds,key,None)
        if value is not None:
            return np.asarray(value.cpu() if torch.is_tensor(value) else value).astype(int)
    for key in ['samples','_samples']:
        value=getattr(ds,key,None)
        if value is not None:return np.asarray([row[1] for row in value],dtype=int)
    raise ValueError('Dataset label metadata requires explicit inspection: '+type(ds).__name__)


def main(out):
    start=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(ROOT),str(ROOT/'scripts/fr_hst')]
    from task_data import train_set
    coordinate=json.loads(COORDINATES.read_text())
    expected_keys=coordinate['parameter_keys'];expected_shapes=coordinate['parameter_shapes']
    expert_root=ROOT/'results/EXPERTS8_FD_REVIEWED_20261110'
    plan=json.loads((expert_root/'queue.json').read_text())['plan']
    heads=ROOT/'results/functional_geometry_v2_no_expert/assets/heads'
    rows=[]
    for task in TASKS:
        write(out/'status.json',dict(stage='eight_task_inventory',task=task))
        checkpoint=expert_root/task/'adaptive_dynamics/epoch08.pt'
        kind='completed_epoch08'
        if not checkpoint.exists():
            checkpoint=expert_root/task/'adaptive_dynamics/resume.pt';kind='full_visual_resume_not_asserted_finished'
        item=dict(task=task,expert_path=str(checkpoint),expert_kind=kind,
                  head_path=str(heads/(task+'.pt')))
        try:
            x=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
            visual=x['visual']
            if set(visual)!=set(expected_keys):raise ValueError('Visual key mismatch')
            if any(list(visual[k].shape)!=s for k,s in zip(expected_keys,expected_shapes)):
                raise ValueError('Visual shape mismatch')
            item.update(parameter_count=sum(v.numel() for v in visual.values()),
                visual_tensors=len(visual),expert_sha256=sha(checkpoint),head_sha256=sha(heads/(task+'.pt')))
            source=x['provenance']
            if source['base_sha256']!=coordinate['base_sha256']:raise ValueError('Common base mismatch')
            if item['head_sha256']!=source['head_sha256']:raise ValueError('Expert/head mismatch')
            del x,visual
            ds=train_set(task)
            y=labels(ds);unique,counts=np.unique(y,return_counts=True)
            item.update(training_examples=len(ds),classes=unique.tolist(),class_counts=counts.tolist(),
                        status='coordinate_head_and_training_metadata_checked')
            del ds
        except Exception as exc:
            item.update(status='needs_resource_fix',error=repr(exc))
        rows.append(item);write(out/'INVENTORY.json',dict(tasks=rows))
    queue=[dict(target=t,protected=[p for p in TASKS if p!=t],
                protection='Joint sum of seven per-sample-mean predictive Fishers; equal weights',
                status='waiting_for_condition_solver_acceptance_and_multi_task_runtime',
                demand_sources=['expert functional demand at current anchor',
                    'CE demand', 'class-conditional and subsequent data-window demands'],
                structural_dimension=None,max_modes=None,max_compensators=None,
                top_k_supports=None,structural_energy_cutoff=None) for t in TASKS]
    write(out/'TASK_QUEUE.json',dict(task_organization='8 one-versus-seven joint problems, not 56 independent pairs',
           tasks=queue,resources=rows,structure_policy='Discover and retain actual functional structure; no preset counts',
           EC='disabled',ODE='deferred_until_control_space_exists',
           elapsed_seconds=time.monotonic()-start,
           no_eight_task_learning_claim=True))
    write(out/'status.json',dict(stage='resource_inventory_complete',
              ready=sum(r['status']=='coordinate_head_and_training_metadata_checked' for r in rows),total=8))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
