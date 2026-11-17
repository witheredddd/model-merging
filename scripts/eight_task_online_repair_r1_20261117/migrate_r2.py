"""Explicit new-code migration of a paused transactional experiment."""
import argparse,json,shutil
from pathlib import Path
from bootstrap import CORE,LEGACY
from checkpoint import Store,atomic_json,digest,WriterLock


def migrate(source,output):
    source=Path(source);output=Path(output)
    if output.exists():raise ValueError('Keep old runs intact; choose a new output path')
    old=Store(source).load()
    if old['phase']!='PAUSED':raise ValueError('Pause the producer at a committed iteration before migration')
    shutil.copytree(source,output)
    protocol=json.loads((output/'PROTOCOL.json').read_text())
    previous=digest(output/'PROTOCOL.json')
    protocol.update(structure_module='Online conditional control space by approved M-residual extension',
        control_rule='r = vstar - B (B^T M B)^dagger B^T M vstar; only machine-independent residual becomes a new mode',
        no_discrete_hard_blocks=True,no_energy_or_novelty_cutoff=True,
        external_resource_management_only=True)
    files=list(Path(__file__).parent.glob('*.py'))+list(CORE.glob('*.py'))+[LEGACY/'task_runtime.py',LEGACY/'prepare.py']
    protocol['code_hashes']={str(p.resolve()):digest(p) for p in files}
    # Holdout demands have their own strata, are never returned to structure
    # learning, and receive independent full-space numerical references.
    scheduler=old['scheduler']
    for task in protocol['tasks']:
        ids=[]
        for gamma in protocol['gammas']:
            case_id=f'{task}_heldout_CE_gamma{gamma:g}'
            scheduler['cases'][case_id]=dict(id=case_id,spec=dict(task=task,source='CE',probe=1,
                role='validation',name=f'{task}_heldout_CE'),gamma=gamma,status='candidate',
                iterations=0,cg_checkpoint=None,inner_rtol=None,visits=0,certified=False)
            ids.append(case_id)
        scheduler['strata'].append(dict(task=task,family='heldout_CE',new=ids,retry=[],next='new'))
    atomic_json(output/'PROTOCOL.json',protocol)
    old['protocol_sha256']=digest(output/'PROTOCOL.json');old['code_hashes']=protocol['code_hashes']
    old['structural_state']=None;old['structure_status']='Online conditional control space; not hard blocks'
    old['revision_migration']=dict(source=str(source),previous_protocol_sha256=previous,
        saved_scheduler_and_CG_preserved=True,new_validation_strata_added=True,
        prior_failed_or_unconverged_results_not_certified=True)
    with WriterLock(output):Store(output).save(old)
    atomic_json(output/'REVISION_MIGRATION.json',old['revision_migration'])
    print(json.dumps(dict(cases=len(scheduler['cases']),phase=old['phase'],code_files=len(protocol['code_hashes'])),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();migrate(args.source,args.output)
