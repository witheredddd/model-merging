"""Preserve all existing iterates, revise execution and explicitly label provisional learning."""
import argparse,json,os,shutil
from pathlib import Path
from bootstrap import CORE,LEGACY
from checkpoint import Store,WriterLock,atomic_json,digest


def migrate(source,output):
    source=Path(source);output=Path(output)
    state=Store(source).load()
    if state['phase']!='PAUSED':raise ValueError('Pause the old producer at a committed checkpoint first')
    if output.exists():raise ValueError('Use an independent destination')
    # Existing checkpoint and response blobs are immutable. JSON updates use
    # atomic replacement, so the old experiment remains intact.
    shutil.copytree(source,output,copy_function=os.link)
    protocol=json.loads((output/'PROTOCOL.json').read_text())
    protocol.pop('audit_interval',None);protocol.pop('quantum_iterations',None)
    protocol.update(checkpoint_iterations=1,audit_policy='Original KKT audit at every checkpoint',
        execution='Concentrate existing progress until convergence, numerical fault or external diagnostic pause',
        learning='Separate revisable provisional space and certified reference space',
        provisional_update='M residual of current intermediate motion; refresh same demand owner, no KKT admission',
        feedback='Galerkin correction of existing full Schur iterate; original x/r/p checkpoints retained',
        certification_only_controls_optimality_claims=True,parameter_support_limit=None,control_dimension_limit=None)
    files=list(Path(__file__).parent.glob('*.py'))+list(CORE.glob('*.py'))+[LEGACY/'task_runtime.py',LEGACY/'prepare.py']
    protocol['code_hashes']={str(p.resolve()):digest(p) for p in files}
    atomic_json(output/'PROTOCOL.json',protocol)
    state['protocol_sha256']=digest(output/'PROTOCOL.json');state['code_hashes']=protocol['code_hashes']
    state['scheduler']['inflight']=None
    state.setdefault('provisional_state',None)
    state['online_repair_migration']=dict(source=str(source),all_existing_CG_states_preserved=True,
        catalog_count=len(state['scheduler']['cases']),independent_provisional_learning_enabled=True,
        solver_rtol_unchanged=protocol['solver_rtol'],original_results_not_overwritten=True)
    Store(source).load(restore_random_state=True)
    with WriterLock(output):Store(output).save(state)
    atomic_json(output/'ONLINE_REPAIR_MIGRATION.json',state['online_repair_migration'])
    print(json.dumps(state['online_repair_migration'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();migrate(a.source,a.output)
