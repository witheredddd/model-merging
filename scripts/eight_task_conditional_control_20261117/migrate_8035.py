"""Explicit protocol migration, preserving complete saved CG iterates, not old gates."""
import argparse,json,shutil,re
from pathlib import Path
import numpy as np
from bootstrap import CORE,LEGACY
from checkpoint import Store,atomic_json,digest,WriterLock
from scheduler import build_catalog,mark_migrated_active


def migrate(source,output):
    source=Path(source);output=Path(output)
    if output.exists():raise ValueError('Migration destination must be new; never overwrite an experiment')
    lock=json.loads((source/'LOCK.json').read_text())
    probes=json.loads((source/'runtime/probes.json').read_text())
    diagonals=np.load(source/'PRECONDITIONER.npz')
    tasks=list(probes['tasks']);classes={t:probes['tasks'][t]['classes'] for t in tasks}
    scheduler=build_catalog(tasks,classes,lock['gammas'])
    output.mkdir(parents=True);(output/'runtime').mkdir();(output/'demands').mkdir()
    for name in ('SCALES.json','PRECONDITIONER.npz','PRECONDITIONER.json'):
        shutil.copy2(source/name,output/name)
    shutil.copy2(source/'runtime/probes.json',output/'runtime/probes.json')
    shutil.copy2(source/'LOCK.json',output/'LEGACY_LOCK.json')
    protocol=dict(schema=2,source=lock['source'],gammas=lock['gammas'],tasks=tasks,
        solver_rtol=lock['solver_rtol'],audit_interval=lock['solver_chunk_iterations'],
        quantum_iterations=1,parameter_metric='I, inherited explicit baseline',
        protection='Joint equal-weight seven-task Fisher sum',
        automatic_wall_time_stop=False,structure_module='UNDEFINED_NOT_A_VECTOR_DICTIONARY',
        demand_catalog='All task/source/gamma combinations present before dispatch',
        legacy_source=str(source),legacy_lock_sha256=digest(source/'LOCK.json'),
        checkpoint=str(lock['source']['checkpoint']),inventory=str(lock['source']['inventory_file']),
        probe_sha256=digest(output/'runtime/probes.json'),
        scales_sha256=digest(output/'SCALES.json'),preconditioner_sha256=digest(output/'PRECONDITIONER.npz'))
    # Runtime/loaders and numerical core are fingerprinted, not just this entrypoint.
    files=list(Path(__file__).parent.glob('*.py'))+list(CORE.glob('*.py'))
    files += [LEGACY/'task_runtime.py',LEGACY/'prepare.py']
    protocol['code_hashes']={str(p.resolve()):digest(p) for p in files}
    atomic_json(output/'PROTOCOL.json',protocol)
    store=Store(output)
    state=dict(schema=2,generation=0,phase='PAUSED',protocol_sha256=digest(output/'PROTOCOL.json'),
        scheduler=scheduler,structural_state=None,structure_status='Awaiting confirmed coupling objective/update rule',
        candidate_events=[],certification_is_not_exploration_admission=True,
        code_hashes=protocol['code_hashes'],preflight_pending=True)
    store.save(state)
    imported=[]
    with WriterLock(output):
        for folder in sorted((source/'demands').glob('*')):
            if not (folder/'OUTPUT_ITERATE.npz').exists():continue
            meta=json.loads((folder/'DEMAND.json').read_text())
            spec=meta['spec'];gamma=float(folder.name.rsplit('_gamma',1)[1])
            matches=[c for c in scheduler['cases'].values() if c['spec']['task']==spec['task'] and
                c['spec']['source']==spec['source'] and c['spec'].get('label')==spec.get('label') and c['gamma']==gamma]
            if len(matches)!=1:raise ValueError('Legacy case does not align with new complete catalog')
            case=matches[0];case_id=case['id'];dest=output/'demands'/case_id;dest.mkdir()
            for name in ('response.npy','DEMAND.json'):
                shutil.copy2(folder/name,dest/name)
            with np.load(folder/'OUTPUT_ITERATE.npz',allow_pickle=False) as x:cg={k:x[k].copy() for k in x.files}
            expected=sum(len(diagonals[t]) for t in tasks)
            if any(cg[k].shape!=(expected,) or not np.all(np.isfinite(cg[k])) for k in ('x','r','p')):
                raise ValueError('Invalid full output Krylov state')
            if abs(float(cg['r']@cg['r'])-float(cg['rr']))>8*np.finfo(float).eps*expected*max(float(cg['rr']),np.finfo(float).tiny):
                raise ValueError('Stored CG squared residual differs')
            case['iterations']=int(cg['iterations']);case['status']='provisional'
            store.save(state,cg,case_id)
            mark_migrated_active(scheduler,case_id,case['cg_checkpoint'],case['iterations'])
            imported.append(dict(case_id=case_id,iterations=case['iterations'],
                saved_array_hashes={k:__import__('hashlib').sha256(cg[k].tobytes()).hexdigest() for k in cg},
                cg_checkpoint=case['cg_checkpoint']))
        if imported:
            case_id=imported[-1]['case_id']
            for i,group in enumerate(scheduler['strata']):
                if case_id in group['retry']:
                    group['retry'].remove(case_id);scheduler['inflight']=dict(case_id=case_id,stratum=i)
                    scheduler['cursor']=(i+1)%len(scheduler['strata']);break
        state['migration']=dict(source=str(source),cases=imported,
            original_rng_not_saved_by_8035=True,old_complete_process_state_not_claimed_recoverable=True,
            numerical_CG_vectors_preserved=True,protocol_intentionally_changed=True)
        store.save(state)
    atomic_json(output/'MIGRATION.json',state['migration'])
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();state=migrate(args.source,args.output)
    print(json.dumps(dict(cases=len(state['scheduler']['cases']),imported=state['migration']['cases']),indent=2))
