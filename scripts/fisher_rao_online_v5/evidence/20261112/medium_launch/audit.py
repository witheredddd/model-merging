"""CPU-only audit of completed medium runs and immutable checkpoint headers."""
from pathlib import Path
import json,hashlib,math,collections,statistics
import sys
import torch
disk=Path('/media/shangxueyuan/data1')
source=Path('/home/shangxueyuan/merging/tsv/scripts/fisher_rao_online_v5_98f36a1')
read=lambda p:json.loads(p.read_text())
all_results={}
for task in (sys.argv[1:] or ('DTD','SUN397')):
    root=disk/f'fisher_rao_online_v5_{task}_medium_seed11_98f36a1_20261112'
    report=read(root/'report.json');events=read(root/'coordinate_events.json')
    if len(events)!=128 or report['status']!='complete' or report['stop_reason']!='step_limit':
        raise RuntimeError(f'{task}: medium run is not complete')
    windows=read(root/'geometry_events.json');config=read(root/'config.json');assets=read(root/'assets.json')
    checkpoint=torch.load(root/'recovery_state.pt',map_location='cpu',weights_only=True,mmap=True)
    if checkpoint['iteration']!=128:raise RuntimeError('Final recovery snapshot does not match attempt count')
    if sum(checkpoint['ledger']['images'].values())!=report['actual_online_images']:
        raise RuntimeError('Report budget differs from committed recovery state')
    state=checkpoint['state'];G=torch.cat([g['Z'] for g in state['groups']],1)
    singular=torch.linalg.svdvals(G)
    archived=[]
    for path in sorted((root/'checkpoints').glob('*.pt')):
        item=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        archived.append(dict(file=path.name,iteration=item['iteration'],online_images=sum(item['ledger']['images'].values()),
            dictionary_version=item['state']['dictionary_version'],dictionary_dimension=item['state']['S'].shape[1],
            groups=[dict(name=g['name'],coordinates=len(g['a'])) for g in item['state']['groups']],bytes=path.stat().st_size))
        del item
    def stat(values):
        values=[float(x) for x in values if x is not None]
        return dict(min=min(values),median=statistics.median(values),max=max(values)) if values else None
    committed=[p for w in windows for p in w.get('proposals',[]) if p.get('committed')]
    fallbacks=collections.Counter(e['solver'].get('fallback') for e in events if e['solver'].get('fallback'))
    result=dict(task=task,source_commit='98f36a1b5135bce79be21b46e0024c22629d683d',
        source_hashes_match=all(hashlib.sha256((source/name).read_bytes()).hexdigest()==digest for name,digest in config['source_sha256'].items()),
        status=report['status'],stop_reason=report['stop_reason'],main_attempts=len(events),accepted=sum(e['accepted'] for e in events),
        base_validation=report['base_validation'],final_validation=report['final_validation'],
        selected_validation=report['Vcal_selected_validation'],independent_final=report['independent_final'],
        independent_selected=report['independent_Vcal_selected'],actual_online_images=report['actual_online_images'],
        seconds=report['seconds'],peak_gpu_bytes=report['peak_gpu_bytes'],compute_counts=report['compute_counts'],
        solver_fallbacks=dict(fallbacks),solver_condition=stat(e['solver'].get('condition') for e in events),
        retained_minimum_singular=stat(e['solver'].get('minimum_retained_singular_value') for e in events),
        coordinate_norm=stat(e['coordinate_norm'] for e in events),coordinate_step=stat(e['actual_coordinate_step_norm'] for e in events),
        parameter_step_reconstruction_error=stat(e['solver'].get('parameter_step_reconstruction_relative_error') for e in events),
        coordinate_accumulation_error=stat(e['coordinate_accumulation_relative_error'] for e in events),
        shrunk_steps=sum(e['step_shrunk'] for e in events),geometry_windows=len(windows),
        geometry_rollbacks=[w for w in windows if w.get('reason')=='geometry_transaction_rolled_back'],
        structural_classifications=dict(collections.Counter(w.get('persistence',{}).get('kind','none') for w in windows)),
        committed_geometry_operations=[dict(window=w['window'],kind=p['kind'],future_CE_gain=p['future_CE_gain'],
            net_gain=p['net_gain'],old_cost=p['old_branch_cost'],new_cost=p['new_branch_cost']) for w in windows for p in w.get('proposals',[]) if p.get('committed')],
        final_dictionary_dimension=state['S'].shape[1],final_coordinate_count=G.shape[1],
        final_parameter_rank=int(torch.linalg.matrix_rank(G,rtol=1e-8)),
        final_parameter_rank_relative_tolerance=1e-8,final_G_singular_values=singular.tolist(),
        final_machine_precision_rank=int((singular>max(G.shape)*torch.finfo(G.dtype).eps*singular[0]).sum()),
        final_groups=[dict(name=g['name'],coordinates=len(g['a'])) for g in state['groups']],
        resume=read(root/'resume_status.json'),checkpoint_headers=archived,
        expected_periodic_checkpoints_preserved=all(i in {a['iteration'] for a in archived} for i in range(0,129,8)),
        probe_training_disjoint=not set(assets['split']['X']).intersection(assets['split']['train']),
        official_test_opened=report['official_test_opened'],independent_functional_groups_established=False,
        million_budget_training_complete=False,long_training_stability_established=False,
        engineering_checks_scope='Only source identity, exact resume, finite/reconstructed steps, no solver fallback/rollback and checkpoint coverage; not a coordinate-amplitude safety guarantee')
    result['engineering_medium_checks_passed']=bool(result['source_hashes_match'] and not result['solver_fallbacks']
        and not result['geometry_rollbacks'] and result['expected_periodic_checkpoints_preserved']
        and result['resume']['resumed_iteration']==8 and result['resume']['coefficient_reconstruction_error']<1e-12
        and result['parameter_step_reconstruction_error']['max']<1e-6)
    (root/'medium_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    all_results[task]=result
print(json.dumps(all_results))
