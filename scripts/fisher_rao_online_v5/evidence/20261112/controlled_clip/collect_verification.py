from pathlib import Path
from collections import Counter
import json,hashlib,math,statistics
import torch

root=Path('/media/shangxueyuan/data1/fisher_rao_online_v5_DTD_controlled_seed11_20261112')
source=Path('/home/shangxueyuan/merging/tsv/scripts/fisher_rao_online_v5_3a57857')
read=lambda p:json.loads(p.read_text())
if not (root/'report.json').exists():
    print(json.dumps(dict(status='running',periodic=read(root/'status.json') if (root/'status.json').exists() else None)))
    raise SystemExit(0)
report=read(root/'report.json');config=read(root/'config.json');assets=read(root/'assets.json')
events=read(root/'coordinate_events.json');windows=read(root/'geometry_events.json')
final=torch.load(root/'final_state.pt',map_location='cpu',weights_only=True,mmap=True)
committed=[p for w in windows for p in w['proposals'] if p.get('committed')]
def ranges(key):
    values=[e[key] for e in events if isinstance(e.get(key),(float,int)) and math.isfinite(e[key])]
    return dict(min=min(values),median=statistics.median(values),max=max(values)) if values else None
summary=dict(status='complete',source_commit='3a57857f24b66e55cdb3d179b8d721ad1374adea',
    source_hashes_match=all(hashlib.sha256((source/name).read_bytes()).hexdigest()==h for name,h in config['source_sha256'].items()),
    main_coordinate_attempts=len(events),accepted_coordinate_attempts=sum(e['accepted'] for e in events),
    geometry_windows=len(windows),geometry_rollbacks=[w for w in windows if w.get('reason')=='geometry_transaction_rolled_back'],
    base_validation=report['base_validation'],final_validation=report['final_validation'],
    selected_validation=report['Vcal_selected_validation'],independent_final=report['independent_final'],
    independent_selected=report['independent_Vcal_selected'],initialization=report['initialization'],
    final_dictionary_version=final['dictionary_version'],final_dictionary_dimension=final['S'].shape[1],
    final_groups=[dict(name=g['name'],rank=g['a'].numel()) for g in final['groups']],
    actual_online_images=report['actual_online_images'],seconds=report['seconds'],peak_gpu_bytes=report['peak_gpu_bytes'],
    operations=report['compute_counts']['operations'],stop_reason=report['stop_reason'],
    structural_classifications=Counter(w.get('persistence',{}).get('kind','no_structure_event') for w in windows),
    committed_geometry_operations=[dict(kind=p['kind'],future_CE_gain=p['future_CE_gain'],
        future_parameter_difference=p['future_parameter_difference'],old_cost=p['old_branch_cost'],new_cost=p['new_branch_cost']) for p in committed],
    capacities=[dict(window=w['window'],before=w.get('capacity_before'),after=w.get('capacity_after')) for w in windows],
    confirmation=[w.get('independent_confirmation') for w in windows],
    parameter_step_norm=ranges('actual_parameter_step_norm'),radius=ranges('radius_before'),
    sensitivity_max=ranges('sample_sensitivity_max'),sensitivity_median=ranges('sample_sensitivity_median'),
    shrunk_coordinate_steps=sum(e.get('step_shrunk',False) for e in events),
    probe_training_disjoint=not set(assets['split']['X']).intersection(assets['split']['train']),
    official_test_opened=False,full_DTD_training_complete=False,independent_functional_groups_established=False)
(root/'controlled_verification.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps(summary))
