"""Reconstruct deterministic main-step data IDs; no images/models or CUDA loaded."""
from pathlib import Path
import json,hashlib
import torch
disk=Path('/media/shangxueyuan/data1');base=Path('/home/shangxueyuan/merging/tsv')
for task in ('DTD','SUN397'):
    output=disk/f'fisher_rao_online_v5_{task}_medium_seed11_98f36a1_20261112'
    assets=json.loads((output/'assets.json').read_text());config=json.loads((output/'config.json').read_text())
    events=json.loads((output/'coordinate_events.json').read_text())
    manifest=base/'results/fisher_rao_v1_20261111'/task/'manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==assets['split_manifest_sha256']
    sample_paths=json.loads(manifest.read_text())['sample_ids']
    pool=assets['split']['train'];seed=config['settings']['seed'];batch=config['args']['batch_size'];rows=[]
    for event in events:
        attempt=event['iteration'];sample_step=1000+attempt-1
        generator=torch.Generator().manual_seed(seed*1000003+sample_step)
        ids=[pool[i] for i in torch.randperm(len(pool),generator=generator)[:batch].tolist()]
        rows.append(dict(main_attempt=attempt,sample_step=sample_step,ids=ids,
            dataset_relative_paths=[sample_paths[i] for i in ids],augmentation_seed=seed*10000+sample_step,
            accepted=event['accepted'],dictionary_version=event['dictionary_version']))
    result=dict(task=task,origin='Posthoc deterministic reconstruction from frozen v5 sample code; not a separate runtime ID log',
        seed=seed,batch_size=batch,train_pool_sha256=hashlib.sha256(json.dumps(pool).encode()).hexdigest(),
        manifest_sha256=assets['split_manifest_sha256'],source_sha256=config['source_sha256'],main_steps=rows,
        geometry_candidate_ids_location='geometry_events.json evidence/confirmation rows')
    (output/'main_training_data_trace.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(task=task,main_attempts=len(rows),origin=result['origin'])))
