"""Read-only scalar-complete, selected-array export of a completed primary anchor."""
import argparse,gzip,hashlib,json,shutil,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);p.add_argument('--anchor',type=int,default=0);p.add_argument('--complete-pilot',action='store_true')
a=p.parse_args();root=a.source;out=a.destination;out.mkdir(parents=True,exist_ok=False);files=[]
def sha(b):return hashlib.sha256(b).hexdigest()
def clean(x):
    if isinstance(x,str):return x.replace(str(root),'$RUN').replace('/media/shangxueyuan/data1/','$RESOURCE/').replace('/home/shangxueyuan/merging/tsv/scripts/','$SOURCE/').replace('/home/shangxueyuan/','$RESOURCE/')
    if isinstance(x,dict):return {clean(k):clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    return x
def emit(name,data,original=None):
    q=out/name;q.parent.mkdir(exist_ok=True,parents=True);q.write_bytes(data)
    entry=dict(file=name,bytes=len(data),sha256=sha(data))
    if original is not None:entry['source_sha256']=sha(original)
    files.append(entry)
def copy(name):
    raw=(root/name).read_bytes()
    if name.endswith('.json'):
        emit(name+'.gz',gzip.compress(json.dumps(clean(json.loads(raw)),allow_nan=False).encode(),mtime=0),raw)
    else:emit(name,raw,raw)
if a.complete_pilot:
    assert (root/'SUPPLEMENT_COMPLETE.json').exists() and json.loads((root/'status.json').read_text())['stage']=='COMPLETE'
records=[];arrays=[]
for f in sorted((root/'evaluations').glob('*.json')):
    r=json.loads(f.read_text())
    if not a.complete_pilot and r['anchor']!=a.anchor:continue
    records.append(r);copy(str(f.relative_to(root)))
    selected=r['family'] in ('fisher_common_objective','naive_freeze_common_objective') and r.get('alpha')==1. and r.get('radius_index')==1
    selected|=r['family']=='sufficiency' and r.get('radius_index')==1 and r.get('sign')==1 and r['method'] in ('full_gradient','mean_gradient_1d','frozen_gradient_rank8','refreshed_gradient_rank8','existing_fisher_rank4')
    selected|=r['family']=='motion_complement' and r.get('radius_index')==1 and r['method']=='outside_frozen_gradient_rank8'
    selected|=r['family']=='expert_grid_matched'
    if selected:
        raw=(root/r['arrays']).read_bytes();assert sha(raw)==r['array_sha256'];copy(r['arrays']);arrays.append(r['arrays'])
for directory in ('solves','deletions','regions','compatibility','bases','projections'):
    for f in sorted((root/directory).glob('*')):
        if f.is_file() and (a.complete_pilot or f.name.startswith(f'a{a.anchor}')) and f.suffix in ('.json','.npz'):copy(str(f.relative_to(root)))
for name in ('CONFIG.json','DATA_SCOPE.json','PARAMETER_REGIONS.json','SOURCE_MIGRATIONS.json','ACTUAL_BASIS_AUDIT.json',f'calibration_a{a.anchor}.json'):
    if (root/name).exists():copy(name)
for f in sorted((root/'analysis').rglob('*') if a.complete_pilot else (root/'analysis'/f'anchor_{a.anchor}').iterdir()):
    if f.is_file():copy(str(f.relative_to(root)))
manifest=dict(captured_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),anchors=sorted({r['anchor'] for r in records}),evaluations=len(records),
    selected_array_payloads=len(arrays),all_scalar_case_records_included=True,all_raw_array_payloads_included=False,
    parameter_vectors_included=False,images_and_image_IDs_included=False,source_run_unchanged=True,
    completion_scope='Complete two-anchor pilot and supplements; large three-seed campaign still running' if a.complete_pilot else 'Anchor primary diagnostics only; supplements and large campaign pending',files=files)
(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in manifest.items() if k!='files'}));print('bytes',sum(x['bytes'] for x in files))
