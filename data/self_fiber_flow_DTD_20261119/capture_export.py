"""Read-only export of committed DTD numerical data; excludes model vectors."""
import csv,gzip,hashlib,json,shutil,sys,time
from pathlib import Path
import numpy as np

root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
def sha(b):return hashlib.sha256(b).hexdigest()
pointer=json.loads((root/'CURRENT.json').read_bytes());raw=(root/pointer['state']).read_bytes()
assert sha(raw)==pointer['sha256']
state=json.loads(raw);generation=state['generation'];manifest=[];refs={}
def clean(x):
    if isinstance(x,str):
        x=x.replace(str(root),'$RUN')
        for prefix in ('/media/shangxueyuan/data1/','/home/shangxueyuan/'):
            x=x.replace(prefix,'$RESOURCE/')
        return x
    if isinstance(x,dict):return {clean(k):clean(v) for k,v in x.items()}
    if isinstance(x,list):return [clean(v) for v in x]
    return x
def visit(x):
    if isinstance(x,dict):
        if str(x.get('file','')).endswith('.pt') and 'sha256' in x:
            refs[x['file']]=dict(x)
        for v in x.values():visit(v)
    elif isinstance(x,list):
        for v in x:visit(v)
def write(name,data,source=None,original=None):
    dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    entry=dict(file=name,bytes=len(data),sha256=sha(data))
    if source is not None:entry.update(source=source,source_sha256=sha(original))
    if name.endswith('.npz'):
        with np.load(dest,allow_pickle=False) as z:
            entry['arrays']={k:dict(shape=z[k].shape,dtype=str(z[k].dtype)) for k in z.files}
    manifest.append(entry)
def export_json(source,name=None):
    raw=(root/source).read_bytes();obj=json.loads(raw);visit(obj)
    write(name or source+'.gz',gzip.compress(json.dumps(clean(obj),ensure_ascii=False,allow_nan=False).encode(),mtime=0),source,raw)
    return obj
def export_binary(source):
    raw=(root/source).read_bytes();write(source,raw,source,raw)
export_json(pointer['state'],'snapshot.json.gz')
write('captured_pointer.json',json.dumps(pointer,indent=2).encode())
for folder in sorted((root/'transactions').iterdir()):
    if not folder.is_dir() or not folder.name.isdigit() or int(folder.name)>generation:continue
    if (folder/'state.json').exists():export_json(str((folder/'state.json').relative_to(root)))
    if (folder/'solver.npz').exists():export_binary(str((folder/'solver.npz').relative_to(root)))
flows=[];prefixes=[]
for f in state['transitions']:
    record=export_json(f);flows.append(record);prefixes.append(f[:-5])
if state.get('pending_transition'):prefixes.append(state['pending_transition']['prefix'])
for prefix in prefixes:
    for p in sorted(root.glob(prefix+'*.npz')):export_binary(str(p.relative_to(root)))
for folder in ('validation','baselines','innovations'):
    for p in sorted((root/folder).glob('*.json')):export_json(str(p.relative_to(root)))
history=(root/'NUMERICAL_HISTORY.jsonl').read_bytes()
history=history[:history.rfind(b'\n')+1]
write('NUMERICAL_HISTORY.jsonl.gz',gzip.compress(history,mtime=0),'NUMERICAL_HISTORY.jsonl',history)
pending=state.get('pending_transition')
if pending:write('pending_transition.json.gz',gzip.compress(json.dumps(clean(pending),allow_nan=False).encode(),mtime=0))
for name,ref in refs.items():
    path=root/name;ref.update(available_at_export=path.exists(),published=False,
        recorded_hash_not_recomputed=True,reason='Full parameter vectors excluded from Git; some old numerical scratch was already garbage collected')
    if path.exists():ref['bytes']=path.stat().st_size
write('PARAMETER_VECTOR_INDEX.json.gz',gzip.compress(json.dumps(clean(refs),allow_nan=False).encode(),mtime=0))
rows=[]
for r in flows:
    cost=r['operator_cost_cumulative'];trial=r['trials'][-1]
    rows.append(dict(anchor=r['anchor_id'],path=r['path_id'],step=r['step'],basis_before=len(r['basis_k']),basis_after=len(r['basis_next']),
        step_size=r['accepted_step_size'],learning_Log_JVP_error=trial['relative_Log_JVP_error'],half_step_error=trial['half_step_error'],
        heldout_Log_JVP_error=r['independent_Log_JVP_error'],actual_arclength=trial['actual_functional_arclength'],
        JVP_calls_cumulative=cost['JVP_calls'],VJP_calls_cumulative=cost['VJP_calls'],
        J_seconds_cumulative=cost['seconds']['J'],JT_seconds_cumulative=cost['seconds']['JT']))
if rows:
    import io
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write('transitions.csv',buf.getvalue().encode())
meta=dict(captured_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),generation=generation,
    original_current_pointer=pointer,task='DTD',phase=state['phase'],completed_transitions=len(flows),
    pending_physical_transition=bool(pending),basis_size=len(state['basis']),anchor_cursor=state['anchor_cursor'],
    capture_boundary='Transactions at or before captured generation; auxiliary observations/log copied subsequently and may be newer',
    full_parameter_vectors_in_git=False,original_run_bytes_not_modified=True,files=manifest)
(out/'MANIFEST.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(dict(output=str(out),generation=generation,files=len(manifest),bytes=sum(x['bytes'] for x in manifest),transitions=len(flows))))
