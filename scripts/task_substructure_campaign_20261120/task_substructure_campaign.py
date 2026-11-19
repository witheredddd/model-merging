"""Server-owned sequential campaign; independent of the desktop and SSH session."""
import argparse,hashlib,json,os,signal,subprocess,sys,time,shutil,tarfile
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True);p.add_argument('--pilot-output',type=Path,required=True)
p.add_argument('--pilot-unit',default='task-substructure-DTD-Cars-20261120.service')
p.add_argument('--pilot-scripts',type=Path,required=True);p.add_argument('--large-scripts',type=Path,required=True)
p.add_argument('--helpers',type=Path,required=True)
for key in ('inventory','probes','coordinates','trace','flow-source'):p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--seeds',type=int,nargs='+',default=[41,42,43]);p.add_argument('--skip-large',action='store_true')
a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
def atomic(path,obj):
    tmp=Path(str(path)+'.tmp');tmp.parent.mkdir(exist_ok=True,parents=True);tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
stop=False;child=None
def interrupt(*_):
    global stop
    stop=True
    if child is not None and child.poll() is None:child.terminate()
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
def status(stage,**kw):
    atomic(a.root/'CAMPAIGN_STATUS.json',dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),stage=stage,**kw))
    print(json.dumps(dict(stage=stage,**kw)),flush=True)
def resource():
    while shutil.disk_usage(a.root).free<120*1024**3:
        status('WAITING_FOR_STORAGE',free_bytes=shutil.disk_usage(a.root).free,
            data_deleted=False,dimension_truncated=False)
        if stop:raise InterruptedError()
        time.sleep(60)
def args(output,anchors,probes=None):
    pairs=[('output',output),('inventory',a.inventory),('probes',probes or a.probes),('coordinates',a.coordinates),('trace',a.trace),('flow-source',a.flow_source)]
    result=[]
    for k,v in pairs:result+=['--'+k,str(v)]
    return result+['--anchors',*map(str,anchors),'--rtol','0.001']
def report(output,anchors):
    for anchor in anchors:
        if not (output/f'bases/a{anchor}.json').exists():continue
        subprocess.run([sys.executable,str(a.helpers/'task_substructure_report.py'),str(output),'--anchor',str(anchor)],check=True)
    summaries=[]
    for f in (output/'analysis').glob('anchor_*/SUMMARY.json'):summaries.append(json.loads(f.read_text()))
    atomic(output/'analysis/INDEX.json',dict(anchor_summaries=summaries,
        all_completed_case_metrics_reported=True,missing_cases_are_not_success=True))
def run_node(name,cmd,output,anchors,expected):
    global child
    done=a.root/'nodes'/f'{name}.json'
    if done.exists() and json.loads(done.read_text()).get('complete'):return
    for attempt in range(1,4):
        resource();status('RUNNING',node=name,attempt=attempt,output=str(output),command=cmd)
        with (a.root/f'{name}.log').open('a') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);last=0;reported=set()
            while child.poll() is None:
                if stop:child.terminate();child.wait();raise InterruptedError()
                if time.monotonic()-last>300:
                    last=time.monotonic()
                    current=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
                    status('RUNNING',node=name,attempt=attempt,output=str(output),current=current)
                    for anchor in anchors:
                        if anchor not in reported and (output/f'ANCHOR_COMPLETE_{anchor}.json').exists():
                            report(output,[anchor]);reported.add(anchor)
                time.sleep(10)
            code=child.returncode;child=None
        current=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
        valid=code==0 and ((output/expected).exists() if expected else current.get('stage')=='COMPLETE')
        if valid:
            atomic(done,dict(complete=True,returncode=code,attempts=attempt));report(output,anchors);return
        atomic(a.root/'failures'/f'{name}_{attempt}.json',dict(returncode=code,status=current,
            failure=json.loads((output/'FAILURE.json').read_text()) if (output/'FAILURE.json').exists() else None))
        if (output/'RESOURCE_LIMIT.json').exists():
            resource()
        if stop:raise InterruptedError()
        time.sleep(30)
    raise RuntimeError('Node failed after three auditable attempts: '+name)
def make_probes(seed):
    out=a.root/'resources'/f'probes_seed{seed}.json'
    if out.exists():return out
    sys.path.insert(0,str(a.pilot_scripts))
    import bootstrap
    sys.path[:0]=[str(bootstrap.LEGACY),'/home/shangxueyuan/merging/tsv/scripts/fr_hst','/home/shangxueyuan/merging/tsv']
    from task_runtime import balanced_ids
    from prepare import labels
    from task_data import train_set
    import numpy as np
    rng=np.random.default_rng(seed);tasks={}
    for task in ('DTD','Cars'):
        ds=train_set(task,lambda x:x);y=labels(ds);classes=np.unique(y);count=len(classes)*(20 if task=='DTD' else 5)
        ids0=balanced_ids(y,count,rng,set());ids1=balanced_ids(y,2*count,rng,set(ids0))
        assert not set(ids0)&set(ids1)
        tasks[task]=dict(ids=[ids0,ids1],classes=classes.tolist(),training_side=True)
    atomic(out,dict(tasks=tasks,seed=seed,official_test_opened=False,sampling='Stratified, disjoint construction/selection/evaluation'))
    return out
def package(output):
    # Raw model vectors and image IDs stay private. The manifest indexes full arrays.
    dest=output/'RESULT_PACKAGE';dest.mkdir(exist_ok=True);files=[]
    for folder in ('evaluations','deletions','solves','linear_responses','analysis','bases','regions'):
        for f in sorted((output/folder).rglob('*')):
            if f.is_file() and f.suffix!='.pt':files.append(dict(file=str(f.relative_to(output)),bytes=f.stat().st_size,sha256=sha(f)))
    for name in ('CONFIG.json','DATA_SCOPE.json','PARAMETER_REGIONS.json','TIMING_HISTORY.jsonl','gradient_timing.jsonl','ACTUAL_BASIS_AUDIT.json'):
        f=output/name
        if f.exists():files.append(dict(file=name,bytes=f.stat().st_size,sha256=sha(f)))
    atomic(dest/'MANIFEST.json',dict(files=files,raw_array_payload_on_server=True,parameter_vectors_excluded=True,
        image_IDs_excluded=True,no_precision_reduction=True))
    with tarfile.open(dest/'REPORTS.tar.gz','w:gz') as t:
        for f in sorted((output/'analysis').rglob('*')):
            if f.is_file():t.add(f,arcname=str(f.relative_to(output)))
        t.add(dest/'MANIFEST.json',arcname='MANIFEST.json')
try:
    import fcntl
    lock=(a.root/'CAMPAIGN.lock').open('ab');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    anchors=[r['step'] for r in json.loads((a.trace/'trace.json').read_text())['rows']]
    plan=dict(pilot_anchors=[0,25],large_anchors=anchors,seeds=a.seeds,skip_large=a.skip_large,
        large_construction_demands=15,large_unseen_demands=4,no_dimension_cap=True,
        stages=['pilot','full-complement and expert baselines','larger three-way probes at every real anchor','full metrics and packages'],
        new_expert_training=False,EC_evolution=False)
    if (a.root/'PLAN.json').exists() and json.loads((a.root/'PLAN.json').read_text())!=plan:raise ValueError('Frozen campaign plan changed')
    atomic(a.root/'PLAN.json',plan)
    sources={str(f):sha(f) for folder in (a.pilot_scripts,a.large_scripts,a.helpers) for f in folder.glob('*.py')}
    frozen=a.root/'SOURCES.json'
    if frozen.exists() and json.loads(frozen.read_text())!=sources:raise ValueError('Campaign source changed; explicit migration required')
    atomic(frozen,sources)
    status('WAITING_FOR_EXISTING_PILOT')
    while subprocess.run(['systemctl','--user','is-active','--quiet',a.pilot_unit]).returncode==0:
        if stop:raise InterruptedError()
        time.sleep(30)
    state=json.loads((a.pilot_output/'status.json').read_text()) if (a.pilot_output/'status.json').exists() else {}
    if state.get('stage')!='COMPLETE':
        run_node('pilot_resume',[sys.executable,str(a.pilot_scripts/'worker.py'),*args(a.pilot_output,[0,25])],a.pilot_output,[0,25],None)
    run_node('pilot_supplement',[sys.executable,str(a.helpers/'task_substructure_supplement.py'),'--scripts',str(a.pilot_scripts),*args(a.pilot_output,[0,25])],a.pilot_output,[0,25],'SUPPLEMENT_COMPLETE.json')
    report(a.pilot_output,[0,25]);package(a.pilot_output)
    if not a.skip_large:
        for seed in a.seeds:
            probes=make_probes(seed);output=a.root/f'large_seed{seed}'
            run_node(f'large_{seed}',[sys.executable,str(a.large_scripts/'worker.py'),*args(output,anchors,probes),'--batch-size','16'],output,anchors,None)
            run_node(f'large_supplement_{seed}',[sys.executable,str(a.helpers/'task_substructure_supplement.py'),'--scripts',str(a.large_scripts),*args(output,anchors,probes),'--batch-size','16'],output,anchors,'SUPPLEMENT_COMPLETE.json')
            report(output,anchors);package(output)
    status('COMPLETE',pilot=str(a.pilot_output),large_runs=[str(a.root/f'large_seed{s}') for s in a.seeds] if not a.skip_large else [])
except InterruptedError:status('PAUSED',resumable=True)
except Exception:
    import traceback
    atomic(a.root/'CAMPAIGN_FAILURE.json',dict(traceback=traceback.format_exc()));status('FAILED');raise
