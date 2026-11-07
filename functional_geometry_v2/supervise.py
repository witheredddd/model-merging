"""Persistent job-specific supervisor; hourly diagnostics never stop training."""
import collections,fcntl,json,math,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path('/home/shangxueyuan/merging/tsv');OUT=ROOT/'results/functional_geometry_v2_no_expert';HERE=Path(__file__).resolve().parent
MON=OUT/'monitor';MON.mkdir(parents=True,exist_ok=True)
lock=(MON/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def read(path):
    try:return json.loads(path.read_text())
    except Exception:return None
def tail(path,n=16000):
    try:
        with path.open('rb') as f:f.seek(max(0,path.stat().st_size-n));return f.read().decode(errors='replace')
    except OSError:return ''
def event(**kw):
    with (MON/'events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**kw))+'\n')
def diag(stage,pid=None,reason='hourly'):
    now=time.time();status=read(OUT/'status.json');validation=read(OUT/'validation_latest.json');alerts=[]
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
    disk=subprocess.run(['df','-h',str(OUT)],capture_output=True,text=True,timeout=20).stdout
    if status and now-status.get('time',now)>900:alerts.append('progress heartbeat older than 15 minutes; inspect stage-specific runtime')
    if validation and validation.get('DTD') and validation['DTD']['accuracy']<.8:alerts.append('DTD below required 80%; not a successful geometry result')
    trainlog=tail(OUT/'training.jsonl');last=None
    for line in reversed(trainlog.splitlines()):
        try:last=json.loads(line);break
        except Exception:pass
    ck=OUT/'training_latest.pt'
    validation_trends={};structure_counts=collections.Counter();nonfinite=[]
    for line in tail(OUT/'validation.jsonl',400000).splitlines():
        try:
            row=json.loads(line);validation_trends.setdefault(row['task'],[]).append(dict(cycle=row['cycle'],**row['current']))
        except (ValueError,KeyError):continue
    validation_trends={k:v[-3:] for k,v in validation_trends.items()}
    for line in tail(OUT/'events.jsonl',400000).splitlines():
        try:
            row=json.loads(line)
            if now-row['time']<=3600:structure_counts[row['event']]+=1
        except (ValueError,KeyError):continue
    if last:
        for key in ['ce','gradient_norm','projected_gradient_fraction','coordinate_norm']:
            if not math.isfinite(last.get(key,0)):nonfinite.append(key)
    if nonfinite:alerts.append('nonfinite training fields: '+str(nonfinite))
    free=shutil.disk_usage(OUT).free
    if free<20*1024**3:alerts.append('disk free below 20 GiB; preserve existing artifacts and inspect growth')
    row=dict(time=now,reason=reason,stage=stage,child_pid=pid,supervisor_pid=os.getpid(),status=status,validation=validation,last_training=last,
        validation_last_three=validation_trends,structure_events_last_hour=dict(structure_counts),disk_free_bytes=free,
        checkpoint_age_seconds=now-ck.stat().st_mtime if ck.exists() else None,gpu=gpu.stdout.strip(),disk=disk.strip(),alerts=alerts)
    atomic(MON/'latest.json',row);atomic(MON/f'diagnosis_{int(now)}.json',row)
    (MON/'latest.md').write_text('# Eight-task diagnosis\n\n'+json.dumps(row,indent=2)+'\n')
    event(event='diagnosis',reason=reason,stage=stage,alerts=alerts)
    return row

micro=32;lr=.02;failures=0;next_diag=time.time();child=None
atomic(MON/'policy.json',dict(hourly_seconds=3600,poll_seconds=15,automatic_resume=True,effective_batch=64,
    oom_microbatch_sequence=[32,16,8,4,2,1],nonfinite_response='resume and halve coordinate learning rate; record change',
    trained_tasks=['Cars','DTD','EuroSAT','GTSRB','MNIST','RESISC45','SVHN','SUN397'],stop_at_hourly_diagnosis=False))
while True:
    if not (OUT/'cold_complete.json').exists():stage='cold';script=['-c','from engine import Engine,cold; cold(Engine())']
    elif not (OUT/'training_complete.json').exists():stage='training';script=['train.py','--micro',str(micro),'--lr',str(lr)]
    elif not (OUT/'final_complete.json').exists():stage='final_analysis';script=['finalize.py','--micro',str(micro)]
    else:
        diag('completed',reason='final');atomic(MON/'supervisor_state.json',dict(state='completed',time=time.time()));break
    logpath=OUT/f'{stage}.log';log=logpath.open('ab',buffering=0)
    child=subprocess.Popen([sys.executable,'-u',*script],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(MON/'supervisor_state.json',dict(state='running',stage=stage,child_pid=child.pid,supervisor_pid=os.getpid(),micro=micro,lr=lr,failures=failures,time=time.time()))
    event(event='launch',stage=stage,pid=child.pid,micro=micro,lr=lr)
    started=time.time()
    while child.poll() is None:
        if time.time()>=next_diag:
            try:diag(stage,child.pid)
            except Exception:event(event='diagnosis_error',traceback=traceback.format_exc())
            next_diag=time.time()+3600
        status=read(OUT/'status.json')
        timeout=5400 if stage!='training' else 1800
        if status and time.time()-max(started,status.get('time',started))>timeout:
            event(event='stalled_child_restart',stage=stage,pid=child.pid,last_status=status)
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL)
            break
        time.sleep(15)
    code=child.wait();log.close();event(event='child_exit',stage=stage,code=code)
    if code:
        failures+=1;error=tail(logpath)
        if 'out of memory' in error.lower() and micro>1:
            old=micro;micro=max(1,micro//2);event(event='repair_microbatch',old=old,new=micro,effective_batch_unchanged=64)
        elif ('nonfinite' in error.lower() or 'nan' in error.lower()) and stage=='training':
            old=lr;lr=max(lr*.5,1e-5);event(event='repair_learning_rate',old=old,new=lr)
        diag(stage,reason='failure_and_resume');time.sleep(min(60,10+failures*5))
    else:failures=0
