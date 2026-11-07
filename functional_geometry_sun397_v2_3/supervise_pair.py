"""Fixed sequential SUN397 pair: r20/a.05 then r20/a.02, hourly supervision."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=Path('/root/autodl-tmp/tsv');PAIR=ROOT/'results/SUN397_rank20_pair'
PAIR.mkdir(parents=True,exist_ok=True);lock=(PAIR/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None
def event(**kw):
    with (PAIR/'queue_events.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**kw))+'\n')
def diagnose(reason,child=None):
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20).stdout.strip()
    row=dict(time=time.time(),reason=reason,pid=None if child is None else child.pid,gpu=gpu,
        experiments={n:dict(status=read(PAIR/n/'status.json'),validation=read(PAIR/n/'validation_latest.json')) for n in ['rank20_angle005','rank20_angle002']})
    write(PAIR/'monitor_latest.json',row);write(PAIR/f'diagnosis_{int(time.time())}.json',row)
arms=[('rank20_angle005',.05),('rank20_angle002',.02)]
write(PAIR/'registered_pair.json',dict(time=time.time(),task='SUN397',arms=arms,rank=20,epochs=20,seed=42,batch=64,lr_z=.01,lr_B=.1,
    order='0.05 then 0.02',same_initial_basis=True,selection='minimum validation CE',test_policy='both selections locked before either test',
    checkpoint_policy='shared initial; complete best fallback; delete obsolete resume before atomic write; prune finished resume only',hourly_diagnostics=True))
next_diag=0
for name,angle in arms:
    out=PAIR/name;out.mkdir(exist_ok=True);micro=32;failures=0
    while not (out/'training_complete.json').exists():
        log=(out/'run.log').open('ab',buffering=0)
        cmd=[sys.executable,'-B','-u',str(HERE/'train.py'),'--output',str(out),'--experiment-id',name,'--rank','20','--angle-cap',str(angle),'--micro',str(micro)]
        child=subprocess.Popen(cmd,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
        write(PAIR/'queue_status.json',dict(stage='training',experiment=name,child_pid=child.pid,supervisor_pid=os.getpid(),micro=micro,time=time.time()))
        event(event='launch',experiment=name,pid=child.pid,micro=micro)
        while child.poll() is None:
            if time.time()>=next_diag:
                try:diagnose('hourly',child)
                except Exception as e:event(event='monitor_error',error=str(e))
                next_diag=time.time()+3600
            time.sleep(10)
        log.close();event(event='exit',experiment=name,code=child.returncode)
        if child.returncode:
            failures+=1;tail=(out/'run.log').read_text(errors='replace')[-20000:]
            if 'out of memory' in tail.lower() and micro>1:
                old=micro;micro=max(1,micro//2);event(event='OOM_microbatch_repair',old=old,new=micro,effective_batch=64)
            elif failures>=3:
                write(PAIR/'needs_repair.json',dict(experiment=name,error=tail,time=time.time()));diagnose('requires code/runtime repair');sys.exit(1)
            time.sleep(10)
    assert (out/'selection_lock.json').exists() and (out/'best.pt').exists()
    if (out/'resume.pt').exists():
        (out/'resume.pt').unlink();event(event='prune_finished_resume',experiment=name,retained='complete best.pt and all logs')
    diagnose('experiment completed')
locks={name:read(PAIR/name/'selection_lock.json') for name,_ in arms}
configs={name:read(PAIR/name/'config.json') for name,_ in arms}
assert len({c['init_id'] for c in configs.values()})==1
write(PAIR/'both_selections_locked.json',dict(time=time.time(),selections=locks,configs=configs,test_accessed=False))
write(PAIR/'queue_status.json',dict(stage='locked_test_and_report',time=time.time()))
with (PAIR/'final_analysis.log').open('ab',buffering=0) as log:
    result=subprocess.run([sys.executable,'-B','-u',str(HERE/'evaluate_pair.py')],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Final evaluation/report failed; locked checkpoints and logs retained')
diagnose('completed');write(PAIR/'queue_status.json',dict(stage='completed',time=time.time()));event(event='pair_complete')
