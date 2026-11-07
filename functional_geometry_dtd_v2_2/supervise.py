"""DTD-only repair run, crash/OOM recovery and hourly diagnostics."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=Path('/home/shangxueyuan/merging/tsv/results/functional_geometry_dtd_v2_2')
OUT.mkdir(parents=True,exist_ok=True);MON=OUT/'monitor';MON.mkdir(exist_ok=True)
lock=(MON/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return None
def write(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
def diag(reason,child=None):
    now=time.time();gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20).stdout
    row=dict(time=now,reason=reason,pid=None if child is None else child.pid,status=read(OUT/'status.json'),validation=read(OUT/'validation_latest.json'),gpu=gpu,
        policy='Continuous B and z CE learning; no fixed-dictionary continuation; no EC before task passes')
    write(MON/'latest.json',row);write(MON/f'diagnosis_{int(now)}.json',row)
micro=32;next_diag=0;failures=0
while True:
    if (OUT/'result.json').exists() or (OUT/'failed_gate.json').exists():diag('training ended; inspect performance gate');break
    with (OUT/'run.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,'-u','train.py','--micro',str(micro)],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
        write(MON/'supervisor.json',dict(supervisor_pid=os.getpid(),child_pid=p.pid,micro=micro,failures=failures))
        while p.poll() is None:
            if time.time()>=next_diag:
                try:diag('hourly',p)
                except Exception:pass
                next_diag=time.time()+3600
            time.sleep(10)
        if p.returncode:
            failures+=1;tail=(OUT/'run.log').read_text(errors='replace')[-12000:]
            if 'out of memory' in tail.lower() and micro>1:micro=max(1,micro//2)
            diag('exception; resume from last complete epoch',p)
            if failures>=3:
                write(OUT/'needs_repair.json',dict(reason='Repeated code/runtime failures; do not label as training success',last_log=tail,time=time.time()));sys.exit(1)
            time.sleep(10)
