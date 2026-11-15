"""One sequential GPU suite; tests must pass before launching this supervisor."""
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parent
OUT=Path('/media/shangxueyuan/data1/adaptive_fiber_20261116_r1')
PY='/home/shangxueyuan/merging/afg_env/bin/python'

def save(x):
    q=OUT/'suite_status.tmp';q.write_text(json.dumps(x,indent=2));q.replace(OUT/'suite_status.json')

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'main/FINAL.json').exists() or (OUT/'main/anchor0_live.json').exists():raise RuntimeError('Refusing to overwrite a started suite')
    (OUT/'suite_lock.json').write_text(json.dumps(dict(created_unix=time.time(),
      code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
      anchors=[0,1,2],steps_per_arm=5,initial_births=1,later_birth_rate=2,max_modes=16,max_compensators=8,
      periodic_full_refresh=2,full_refresh_omitted_after_last_step=True,
      rank_diagnostic='All prefix ranks, plus actual finite 1/2/4/8 prefixes when available; no preset final rank',
      fixed_budget_references=True,p6_and_test_opened=False),indent=2))
    common=['--anchors','0,1,2','--steps','5','--taus','.01,.1,.3,1','--heldout']
    jobs=[('adaptive',[PY,str(ROOT/'run_a100.py'),'--output',str(OUT/'main'),*common,'--max-modes','16','--max-compensators','8','--births-per-step','2','--full-refresh-every','2']),
          ('comparators',[PY,str(ROOT/'run_comparators_a100.py'),'--output',str(OUT/'comparators'),*common,'--arms','static_pair2d,dynamic_pair2d,global2d,full304'])]
    for phase,cmd in jobs:
        save(dict(stage=phase,time_unix=time.time(),command=cmd))
        with (OUT/(phase+'.log')).open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        if result.returncode:
            save(dict(stage='failure',phase=phase,returncode=result.returncode,time_unix=time.time()));raise RuntimeError(phase+' failed; see log')
    save(dict(stage='complete',time_unix=time.time()))

if __name__=='__main__':main()
