"""Anchor0 first; same frozen fast protocol then extends to anchors1/2."""
import os,json,time,subprocess,hashlib
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parent;OUT=Path('/media/shangxueyuan/data1/adaptive_fiber_20261116_r2')
PY='/home/shangxueyuan/merging/afg_env/bin/python'

def save(x):
    q=OUT/'suite_status.tmp';q.write_text(json.dumps(x,indent=2));q.replace(OUT/'suite_status.json')

def main():
    if (OUT/'suite_lock.json').exists():raise RuntimeError('Suite already started; preserve output and resume explicitly')
    verified=json.loads((OUT/'preflight/resume_verified.json').read_text())
    assert verified['passed'] and (OUT/'r1_stop_and_preservation_audit.json').exists()
    lock=dict(created_unix=time.time(),source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
      anchors_in_order=[0,1,2],steps=5,anchor0_resume=verified,arms=['static_pair2d','dynamic_pair2d','global2d'],
      full304_dynamic='Deferred to independent cost experiment; previous local capacity retained',
      full_refresh='Exception-only, maximum1 per anchor for failed local reliability; no periodic calibration',
      dimension_ablation='Predicted capacity online; actual finite audit only at frozen endpoint',
      initial_anchor0_prefix='Two accepted r1 updates preserved; r2 cost counters exclude inherited r1 work',
      p4_p5='Reused exploratory final evaluation, no decisions',p6_and_test_opened=False)
    (OUT/'suite_lock.json').write_text(json.dumps(lock,indent=2))
    for k in [0,1,2]:
        main_cmd=[PY,str(ROOT/'fast_online.py'),'--output',str(OUT/f'anchor{k}/adaptive'),'--anchor',str(k),'--steps','5']
        if k==0:main_cmd+=['--resume',verified['checkpoint']]
        controls=[PY,str(ROOT/'fast_comparators.py'),'--output',str(OUT/f'anchor{k}/comparators'),
          '--anchors',str(k),'--steps','5','--arms','static_pair2d,dynamic_pair2d,global2d','--heldout']
        for phase,cmd in [('adaptive',main_cmd),('comparators',controls)]:
            save(dict(stage=phase,anchor=k,time_unix=time.time(),command=cmd))
            with (OUT/f'anchor{k}_{phase}.log').open('w') as log:
                r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:
                save(dict(stage='failure',anchor=k,phase=phase,returncode=r.returncode));raise RuntimeError('Fast suite phase failed')
        # This is a completion milestone, not a decision to tune protocol using heldout.
        (OUT/f'anchor{k}_complete.json').write_text(json.dumps(dict(done=True,anchor=k,time_unix=time.time())))
    save(dict(stage='complete',time_unix=time.time()))

if __name__=='__main__':main()
