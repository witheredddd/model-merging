"""Sequential bounded task checks and 128-attempt runs; no million-budget auto-run."""
from pathlib import Path
import subprocess,json,math,time,shutil,signal,sys

ROOT=Path('/home/shangxueyuan/merging/tsv')
SOURCE=ROOT/'scripts/fisher_rao_online_v5_98f36a1'
DISK=Path('/media/shangxueyuan/data1')
STATE=DISK/'fisher_rao_v5_medium_98f36a1_20261112.json'
PYTHON='/home/shangxueyuan/merging/afg_env/bin/python'
OUTPUTS={task:DISK/f'fisher_rao_online_v5_{task}_medium_seed11_98f36a1_20261112' for task in ('DTD','SUN397')}
status=dict(source_commit='98f36a1b5135bce79be21b46e0024c22629d683d',stage='starting',tasks={})
child=None;stopping=False

def write():
    status['updated_unix']=time.time();temp=STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(status,indent=2));temp.replace(STATE)

def terminate(number,frame):
    global stopping
    stopping=True
    if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)

signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)

def run(task,steps,resume=False):
    global child
    output=OUTPUTS[task]
    command=[PYTHON,'-u',str(SOURCE/'runner.py'),'--task',task,'--root',str(ROOT),'--output',str(output),
        '--steps',str(steps),'--budget-images','1000000','--seed','11','--probe-size','128','--batch-size','32',
        '--geometry-every','8','--lookahead-steps','2','--report-every','8','--report-images','4096','--checkpoint-every','8']
    if resume:command+=['--resume',str(output/'recovery_state.pt')]
    status.update(stage='medium' if resume else 'adapter_preflight',active_task=task)
    status['tasks'][task]=dict(stage=status['stage'],target_attempts=steps,output=str(output));write()
    print(json.dumps(dict(starting=task,resume=resume,steps=steps,command=command)),flush=True)
    child=subprocess.Popen(command);result=child.wait();child=None
    if stopping:raise RuntimeError('Stop requested; no next task will start')
    if result:raise RuntimeError(f'{task} runner exited {result}')
    report=json.loads((output/'report.json').read_text());events=json.loads((output/'coordinate_events.json').read_text())
    if report['status']!='complete' or len(events)!=steps:raise RuntimeError(f'{task} did not reach requested attempt count')
    if not all(math.isfinite(report[k]['CE']) for k in ('base_validation','final_validation')):
        raise RuntimeError(f'{task} nonfinite task loss')
    if not any(e['accepted'] for e in events):raise RuntimeError(f'{task} no accepted parameter write')
    for event in events:
        value=event['solver'].get('parameter_step_reconstruction_relative_error',0.)
        if not math.isfinite(value) or value>1e-6:raise RuntimeError(f'{task} unstable coordinate reconstruction')
    status['tasks'][task].update(stage='medium_complete' if resume else 'adapter_passed',
        accepted=sum(e['accepted'] for e in events),online_images=report['actual_online_images'],
        validation=report['final_validation'],resume_verified=(output/'resume_status.json').exists());write()
    if not resume:
        for name in ('report','coordinate_events','geometry_events','recovery_status'):
            shutil.copy2(output/f'{name}.json',output/f'preflight_{name}.json')

try:
    write()
    for task in OUTPUTS:run(task,8)
    for task in OUTPUTS:run(task,128,resume=True)
    status.update(stage='bounded_medium_complete',active_task=None,long_training_started=False);write()
except Exception as error:
    status.update(stage='stopped' if stopping else 'failed',error=str(error),long_training_started=False);write()
    print(json.dumps(status),flush=True);sys.exit(1)
