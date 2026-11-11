from pathlib import Path
import json,collections,math,subprocess,time
disk=Path('/media/shangxueyuan/data1')
queue=disk/'fisher_rao_v5_medium_98f36a1_20261112.json'
out=dict(queue=json.loads(queue.read_text()) if queue.exists() else None,tasks={})
out['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
out['service']=subprocess.run(['systemctl','--user','is-active','fisher-rao-v5-two-task-medium-20261112.service'],capture_output=True,text=True).stdout.strip()
for task in ('DTD','SUN397'):
    root=disk/f'fisher_rao_online_v5_{task}_medium_seed11_98f36a1_20261112'
    row={}
    for key,file in [('checkpoint','recovery_status.json'),('status','status.json'),('resume','resume_status.json')]:
        if (root/file).exists():
            value=json.loads((root/file).read_text())
            if key=='status':value={k:v for k,v in value.items() if k!='latest_coordinate_event'}
            if key=='resume':value={k:v for k,v in value.items() if k!='args'}
            row[key]=value
    row['historical_ckpts']=len(list((root/'checkpoints').glob('*.pt')))
    if (root/'recovery_status.json').exists():row['checkpoint_age_seconds']=time.time()-(root/'recovery_status.json').stat().st_mtime
    if (root/'coordinate_events.json').exists():
        events=json.loads((root/'coordinate_events.json').read_text())
        row['committed_attempts']=len(events);row['accepted']=sum(e['accepted'] for e in events)
        row['fallbacks']=dict(collections.Counter(e['solver'].get('fallback') for e in events))
        errors=[e['solver'].get('parameter_step_reconstruction_relative_error',0.) for e in events]
        row['max_parameter_step_reconstruction_error']=max(errors,default=0.)
    if (root/'report.json').exists():
        report=json.loads((root/'report.json').read_text())
        row['last_completed_phase']={k:report[k] for k in ('status','stop_reason','task','base_validation','final_validation','independent_final','actual_online_images','seconds')}
    out['tasks'][task]=row
print(json.dumps(out))
