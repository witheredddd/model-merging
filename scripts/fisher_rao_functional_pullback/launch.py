from pathlib import Path
import subprocess,json,shutil
root=Path('/home/shangxueyuan/merging/tsv');disk=Path('/media/shangxueyuan/data1')
source=root/'scripts/functional_pullback_phase1_20261113'
output=disk/'DTD_functional_pullback_phase1_20261113'
python='/home/shangxueyuan/merging/afg_env/bin/python'
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if free<32768:raise RuntimeError('Less than 32GiB free; preserve existing job, do not launch DTD')
base=[python,'-u',str(source/'run.py'),'--root',str(root),'--output',str(output)]
subprocess.run(base+['--steps','44'],check=True)
report=json.loads((output/'report.json').read_text())
shutil.copy2(output/'report.json',output/'pilot_report.json')
if report['status']!='complete' or report['horizontal_accepted']<1:
    raise RuntimeError('Horizontal pilot did not pass; no automatic continuation')
subprocess.run(base+['--steps','128','--resume'],check=True)
