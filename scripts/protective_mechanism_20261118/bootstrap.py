from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent.parent
ROOTS=[BASE,BASE/'_publish/a100-code-only-git/scripts']
def find(*names):return next(p/n for p in ROOTS for n in names if (p/n).exists())
CORE=find('fullspace_conditional_20261116','fullspace_conditional')
LEGACY=find('eight_task_coupling_20261116','eight_task_coupling')
R3=find('functional_control_r3_20261117','functional_control_r3')
sys.path.extend(map(str,[R3,CORE,LEGACY]))
