from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent.parent
ROOTS=[BASE,BASE/'_publish/a100-code-only-git/scripts']
def locate(*names):return next(p/n for p in ROOTS for n in names if (p/n).exists())
CORE=locate('fullspace_conditional_20261116','fullspace_conditional')
LEGACY=locate('eight_task_coupling_20261116','eight_task_coupling')
R3=locate('functional_control_r3_20261117','functional_control_r3')
SPACE=locate('functional_space_20261117','functional_space')
sys.path.extend(map(str,[CORE,LEGACY,R3,SPACE]))
