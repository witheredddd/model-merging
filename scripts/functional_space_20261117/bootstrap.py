"""Locate the verified full-domain geometry and R3 numerical primitives."""
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent.parent
ROOTS=[BASE,BASE/'_publish/a100-code-only-git/scripts']
CORE=next(p/n for p in ROOTS for n in ('fullspace_conditional_20261116','fullspace_conditional') if (p/n).exists())
LEGACY=next(p/n for p in ROOTS for n in ('eight_task_coupling_20261116','eight_task_coupling') if (p/n).exists())
R3=next(p/n for p in ROOTS for n in ('functional_control_r3_20261117','functional_control_r3') if (p/n).exists())
sys.path.extend([str(R3),str(CORE),str(LEGACY)])
