from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent.parent
ROOTS=[BASE,BASE/'_publish/a100-code-only-git/scripts']
CORE=next(p/name for p in ROOTS for name in ('fullspace_conditional_20261116','fullspace_conditional') if (p/name).exists())
LEGACY=next(p/name for p in ROOTS for name in ('eight_task_coupling_20261116','eight_task_coupling') if (p/name).exists())
OLD=next((p/name for p in ROOTS for name in ('eight_task_online_repair_r2_20261117','eight_task_online_repair') if (p/name).exists()),BASE/'eight_task_online_repair')
sys.path.extend([str(CORE),str(LEGACY),str(OLD)])
