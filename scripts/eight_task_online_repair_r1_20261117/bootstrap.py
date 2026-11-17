from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent.parent
CORE=BASE/'fullspace_conditional_20261116'
LEGACY=BASE/'eight_task_coupling_20261116'
if not CORE.exists():CORE=BASE/'fullspace_conditional'
if not LEGACY.exists():LEGACY=BASE/'eight_task_coupling'
sys.path.extend([str(CORE),str(LEGACY)])
