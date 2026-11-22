"""Resolve the checked-out project and the FP64 runtime modules."""
import os
import sys
from pathlib import Path


def configure_runtime():
    candidate = Path(__file__).resolve().parents[2]
    if not (candidate / "src").is_dir():
        candidate = Path("/home/shangxueyuan/merging/tsv")
    root = Path(os.environ.setdefault("TSV_ROOT", str(candidate))).resolve()
    scripts = Path(os.environ.get("TSV_RUNTIME_SCRIPTS", root / "scripts"))
    sys.path[:0] = [str(scripts / "eight_task_coupling_20261116"),
                    str(scripts / "fullspace_conditional_20261116")]
    return scripts
