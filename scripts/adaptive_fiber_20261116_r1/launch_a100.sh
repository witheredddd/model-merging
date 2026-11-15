#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
CODE=/home/shangxueyuan/merging/tsv/scripts/adaptive_fiber_20261116_r1
PY=/home/shangxueyuan/merging/afg_env/bin/python
OUT=/media/shangxueyuan/data1/adaptive_fiber_20261116_r1
mkdir -p "$OUT"
cd "$CODE"
"$PY" -m py_compile fiber_core.py run_a100.py run_comparators_a100.py
# This is an actual online-learning and model-evaluation program, not a post-hoc audit.
"$PY" -m pytest -q "$CODE"
# run_suite refuses to overwrite an already started suite. Do not launch twice.
"$PY" run_suite.py
