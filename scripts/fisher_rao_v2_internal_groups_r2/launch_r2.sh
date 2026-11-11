#!/bin/bash
set -euo pipefail
cd /home/shangxueyuan/merging/tsv
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
exec /home/shangxueyuan/merging/afg_env/bin/python -u scripts/fisher_rao_v2_internal_groups_r2/pipeline.py \
  --output /media/shangxueyuan/data1/fisher_rao_v2_internal_groups_20261112_r2 \
  --task DTD --warmup 100 --steps 50 --budget 8000 --candidates 8 --seeds 11 23 37 53 --X-n 512
