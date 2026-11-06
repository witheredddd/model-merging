#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/shangxueyuan/merging/tsv
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1
exec 9>"$ROOT/results/TFPML_v3_expression.lock"
flock -n 9 || { echo 'v3 expression training already running'; exit 1; }
exec /home/shangxueyuan/merging/afg_env/bin/python \
  "$ROOT/scripts/tfpml/train_expression.py" --epochs 20 --frequencies 16 --d 3 --evaluate-test
