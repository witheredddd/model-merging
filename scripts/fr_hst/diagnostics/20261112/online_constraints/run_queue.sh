#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/merging-fr
out=results/ONLINE_CONSTRAINTS_20261112
for seed in 42 43; do
 for mode in adamw online random fisher; do
  name=${mode}_${seed}
  echo "START $name $(date -Is)"
  ./run-python -u scripts/fr_hst/online_constraints.py --mode "$mode" --seed "$seed" --epochs 8 --output "$out/$name" > "$out/$name.log" 2>&1
  echo "DONE $name $(date -Is)"
 done
done
echo ALL_COMPLETE
