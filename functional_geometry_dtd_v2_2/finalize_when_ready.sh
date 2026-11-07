#!/usr/bin/env bash
set -euo pipefail
cd /home/shangxueyuan/merging/tsv/scripts/functional_geometry_dtd_v2_2
fg_result_root=/home/shangxueyuan/merging/tsv/results/functional_geometry_dtd_v2_2
fg_python=/home/shangxueyuan/merging/afg_env/bin/python
exec 9>"$fg_result_root/postprocessing.lock"
flock -n 9 || exit 0
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
while [ ! -f "$fg_result_root/result.json" ]; do
    if [ -f "$fg_result_root/failed_gate.json" ]; then exit 1; fi
    sleep 10
done
"$fg_python" provenance.py
"$fg_python" reconstruct.py
"$fg_python" span_check.py
"$fg_python" report.py
"$fg_python" -c 'import json,time;from pathlib import Path;p=Path("/home/shangxueyuan/merging/tsv/results/functional_geometry_dtd_v2_2/postprocessing_complete.json");p.write_text(json.dumps({"complete":True,"time":time.time(),"reconstruction_verified":True,"report":"REPORT.md","evidence":"evidence.zip"},indent=2))'
