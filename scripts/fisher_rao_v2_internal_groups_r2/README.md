# Fisher–Rao v2: single-task internal functional groups

Executed DTD prototype, 2026-11-12. The nine Python source files match the hashes in [`evidence/20261112/run_config.json`](evidence/20261112/run_config.json). The supplied replacement specification is in [`EXECUTION_SPEC.md`](EXECUTION_SPEC.md).

The model is generated as `theta_w + S C sum_m U_m z_m`. Candidate directions come from eight training minibatches and four early warmup windows of the same task; every response is recomputed at the fixed warm anchor. Full expert endpoints do not initialize the groups. Functional groups and prediction fibers are separate objects. There is no task-clustering initialization.

## Result and interpretation

The development run completed with **M=1**. No multi-group proposal passed the structural screening, so A–D reduced to the same single-group model. Multi-group superiority and compensated deletion/merge necessity were therefore **not established**, rather than disproved by a completed multi-group comparison. Fiber learning, eight-task transfer and EC remained disabled. No official test predictions were read.

| Independent seed | Warm Vcal CE | Selected Vcal CE | Xprime accuracy | Accepted low-dimensional updates |
|---|---:|---:|---:|---:|
| 37 | 0.9603 | 0.9360 | 73.44% | 3 |
| 53 | 0.9902 | 0.9210 | 75.78% | 22 |

Both optimizers stopped before the 50-update limit after a minibatch backtracking failure. The initial 12-dimensional dictionary retains approximately all of the sampled candidate response energy; its rank-two functional chart retains approximately 55–67%. Normalized endpoint FR errors against the matched-forward-budget AdamW reference are approximately 0.970 and 0.867 for seeds 37 and 53. These results do not establish high-quality expert reconstruction or a stable multi-group decomposition.

Raw outcomes: [`development_report.json`](evidence/20261112/development_report.json), [`readout_audit.json`](evidence/20261112/readout_audit.json). Unsupported group candidates can contain a default stability value; interpret it together with their support and qualification flags, not as an observed bootstrap success rate.

## Files

- `runtime_task.py`: one task, fixed head/splits, generated-weight forward, JVP/VJP.
- `structure.py`: same-anchor candidate response grouping, reachability, overlap and independent-view checks.
- `learn.py`: joint group/coordinate optimization, equal-rank controls, compensated deletion/merge, AdamW reference and actual response trajectories.
- `pipeline.py`: warmup, development selection, then independent-seed validation.
- `fr_math.py`: Fisher–Rao sphere functions and damped lift.
- `fiber_stage.py`: deferred single-task wrapper around the frozen correction helpers; not exercised in this run.
- `ec_stage.py`: deferred group-weight multiobjective search; refuses to run without eight validated compatible tasks.

`../fisher_rao_v2/{common,fiber,horizontal}.py` and `../fisher_rao_v1/geometry.py` are frozen numerical dependencies of the optional fiber wrapper. They are not the entry points of this replacement design. The existing repository supplies `scripts/dtd_grassmann4.py`, `scripts/tfpml/runtime.py`, `scripts/fr_hst/task_data.py` and the CLIP/dataset modules.

## Reproduce with the existing A100 asset layout

The code requires the installed CLIP ViT-B/16 checkpoint, fixed classification head and dataset files. Weights and datasets are not in this commit. DTD uses the installed 3760-image mirror training pool, reserves 1536 development images and trains on 2224; Oxford official fold identity was not independently recovered.

The runtime reuses only the historical split manifest and asset hashes. A checkout without the old results can restore that small manifest without obtaining any expert checkpoint:

```bash
mkdir -p results/fisher_rao_v1_20261111/DTD
cp scripts/fisher_rao_v2_internal_groups_r2/evidence/20261112/DTD_split_manifest.json \
  results/fisher_rao_v1_20261111/DTD/manifest.json
python -m unittest discover -s scripts/fisher_rao_v2_internal_groups_r2 -v
python -u scripts/fisher_rao_v2_internal_groups_r2/pipeline.py \
  --root "$PWD" --output /path/to/new-results --task DTD \
  --warmup 100 --steps 50 --budget 8000 --candidates 8 \
  --seeds 11 23 37 53 --X-n 512
```

Use a fresh output directory for a different code/config hash. `launch_r2.sh` preserves the actual server paths. [`RUNNING.md`](RUNNING.md) is the historical deployment record; its initial running status is superseded by the completed report above. Eleven tests and the isolated real-CLIP smoke run passed; these validate interfaces and derivatives, not the scientific multi-group hypothesis.
