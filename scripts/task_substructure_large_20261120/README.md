# Expanded repeated task-substructure experiment

Use with the pilot and the server campaign controller. The underlying objectives and numerical scope are documented in the [pilot method](../task_substructure_20261120/README.md).

## Predeclared expansion

- Three independently seeded **data resamplings** (41, 42, 43), not independent retraining trials.
- All 17 actual training positions: 0, 25, 27, 50, 54, 75, 81, 100, 108, 125, 135, 150, 162, 175, 189, 200, 216.
- DTD: 940 construction, 940 selection and 940 final-evaluation images. Cars: 980 in each role. Within each run, all three image sets are disjoint and class-stratified. Seeds can overlap each other; experts may have trained on them. No official test images are opened.
- Fifteen construction label-weight demands (mean, positive sine/cosine frequencies 1–7); four new demands (frequencies 8 and 9) plus mean on independent images. Output every resolved rank, the family-centered spectrum and the mean-CE-removed spectrum. Fifteen is an input-family rank bound, not an intrinsic-dimension or space-cap claim.
- At every position, compare frozen anchor-0 motions, newly constructed local motions, mean CE, full gradients, existing Fisher coordinates and entire motion complements. Frozen physical supports from anchor 0 are compared with locally selected supports.
- Fisher/Schur comparisons at positions 0, 25, 100 and 216 include multiple prediction/parameter cost ratios, full complements, naive freezing, independently normalized directions and common-objective finite actions.
- For each tested forbidden component, evaluate fixed-driver amplitudes `-1,-0.5,0,0.5,1,1.5,2`. If `v_full` and `v_freeze` solve the same quadratic, the conditional optimum is `t*v_full+(1-t)*v_freeze`; compare it with `t*(I-P)*v_full+v_freeze`. Both actions in each pair have identical driver motion; a common scaling keeps every action within the same radius bound. Record real DTD/Cars effects and conditional residual bounds. The entire complement participates; no local graph must individually fulfill a task.

## Storage and reproducibility

Raw original checkpoints, computed gradients, the frozen anchor-0 basis and solver motions are retained. Temporary finite-intervention controls and later-position bases are archived after their computations finish: their construction inputs, eigen-coefficients, both physical orthogonalization passes, normalization scalars, source, original hashes, actual functional arrays and case metrics remain. This avoids storing redundant full-size vectors at all 51 seed/position combinations. It only affects **this new diagnostic**; old functional-flow trajectories are untouched.

`reconstruct.py ROOT CASE DESTINATION` rebuilds a control, including archived local bases, on CPU. CPU/GPU reduction order can differ in last bits; original file hashes are provenance rather than a bitwise cross-device promise. Two actual 86-million-dimensional controls were reconstructed with relative error about `1e-13`. External expert controls also require the frozen original checkpoints. `CONTROL_ARCHIVE_a*.json`, `BASIS_ARCHIVE_a*.json` and case JSON distinguish retained and recipe-based vectors.

The expanded basis is orthogonalized in actual parameter coordinates with two passes; its physical Gram error is reported. A production regression deletes a synthetic saved basis and reconstructs a control through its retained numerical recipe. Every case also audits FP32 parameter rounding (changed-coordinate fraction, rounded-step norm and relative parameter rounding error). The actual model metrics remain FP64; this audit does not constitute an FP32 forward-performance test.

The supervisor starts a job only with at least 120 GiB free. The worker checks a 30 GiB reserve before allocating another case. Low space records a resumable resource hold; it does not reduce dimension, truncate a demand family or delete source data. A large campaign may take days and can require additional disk space; progress and resource holds are explicit.

`ANCHOR_COMPLETE_*.json` allows an interrupted run to skip finished anchors. The supplement has its own completion marker. `COMPLETE` requires the worker, supplement and per-anchor reports to finish. Negative and uncertain results remain in all tables.

Per-evaluation timing fields and append-only `TIMING_HISTORY.jsonl` carry process IDs. Combine segments across restarts instead of interpreting one process's final operator counters as the complete campaign cost. Reported J/JT timings are host-measured; case wall time includes synchronous output transfer and metric computation.
