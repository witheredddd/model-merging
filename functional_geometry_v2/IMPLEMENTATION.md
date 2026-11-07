# Functional Geometry EC v2.1

This implementation follows the supplied v2 no-expert specification. All old results remain separate. The pretrained checkpoint is read-only. No old expert or expert delta is an input.

Run `prepare.py`, then `python -c "from engine import Engine,cold; cold(Engine())"`, then `python supervise.py` on the configured server. `supervise.py` requires Linux flock and starts only one child pipeline. It resumes training, performs final geometry analysis, runs the small EC pilot, locks choices, and evaluates tests. `monitor/latest.json` and timestamped reports persist hourly diagnoses. A successful process exit does not imply passing the DTD accuracy or geometry gates.

The running deployment is managed by the user service `functional-geometry-v2.service`, with `Restart=on-failure` and `KillMode=control-group`. The monitor itself is therefore restarted if it crashes, and old worker processes are stopped before a service restart. Check with `systemctl --user status functional-geometry-v2.service`. Hourly diagnosis does not pause training. This is a job-specific server service, independent of the local Codex or SSH connection.

## Executable algorithm

1. Verify all eight train/validation datasets. Build frozen zero-shot heads from the pretrained text encoder using existing templates and dataset class order.
2. At theta0, collect four normalized full-parameter gradients per task. Form the 32-by-32 Gram in CPU FP64 chunks. Orthogonalize up to eight task means, then add balanced residual snapshots until rank 12. Save the basis and real predictive Fisher matrices at theta0.
3. Initialize eight coordinate vectors to zero. In each of 2000 rounds, update each task once: reconstruct theta0+Bz, compute full CE gradient, project into current B, and update only that task's coordinates with its independent Adam.
4. Every 50 rounds validate each expert, select minimum validation CE, and save checkpoints. Every 16 rounds consider expansion for tasks with two stalled validation checks and insufficient projected gradient. Construct residual using current task training data excluding the candidate's held-out training indices. A backtracked unit-residual step must improve both construction and independent training CE. Measure all heads' loss/logit/KL responses at the SAME common reference. Append the orthogonal residual with all task coordinates zero; check all expert reconstructions remain identical. No isolated-task requirement is imposed.
5. Preserve rank-32 artifacts before any permitted increase to 48. If capacity remains insufficient, keep optimizing and log the failure. Do not lower the DTD 80% gate.
6. With the final shared basis, compute eight predictive Fisher matrices at each of theta0, the saved common intermediate reference, and the selected experts' mean. Calibration, holdout and third audit use disjoint training images. Keep CE responses signed and separate from Fisher.
7. Analyze symmetric traceless commutant spectra. Fit candidate partitions using calibration matrices only; apply independent gates, per-image bootstrap and independently rotated task nulls. Report no stable common block when gates fail. No module boundaries or predefined block count enter this process.
8. Reconstruct expert updates from coordinates. When a partition is validated, form block projections; otherwise use a global coefficient per expert. Check the one-hot coordinate reconstruction. Run the explicitly exploratory coefficient-search pilot on a fixed validation subset, lock selected coordinates, then evaluate experts and the locked merge on test once, caching completed outputs.

## Deliberate engineering changes

- The supplied core ranked candidates with `train_error + heldout_error`. v2.1 ranks using training error and eigengap only, then applies the holdout gate. Original source is retained as `functional_commutant_core.original.py`.
- Reserve 48 columns in GPU memory while the initial active limit is 32. This keeps reconstruction GEMV shape constant when a column is appended and supports an exact no-position-jump check. Reserved capacity is not the active rank or a claim of a 48-dimensional optimum.
- Recoverable truncated-image decoding is enabled per process, preserving original file bytes and split membership. `audit_images.py --all` records strict decoding failures, hashes and recoverability. It never silently removes examples.
- No optional span-changing Grassmann rotation is used. Append-only tracking preserves old models and the saved intermediate reference exactly in coordinate form.
- EuroSAT uses the mirror's explicit 16,200/5,400/5,400 train/validation/test split. Exact byte-hash intersections across its splits are empty. A stale generic seed-0 sentence written by the legacy helper is corrected in `assets/EuroSAT_split_audit.json`; historical split equivalence is not claimed.
- The 2000 balanced updates/task are an initial overnight budget, not equal epochs across datasets. Report actual epochs and validation curves; do not describe all tasks as fully optimized solely because this budget finishes.

## Cost and limitations

D=84,934,656. A float32 direction is 339,738,624 bytes; rank 12 is 4.08 GB, rank 32 is 10.87 GB, and reserved rank 48 is 16.31 GB (decimal). Cold gradients occupy about 10.87 GB CPU memory. Chunked Gram construction costs O(D*32^2), and initial basis construction O(D*32*12), without a D-by-D matrix. A training step uses O(D*r) projections plus the ordinary backbone backward pass. Adam states are only O(8*r), but full gradients and the dense basis still dominate memory. Runtime logs contain actual peak GPU memory and step timings.

Predictive Fisher costs O(8*r*n) directional forward evaluations per reference and O(8*r^2) stored mean matrices. Three independent sets of eight images/task are a compute-limited first estimate. Per-image matrices are retained for bootstrap. These samples may be too noisy to detect stable structure; bootstrap and null results cannot establish global functional isolation. The dense symmetric traceless commutant operator scales quadratically in its O(r^2) unknown count, so geometry analysis becomes costly as rank grows.

The EC pilot is small and exploratory. No full-finetuning comparator was added. DTD below 80% remains a failure. Validation/test comparisons from historical runs must not be silently mixed with this run's exact split.

Final artifacts: `result.json`, `geometry.json`, `history.json`, `config.json`, `basis_latest.pt`, `selected_experts.pt`, `merged_coordinates.pt`, functional matrices/analyses, structure events, and monitoring reports under `results/functional_geometry_v2_no_expert`.
