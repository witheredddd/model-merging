# Unattended DTD/Cars parameter-substructure campaign

The controller runs on the A100 host independently of Codex, the desktop, or an SSH connection. It supervises one new experimental process at a time, saves phase status, retries interrupted jobs from immutable inputs, runs complements and expert-increment baselines, generates reports and indexes data. Original unrelated jobs are not controlled by this program.

## Order

1. Wait for an already running pilot, or start/resume it if missing/incomplete.
2. Run all resolved-rank Euclidean motion complements and expert coefficient baselines.
3. Generate per-anchor metric tables, numerical deletion reports and a report archive.
4. Generate disjoint larger probes for seeds 41, 42 and 43, then run the expanded profile across all 17 true positions, serially.
5. For each seed, run the supplement and generate per-anchor reports, array manifests and `RESULT_PACKAGE/REPORTS.tar.gz`.

`CAMPAIGN_STATUS.json` distinguishes `RUNNING`, `WAITING_FOR_STORAGE`, `PAUSED`, `FAILED` and `COMPLETE`. Job logs, node completion markers and failure records remain. Three failed attempts cause explicit failure; the controller does not mark skipped or failed experiments complete. Source hashes and the plan are frozen; documented migrations are explicit. A successful node is checkpointed before moving on. Resume uses the same command and output directory.

The server user has `Linger=yes`. A regular enabled user service, `task-substructure-campaign-20261120.service`, launches the campaign; the laptop can shut down. On server reboot, the enabled service can resume the queue, subject to unchanged inputs, available GPU and disk. This is not a ChatGPT automation and does not rely on messages or callbacks to the desktop.

Example command (replace paths):

```bash
python task_substructure_campaign.py \
  --root /data/task_substructure_campaign --pilot-output /data/task_substructure_pilot \
  --pilot-scripts /repo/scripts/task_substructure_20261120 \
  --large-scripts /repo/scripts/task_substructure_large_20261120 \
  --helpers /repo/scripts/task_substructure_campaign_20261120 \
  --inventory /resources/INVENTORY.json --probes /resources/probes.json \
  --coordinates /resources/config.json --trace /resources/trace \
  --flow-source /resources/prior_DTD_flow --seeds 41 42 43
```

`--skip-large` is a smoke-test option, not used in the deployed campaign.

## Reports and interpretation

`analysis/anchor_*/all_metrics.csv` contains every scalar case metric; `per_class_part*.csv.gz` contains every class outcome in bounded shards. Separate files cover common-objective finite tests, conditional driver curves, deletion gaps, all native regions, all resolved ranks, signed compatibility and selection-versus-final evaluation. Raw case JSON and NPZ retain actual logits and functional responses. `SUMMARY.json`, `REPORT.md` and `analysis/INDEX.json` describe coverage, assumptions and unfinished claims.

The result package holds reports and SHA256 indexes. Full numerical arrays and model vectors stay on the A100; the reports archive is not a complete self-contained model or data backup. Image IDs are not packaged. This program does not require Git credentials or internet access, and does not promise automatic Git upload; publication can use the generated packages later.

This campaign completes the proposed local sufficiency/substitution/compatibility experiment. It does not silently add full expert training or EC evolution: Val >=80% under controlled training and superiority of a real EC Pareto front remain separate, untested end-to-end claims. Spectral rank is bounded by the demand family and is not evidence of intrinsic low dimension. Constructed-gradient null complements, proxy cost reductions and finite local improvements alone do not establish necessary physical groups or a manifold.

After each large run's last-anchor report, the aggregate tool rebuilds `analysis/resampling_metrics.csv`, sampled CE hypervolumes and support-membership reproducibility across available seeds. Missing seeds are not imputed. Hypervolume references `(0,0)` and `(0.01,0.01)` are predeclared CE-change references; no-op is added to every budget-feasible family. Original-scale expert controls have a different parameter budget and remain separate. These summaries are resampling statistics and sampled local geometry, not independent training trials or EC superiority. Physical radii are matched within each resampling; their numerical values can differ across seeds and remain in the tables.

Validation: seven numerical/production regression tests, two real high-dimensional control reconstructions, actual finite-model evaluations and a mocked end-to-end campaign covering supplementary jobs, reports, node completion and packaging. Live progress must be checked independently of these tests.
