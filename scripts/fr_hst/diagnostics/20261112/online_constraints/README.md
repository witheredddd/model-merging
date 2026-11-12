# DTD online functional constraints — interim results, 2026-11-12

This snapshot contains six completed runs and one active run. `fisher_43` was queued but had not started. See `manifest.json` for the exact snapshot time and per-file SHA256 values. Training continued on the server after this snapshot; this archive is not the final eight-run result.

## Result

Checkpoints are selected by minimum validation cross-entropy, not maximum accuracy. Validation has 376 examples; no test results or model-merging results are claimed.

| Method | Seed | Completed epochs | Selected epoch | Selected accuracy | Selected CE |
|---|---:|---:|---:|---:|---:|
| AdamW | 42 | 8 | 8 | 83.24% | 0.6761 |
| AdamW | 43 | 8 | 7 | 84.31% | 0.6661 |
| Online constraints | 42 | 8 | 4 | 84.31% | 0.6630 |
| Online constraints | 43 | 8 | 5 | 83.24% | 0.6517 |
| Random direction | 42 | 8 | 4 | 81.38% | 0.7535 |
| Candidate-space Fisher preconditioner | 42 | 8 | 5 | 80.05% | 0.7160 |

Across the two completed seeds, selected accuracy averages 83.78% for both AdamW and online constraints. Online constraints have lower selected CE in both seeds, but there is no replicated accuracy advantage and two seeds do not establish statistical significance.

Online seed 42 deteriorates to 76.33% accuracy and CE 0.9321 at epoch 8. Its 193 accepted corrections have 24 independent-audit CE violations and zero FR violations. Thus bounded local FR changes do not establish stable long-horizon learning. Raw per-step records, including failures, are retained. `summary.json` provides corresponding counts for every archived run.

## Protocol

- Retained OpenAI CLIP ViT-B/16 initialization; the existing 48 full visual transformer matrices remain trainable. The classification head and remaining parameters are frozen. This is high-dimensional adaptation, with additional candidate-space control.
- Historical trained checkpoints and the historical classification head were unavailable following the earlier requested cleanup. The zero-shot head was rebuilt using retained CLIP weights and the repository's eight DTD prompts. Its hash is recorded in every config. Historical results are contextual references, not exact reproductions.
- The existing split uses 3384 training examples and 376 validation examples. From the former, reserve 32 for geometry discovery, 32 for replication and gating, and 32 for audit; use 3288 for SGD. Both discovery and replication sets gate corrections. Audit does not gate or select checkpoints. The same 32 audit examples are reused, so per-step violations are correlated observations, not independent trials.
- AdamW: learning rate 1e-5, weight decay 0.1, batch 64, eight epochs, 416 steps per run. Matched sample order and initialization within each seed.
- Every 16 steps, generate three hypothetical next AdamW displacements from different training minibatches at the same parameter and optimizer state. Compare them with the actual minibatch's proposed displacement; rescale contrast vectors to the actual displacement norm. No parameter-layer groups are prescribed.
- Compute a candidate-space Fisher Gram matrix using centered FP32 logit differences, softmax probability weighting, and FD coefficient 0.25. Repeat with half that coefficient on discovery data. Normalize the Gram matrix by its diagonal, then inspect its smallest eigenmode. Stability requires minimum eigenvalue below 0.5 on both sets, coefficient-direction absolute alignment above 0.8, and relative FD Gram discrepancy below 0.2. These are experimental thresholds, not grouping theorems or basis-invariant natural modules.
- Measure finite CE and squared Fisher–Rao distance for three individual contrasts, an equal joint contrast, and the inferred joint direction, at coefficients -1, -0.5, 0.5, 1. FR squared is the mean of `4 acos(sum sqrt(p*q))^2`. All measurements are retained in `geometry.jsonl`.
- Use a discovered direction only on subsequent training steps, up to the next refresh. Preserve the entire ordinary AdamW step and add a bounded correction. Its sign follows the current training gradient. Try amplitudes 0.5, 0.25, 0.125; reject unless both gating sets meet CE increase <=0.005 and FR squared <=0.0025 relative to the ordinary-step endpoint. The correction norm is bounded by the current ordinary update norm. Log independent audit values after acceptance.
- Random and Fisher controls use the same candidate construction and finite acceptance gates. Fisher here means a damped 3-by-3 candidate-space preconditioner, not a full-network natural-gradient implementation. They attempt corrections in every eligible window; online constraints act only after stable discoveries. Intervention counts and compute therefore differ and are reported, not treated as matched correction budgets.

## Reproduction and evidence

Run from repository root with the retained dataset and pretrained assets described in `RUNNING.md`:

```bash
python scripts/fr_hst/online_constraints.py --mode online --seed 42 --epochs 8 --output results/online_42
python summarize_online_constraints.py
```

`run_queue.sh` records all eight planned runs. It is the original server launcher and contains a server-specific working directory. `config.json` files record source/base/head hashes, sample indices, trainable parameter names, and all arguments. `validation.jsonl`, `geometry.jsonl`, and `controls.jsonl` retain epoch scores, finite probes, gates and audits. `complete.json` marks a finished run. The summarizer verifies archived file hashes before regenerating its output. Checkpoint weights, datasets, optimizer states and credentials are excluded.

This experiment tests one realization of the proposed local control hypothesis. It does not yet establish a transferable constraint representation, unique functional grouping, or cross-task merging benefit. The online mechanism refreshes a local direction; it does not train a persistent learned constraint predictor. No hyperparameter tuning or significance claim is included in this snapshot.
