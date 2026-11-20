# Re-audit of commit `04047a0` — what changed and what it changes

This directory now contains, on top of the original step-0 upload:

- `code/step0_reaudit.py`, `code/reaudit_math.py`, `code/patch_solver_lambda.py`
  and their tests (the supplied audit patch);
- `code/schur_resumable.py`, `code/run_step0.py`, `code/test_schur_resumable.py`
  with the implementation fixes below;
- `results/step0_anchor0/reaudit_step0/` — the audit's own outputs.

The original commit `04047a0` and its manifest stay valid for what they contain.
`HASHES.sha256` in this commit covers the new file set.

## Changes I made to the supplied patch

One line, because the audit could not run without it:

```python
# step0_reaudit.py
- raw_modes = torch.load(root/'candidate_modes.pt', ..., weights_only=True)
+ raw_modes = torch.load(root/'candidate_modes.pt', ..., weights_only=False)
```

`candidate_modes.pt` also stores numpy eigenvalue arrays, which the `weights_only`
unpickler rejects on torch 2.8. The file is produced by this experiment itself.
Nothing else in the patch was modified. Its 8 tests pass unchanged.

## Implementation fixes (applied in place, not re-run)

1. **Objective-gap bound for general `lambda`.** `0.5*||r||^2` is an upper bound on
   the optimum gap only at `lambda = 1`. Written as `0.5*||r||^2/lambda` in the
   bound, the stopping rule and the report. At the running `lambda = 1` the
   published numbers are unchanged.
2. **Checkpoint identity, fail-closed.** Each solve now records
   `solves/<label>.npz.identity.json` holding the task, probe, anchor, `lambda`,
   `beta`, the gradient digest, the active mask digest, the mode digests of the
   forbidden subspace, the parameter-point digest and the geometry's output
   coordinate id. A checkpoint is reused only on an exact match; an unidentified
   checkpoint raises instead of being loaded. `test_checkpoint_identity_fails_closed`
   pins all three failure modes.
   Consequence: the existing `step0_anchor0/solves/*.npz` have no identity record,
   so re-running that directory now refuses to reuse them. That is intended.

## What the re-audit changes

Run on the A100 against the original artifacts, no new CG:

```
python step0_reaudit.py --output .../step0_anchor0 \
  --inventory ... --probes ... --trace ... --functional-radii 0.0025 0.01 0.04
```

### 1. The pooled-motion identity holds

Weighting the four stored motions by their image counts (59/47/47/47) gives a
motion whose projected stationarity is `3.57e-05` — the four demands share one `H`,
so this is a strict identity, and it confirms the stored motions belong to one
common quadratic.

### 2. The calibration really was misaligned

`kappa` from the full 86M gradient: **553.6407**. `kappa` from the active 48-matrix
gradient: **317.1943**. The `beta` used to build `H` was derived from the former
while every solve was restricted to the latter. The audit keeps the original
`beta` (so the old solutions stay interpretable) and reports the active value as
an audit quantity only.

### 3. Sufficiency, measured directly (probe 2, `full_q = -454.424`)

How much of the selection demand's full quadratic gain a span achieves on its own.
This is a different quantity from the Schur deletion; they must not be substituted.

| span | rank | `q(v_R*)` | retention `q_R/q_full` |
|---|---:|---:|---:|
| candidate (H-metric) | 1 | -5.753 | 1.27% |
| candidate (H-metric) | 2 | -8.996 | 1.98% |
| candidate (H-metric) | 3 | -9.752 | 2.15% |
| mean gradient, raw | 1 | -13.635 | **3.00%** |
| mean regularized motion | 1 | -11.580 | 2.55% |
| mean regularized + candidate | 2 | -11.851 | 2.61% |
| mean regularized + candidate | 3 | -11.852 | 2.61% |

Adding the candidate directions to the mean-regularized direction moves retention
from 2.55% to 2.61% — the candidate space contributes essentially nothing on a
fresh demand, by its own measure.

### 4. Real CE at matched small radius (probe 3, held out)

Baseline is real model output now, not the sentinel row: probe 2 `CE=2.1151,
Top1=0.4245`; probe 3 `CE=2.2089, Top1=0.3936`.

Every arm at every radius **decreases** CE. The earlier `+5 nats` table was
entirely a step-size artifact: `||v|| = 28` against a calibrated radius of
`1.06e-4`.

Equal physical radius, probe 3, `delta CE` in nats:

| arm | r=0.0025 | r=0.01 | r=0.04 |
|---|---:|---:|---:|
| mean gradient raw | **-5.43e-04** | **-2.16e-03** | **-8.55e-03** |
| mean regularized | -4.67e-04 | -1.87e-03 | -7.42e-03 |
| mean reg + candidate r2 | -4.67e-04 | -1.87e-03 | -7.42e-03 |
| candidate r3 | -4.41e-04 | -1.76e-03 | -7.01e-03 |
| candidate r2 | -3.76e-04 | -1.50e-03 | -5.97e-03 |
| candidate r1 | -2.85e-04 | -1.14e-03 | -4.52e-03 |
| g0 full optimum | -2.93e-04 | -1.17e-03 | -4.64e-03 |
| g0 half joint | -2.53e-04 | -1.01e-03 | -4.00e-03 |
| g0 restricted | -2.06e-04 | -8.20e-04 | -3.25e-03 |

At `r = 0.04` the common-tau protocol gives full `-4.64e-03`, half-joint
`-3.88e-03`, restricted `-3.12e-03`: the half-amplitude compensated update lands
halfway, as the quadratic predicts. That is a model prediction holding on real CE,
not an independent discovery.

## The conclusion that survives

- The Schur core is not wrong, and the `lambda = 1` quadratic solves remain valid.
- **The candidate low-rank movement space is not special.** It loses to a single
  mean-gradient direction on both the quadratic retention (2.15% vs 3.00%) and the
  measured CE at every matched radius, and it adds nothing on top of the
  mean-regularized direction.
- Rank selection by deletion is invalid (deletion is monotone in rank); the
  original `selected_rank = 2` had no adequate basis. Sufficiency was the missing
  quantity and it is now reported.
- Nothing here establishes a physical parameter support, an intrinsic dimension or
  a reusable movement basis.

## Still open

- `random_coordinates` is a dense isotropic direction, a weak null control; it is
  now labelled as such and should be replaced by a rank-matched H-metric
  `mean + direction` control in any follow-up.
- The demand gradients and their cosines were computed on the full 86M vector,
  while the solves live in the 48-matrix domain. `cos(P_D g_i, P_D g_j)` has not
  been reported yet.
- The 200-image probe split 59/47/47/47 and the near-flat H spectrum cannot
  distinguish "the task genuinely needs many directions" from "four small-batch
  gradient estimates are mostly noise".
- No `lambda`/`beta` change, new candidate basis, new anchor or new CG was run.
  Any of those needs a fresh output directory and a full re-solve.
