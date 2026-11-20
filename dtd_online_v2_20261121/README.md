# DTD online functional control — V2 / V2.1 and the radius gate (2026-11-21)

Second round on the online controller. V1 was uploaded at commit `cba2a095` in
`dtd_online_v1_20261120/`. This round adds the budget-consistent fix (V2), the
protocol instrumentation (V2.1), a two-step three-arm run on each, and the
predeclared radius calibration (Stage B).

**Result in one line:** the mechanism runs and every invariant holds; V2.1
reproduces V2's parameter trajectory bit for bit; and the predeclared radius grid
admits no value, because the first-order Jacobian stops describing the model well
before any of the four candidate radii.

## Provenance

| item | value |
|---|---|
| machine | `10.159.23.24`, 1x A100-80GB, float64 |
| anchor | recorded normal-training step 0, `DTD_DYNAMIC_TRACE_20261109` |
| probes | p0 200 images (train + Jacobian), p2 106 (adaptive selection), p3 94 (opened once, after the loop) |
| prior round | `dtd_online_v1_20261120/` (V1, unchanged upstream) |
| V2 | supplied as a patch over V1: target from the budgeted step, trust-region fit at the same radius, shared backtracking, corrected reason labels, iteration default 96 |
| V2.1 | this directory's `code/`: V2 plus instrumentation and a same-scale birth rule (`code/V2_1_NOTES.md`) |

## Stage A — two steps, three arms

Same anchor, same seed 42, same frozen probes, `radius = 1e-4`, `max-modes 8`.
The **radius is an execution budget, not a training scale**; see Stage B.

Paired on the identical 94 p3 images:

| arm | rank | mean delta CE vs anchor | SE | 95% CI | images better / worse | written norm per step |
|---|---:|---:|---:|---|---:|---|
| online | 2 | −1.1416e-04 | 8.3e-05 | [−2.77e-04, +4.86e-05] | 55 / 39 | 5.17e-05, 5.85e-05 |
| frozen | 1 | −6.3174e-05 | 8.0e-05 | [−2.19e-04, +9.30e-05] | 49 / 45 | 5.17e-05, 3.93e-05 |
| reference | — | −1.1568e-04 | 8.0e-05 | [−2.72e-04, +4.10e-05] | 57 / 37 | 1.00e-04, 1.00e-04 |

| paired difference | estimate | t | 95% CI |
|---|---:|---:|---|
| online − reference | +1.52e-06 | +0.27 | [−9.7e-06, +1.3e-05] |
| online − frozen | −5.10e-05 | −1.80 | [−1.07e-04, +4.6e-06] |
| frozen − reference | +5.25e-05 | +1.91 | [−1.5e-06, +1.06e-04] |

No interval excludes zero and Top-1 is unchanged (37/94) in every arm. The
reference writes the full budget every step; the controller writes about 55% of
it, so this is not a like-for-like delivered-motion comparison, and the protocol
records `write_norm` and `executed_delivery_error` for exactly that reason.

### Invariants (all six hold in all three arms)

| invariant | evidence |
|---|---|
| `target_update_norm <= radius` | 1.000e-04 both steps |
| every fit and write inside the radius | write ≤ 5.85e-05; a violated fit raises |
| executed error equals the written step's error | `executed_delivery_error` reproduces `‖Jδ−y‖/‖y‖` to 1e-15 |
| no write outside the span | `actual_outside_mode_span` ≤ 9.5e-22 |
| frozen reason is distinct | `frozen_by_policy` with `arm_policy.frozen=true`, and the residual at that step was 0.735, so the freeze really was the cause |
| reference backtracking is shared | both arms call `guarded_backtrack` with the same ladder and accept rule |

### V2.1 reproduces V2 exactly

The V2.1 changes are instrumentation and the birth-evidence rule. Running the
same three arms again gives **bit-identical parameter vectors** for all three
arms (`torch.equal` on the 86,192,640-element `theta`, max abs difference 0.0)
and identical p3 deltas to the last digit. `ENDPOINT.pt` hashes differ only
because the file also stores the source lock.

New per-step quantities V2.1 records, from the runs in `runs/v2_1/`:

| step | executed_scale | delivery error | shape error | fit multiplier | fit constrained |
|---|---:|---:|---:|---:|---|
| online 0 | 1.0 | 0.01903 | 0.01903 | — (empty span) | — |
| online 1 | 1.0 | 0.01593 | 0.01593 | 0.0 | false |
| frozen 1 | 1.0 | 0.73479 | 0.73479 | 0.0 | false |
| reference 0/1 | 1.0 | 0.0 | 0.0 | — | — |

Two things this shows: no arm backtracked in this run, so delivery and shape
coincide (they diverge only when the guard backs off); and the trust region was
**not binding** — the controller wrote 55% of the budget because its span could
not use more, not because it was clipped.

## Stage B — the predeclared radius gate

Protocol fixed before running: direction = AdamW's first proposal from the
gradient on update-side batch U0; radii `[0.005, 0.02, 0.05, 0.09]`, fallback
`[0.0017]` only if none pass; `eps_lin <= 0.20` and finite CE improvement on the
evaluation batch and on the p2 selection split; no inverse CG; p3 never opened.

| radius | `‖δ‖` | eps_lin | ΔCE evaluation batch (16 imgs) | ΔCE p2 selection (106 imgs) | passes |
|---:|---:|---:|---:|---:|:--|
| 0.005 | 5.0e-03 | 0.284 | +1.23e-03 | −3.43e-03 | no |
| 0.02 | 2.0e-02 | 0.412 | +4.70e-03 | −1.23e-02 | no |
| 0.05 | 5.0e-02 | 0.658 | +1.13e-02 | −2.38e-02 | no |
| 0.09 | 9.0e-02 | 0.925 | +2.12e-02 | −2.79e-02 | no |
| 0.0017 (fallback) | 1.7e-03 | 0.246 | +4.21e-04 | −1.20e-03 | no |

`chosen_radius = null`. Per the protocol this is reported and not rescued by
defaulting to the largest radius.

**The binding constraint is the linearisation, not the CE.** A 16-image batch has
a CE standard error around 0.25, so its column is noise; the 106-image selection
split decreases at every radius. What fails everywhere is `eps_lin`.

The 0.20 threshold is crossed between `‖δ‖ = 2.8e-4` (eps 0.097) and
`9.2e-4` (eps 0.210), so the admissible radius is roughly `<= 8e-4`.

### Consequence for any one-epoch trial

Ordinary AdamW takes steps of norm 0.092 from this anchor; 47 such steps move the
parameters by order 0.6. A controller limited to `rho <= 5e-4` moves by order
0.02 over the same 47 steps — two orders of magnitude less.

So a one-epoch trial at a linearisation-valid radius cannot show a CE difference
against ordinary training, however good the method is. What it *can* answer is
the structural question: whether births slow down, whether the rank stabilises,
and whether an earlier dictionary still explains a later target (`e_shape`). A
CE-parity criterion at this scale is pre-doomed, and that is a property of the
radius, not of the algorithm.

## Not done

- **Stage C has no data adapter.** `datasets/dtd/train` holds 3,760 images; the
  planned split is 3,008 update / 376 training-side guard / 376 separate
  validation, leaving `datasets/dtd/val` (1,880) untouched. `run_a100.py` still
  uses the frozen 200-image development probe and must not be presented as an
  epoch of training.
- No Cars protection run, no EC, no new Schur work.

## Files

| path | content |
|---|---|
| `code/` | V2.1 package; `SHA256SUMS.txt` covers it |
| `code/V2_1_NOTES.md` | every V2.1 change and the homogeneity argument for scaling on backtrack |
| `code/stage_b_calibration.py` | the Stage B script as run |
| `runs/stageA_v2/` | V2 three-arm run (steps, lock, status, finish) |
| `runs/stageA_v2_paired/PAIRED_P3.json` | paired per-image comparison |
| `runs/v2_1/` | V2.1 three-arm run |
| `runs/stageB/STAGE_B.json` | the gate result |

`HASHES.sha256` covers every file here. `EXCLUDED_BINARIES.sha256` covers the six
`ENDPOINT.pt` payloads (2.1-4.8 GB each), which are not in git.
