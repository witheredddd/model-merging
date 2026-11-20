# DTD online functional control V1 — first real A100 run (2026-11-20)

Integration pilot for the supplied `task_functional_online_v1` package. The package
is preserved verbatim under `code/`, apart from two fixes recorded in `PATCHES.md`.

**Result in one line:** the mechanism runs end to end on real DTD data and every
gate in `INTEGRATION_CHECKS.md` passes, but at the configured write radius the
three arms are statistically indistinguishable on held-out images.

## Provenance

| item | value |
|---|---|
| machine | `10.159.23.24`, 1x A100-80GB, float64 |
| anchor | recorded normal-training step 0, `DTD_DYNAMIC_TRACE_20261109` |
| model | CLIP ViT-B/16 visual tower, 86,192,640 coordinates |
| active domain | 48 Transformer attention/MLP weight matrices, 84,934,656 coordinates |
| probes | p0 200 images (train + Jacobian), p2 106 (adaptive selection), p3 94 (final, opened once) |
| supplied package | `task_functional_online_v1`, 16/16 CPU tests pass on this machine |
| prior offline line | `dtd_schur_20261120` at commit `fdd2464` — a different question, not evidence here |

## Runs

| directory | what it is |
|---|---|
| `runs/online_dtd_step0` | first attempt, failed on the protection-weight bug before any step |
| `runs/online_dtd_step0_r2` | `--steps 1 --cg-maxiter 24` — inverse CG did not converge, no birth, zero write |
| `runs/online_dtd_step0_r3` | `--steps 1 --cg-maxiter 80` — birth admitted at 57 iterations |
| `runs/online_dtd_two_online` | 2 steps, online arm, rank 2 |
| `runs/online_dtd_two_frozen` | 2 steps, `frozen_after=1`, rank 1 |
| `runs/online_dtd_two_reference` | 2 steps, ordinary AdamW proposal, radius and CE guarded |
| `runs/online_paired_p3` | paired per-image comparison of the three endpoints on p3 |

All three-arm runs used `--seed 42 --radius 1e-4 --cg-maxiter 80 --max-modes 3`.

## Gate results (`INTEGRATION_CHECKS.md`)

| gate | result |
|---|---|
| `verify_normal_recipe()`, 48 tensors, 84,934,656 active | pass |
| geometry rebuilt at the current point every step | pass |
| `birth.solver_converged` is a true-residual test | pass — 57 and 63 iterations, 9.68e-04 and 9.79e-04 |
| birth needs residual reduction and real p2 CE gain | pass — 0.981 / 0.729, −1.34e-04 / −3.02e-04 |
| `actual_outside_mode_span` ~ 0, `write_norm <= radius` | pass — 4.2e-20, write = 1.0e-04 |
| `steps.jsonl` carries proposal / write / outside / birth / operator audit | pass |
| p3 not opened before the end | pass |

## Three-arm comparison

Paired on the identical 94 p3 images (`runs/online_paired_p3/PAIRED_P3.json`):

| arm | rank | mean delta CE vs anchor | SE | 95% CI | images better / worse |
|---|---:|---:|---:|---|---:|
| online | 2 | −2.029e-04 | 1.51e-04 | [−5.00e-04, +9.40e-05] | 54 / 40 |
| frozen | 1 | −1.384e-04 | 1.75e-04 | [−4.81e-04, +2.05e-04] | 49 / 45 |
| reference | — | −1.157e-04 | 7.99e-05 | [−2.72e-04, +4.10e-05] | 57 / 37 |

| paired difference | estimate | t | 95% CI |
|---|---:|---:|---|
| online − frozen | −6.45e-05 | −1.12 | [−1.78e-04, +4.86e-05] |
| online − reference | −8.72e-05 | −1.21 | [−2.28e-04, +5.36e-05] |
| frozen − reference | −2.27e-05 | −0.22 | [−2.25e-04, +1.80e-04] |

Every interval contains zero and Top-1 is unchanged in all three arms (37/94
throughout). The ordering online > frozen > reference is suggestive and nothing
more.

## Three findings that are not noise

1. **The default `--cg-maxiter 24` is too small.** At 24 iterations the parameter
   space inverse stopped at a relative residual of 8.3e-03 against a 1e-3 target,
   the birth was refused, no mode existed, and the arm wrote zero for every step.
   At 57 and 63 iterations the same solves converged. One CG iteration costs
   about 4 s at a 16-image training batch.
2. **The dictionary grows one mode per step; nothing is reused.** At step 1 the
   existing mode explained only 25% of the new target response (`err0 = 0.745`),
   so a second mode was born. This is the online counterpart of the flat
   H-metric spectrum seen in the offline audit.
3. **The written step is ~900x smaller than the optimizer proposal it is meant to
   carry.** `proposal_norm = 0.0921` (= `1e-5 * sqrt(84,934,656)`), `radius = 1e-4`.
   The controller therefore executes a direction at a thousandth of the model
   scale, which is exactly why the held-out effect sits at 1e-4 nats and cannot
   be resolved with 94 images.

## What is not here

- No full DTD training, no official validation or test, no EC Pareto search.
- No physical parameter support claim. Modes are full-coordinate directions that
  cross all 48 matrices; the protection graph is a direction-space construction.
- p2 is used adaptively for birth and for the write guard, so p2 numbers are
  optimistically biased by construction. Only p3 is reported as a result.
- `ENDPOINT.pt` for each run (2.1-4.8 GB, containing the parameter vector and the
  learned modes) is not in git. `EXCLUDED_BINARIES.sha256` covers them.

## Reproduce

```bash
cd code
python -m unittest discover -s tests            # 16/16
python run_a100.py \
  --runtime-code-dir <dir with the project bootstrap.py> \
  --inventory <INVENTORY.json> --probes <probes.json> --trace <trace> \
  --output <new empty dir> --arm online --anchor 0 --steps 2 --seed 42 \
  --max-modes 3 --radius 1e-4 --cg-maxiter 80
python paired_endpoints.py --runtime-code-dir ... --output <dir> <run dirs...>
```

`RUN_LOCK.json` pins the source hashes, the input hashes and the anchor, so a
changed source requires a new output directory. That is the package's own design
and it caught the source edit here.
