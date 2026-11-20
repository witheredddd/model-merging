# DTD Schur conditional compensation — step 0 (2026-11-20)

One real DTD anchor, one active domain, one measured chain:

    real CE demands -> full-domain regularized motions -> H-metric candidate
    motion space -> Schur conditional deletion -> fixed-driver compensation
    -> finite CE on held-out images

This is a **diagnostic**, not a discovered parameter structure and not an online
training loop. Nothing here claims irreplaceable weights, an intrinsic dimension,
or a reusable low-rank movement basis.

## Provenance

| item | value |
|---|---|
| machine | `10.159.23.24`, 1x A100-80GB, float64 throughout |
| anchor | recorded normal-training position step 0 of `DTD_DYNAMIC_TRACE_20261109` |
| model | CLIP ViT-B/16 visual tower + DTD head, 86,192,640 visual coordinates |
| active domain | 48 Transformer weight matrices, 84,934,656 coordinates |
| probes | training-side development images, disjoint roles 200 / 106 / 94 (DTD, 47 classes) |
| objective | `q(v) = g^T v + lambda/2||v||^2 + beta/2||Jv||^2`, `lambda=1`, `beta=alpha/kappa`, `alpha=1` |
| kappa | 553.640652 = `||Jg||^2/||g||^2` on the construction probe; matches the earlier pilot's 553.6406517182841 |
| official test set | **never opened** |

`J` is the Jacobian of the `2 sqrt(softmax(logits))/sqrt(N)` embedding, not of raw
logits. `H = lambda I + beta J^T J` is a local optimisation cost, **not** the
network's CE Hessian. It is never formed.

## Code

| file | role |
|---|---|
| `code/conditional_schur_core.py` | supplied, tested mathematical core, unchanged |
| `code/test_conditional_schur_core.py` | its six dense synthetic tests |
| `code/schur_resumable.py` | A100 adapter: checkpointable dual CG, true-residual replacement, objective-gap stopping rule |
| `code/test_schur_resumable.py` | seven dense tests for the adapter and the projector precondition |
| `code/run_step0.py` | the only driver; reads the existing `TrainingPositions`, `FullFunctionalGeometry`, `EightTaskContext` |
| `code/bootstrap.py` | path resolution for the pre-existing project modules |

All twelve tests pass on the A100 (`python -m unittest test_conditional_schur_core
test_schur_resumable`).

### Three corrections the real problem forced

1. **The dual CG must be stopped by the primal residual, not by itself.**
   An output-space residual of 1e-5 can leave `||P(g+lambda v+beta J^T Jv)||/||P g||`
   at 3.7e-2, because the two differ by a factor `beta*||J||` (~1e4 here).
2. **The dual recurrence residual lies on this conditioning.** `I + (beta/lambda) J P J^T`
   has `beta*||J||^2 ~ 1e3`; the recurrence residual falls to 1e-9 while the true
   residual stalls near 1e-5. The adapter replaces it by `rhs - A x` every chunk.
3. **Modes passed to `P_N = P_D - sum u u^T` must satisfy `P_D u = u`.**
   A raw gradient carries `||g_outside|| = 26.4` of `||g|| = 44.65`; with it,
   `P_N` is not idempotent, the restricted quadratic has no minimiser, and the
   primal residual can never settle. `test_out_of_domain_mode_breaks_the_projector`
   pins this. `require_in_domain` now rejects such modes instead of silently
   producing numbers.

These three were found by watching the primal residual, not by any test of the
mathematical core — the core is correct.

## Results

### Numerical gates

| gate | value |
|---|---|
| JVP/VJP adjointness relative error | 6.14e-15 |
| candidate modes: max orthonormality error | 2.17e-18 |
| candidate modes: outside-active-domain norm | 0.0 |
| compensation linearity relative error | 2.18e-07 |
| Schur identity error, deletion | 5.98e-04 on a gap of 40.762 (1.5e-5 relative) |

### Full-domain motions (4 disjoint construction demands)

| demand | q | iters | stationarity | `||v||` |
|---|---:|---:|---:|---:|
| g0 | -490.0922 | 8 | 3.70e-05 | 28.153 |
| g1 | -767.4267 | 8 | 3.96e-05 | 34.958 |
| g2 | -632.0179 | 8 | 3.81e-05 | 31.266 |
| g3 | -581.0609 | 8 | 3.37e-05 | 30.729 |

Pairwise cosine between the four demand gradients: 0.05, 0.13, -0.01, 0.06, -0.06.
These are independent *estimates of one task*, not distinct semantic functions.

### H-metric motion Gram — **no low-rank structure**

eigenvalues `395.65, 315.68, 287.43, 236.53`; cumulative fraction
`0.320, 0.576, 0.809, 1.000`.

Rank is bounded by the four construction demands and is not an intrinsic
dimension, but within that bound the spectrum is nearly flat: rank 1 carries 32%
of the H-energy of the movement set.

### Deletion on an independent demand (probe 2, never probe 3)

| family | rank | proxy deletion | / abs(q_full) | Schur identity error |
|---|---:|---:|---:|---:|
| random coordinates | 1 | 2.3e-05 | 5.0e-08 | 4.8e-10 |
| existing Fisher | 1 | 0.919 | 0.20% | 5.1e-05 |
| existing Fisher | 2 | 1.208 | 0.27% | 1.0e-04 |
| candidate (H-metric) | 1 | 4.205 | 0.93% | 5.9e-05 |
| candidate (H-metric) | 2 | 6.354 | 1.40% | 1.3e-04 |
| candidate (H-metric) | 3 | 7.192 | 1.58% | 1.4e-04 |
| mean gradient | 1 | **11.580** | 2.55% | 3.1e-04 |

All solves converged (stationarity 2.4e-05 to 2.5e-05, 8 iterations).

The random control at 2.3e-05 is the apparatus's own null: forbidding a
meaningless direction costs nothing, so the rest of the column has resolution.

Reading: a **single mean-gradient direction** carries 2.8x more of a fresh
demand's optimum than the top H-metric candidate mode, and 12.6x more than the
best existing Fisher direction. Deletion is monotone in the size of the forbidden
subspace, so it cannot select a rank by itself; the reported rank is chosen only
by the advantage over dimension-matched baselines at ranks where every baseline
family is present.

### Deletion on the construction demand (self-referential)

With the candidate space built from g0's own motion and then forbidden on g0:
`q(full) = -490.0922`, `q(restricted) = -449.3299`, deletion **40.762** (8.3%).
Compare 6.354 on the independent demand. Freezing the span used to construct a
gradient trivially removes that gradient; the 6.4x drop is the size of that
self-reference and is why this number is not evidence.

### Conditional compensation

Driver `x` = the projection of the unrestricted g0 motion onto the selected modes.
`n(0)` = the complement's own optimum; compensation = `n(x) - n(0)`, kept separate
from `n(0)` on purpose.

| amplitude | `||x||` | `||n(0)||` | `||compensation||` | compensation/driver | q(joint) - q(full) |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 2.048 | 27.074 | 0.442 | 0.2158 | 22.929 |
| 0.50 | 4.096 | 27.074 | 0.884 | 0.2158 | 10.191 |
| 1.00 | 8.193 | 27.074 | 1.768 | 0.2158 | -8.5e-08 |

The ratio is constant across amplitudes to ten digits, as
`n(t x) = n(0) + t[n(x) - n(0)]` requires; the one measured linearity check at
half amplitude agrees to 2.18e-07. At t=1 the joint update reproduces the
unrestricted optimum, so the compensator is not a decorative term.

Driver leakage into the complement: 1.06e-15.

### Finite DTD outcomes — **and the step-size problem**

| state | probe 0 delta CE | probe 2 delta CE | probe 3 delta CE | probe 3 delta Top-1 | flips |
|---|---:|---:|---:|---:|---:|
| full optimum | +5.129 | +5.125 | +4.953 | -0.372 | 0.926 |
| restricted optimum | +4.011 | +4.008 | +3.861 | -0.372 | 0.936 |
| joint, amplitude 1 | +5.129 | +5.125 | +4.953 | -0.372 | 0.926 |
| driver only, amplitude 1 | +3.107 | +3.131 | +3.024 | -0.383 | 1.000 |

**Every one of these steps destroys the model.** This is not a bug and it is the
most important caveat in this directory:

- the raw quadratic optimum has `||v|| ~ 28`, while the calibrated physical radius
  used by the earlier pilot is `rho = 0.0025/sqrt(kappa) = 1.06e-4` — the raw
  optimum sits **2.6e5 times outside** the region where the local quadratic was
  ever validated;
- the first-order prediction is `g.v <= -886` nats (`q = -490.09`,
  `0.5||v||^2 = 396.3`); the measured change is `+5` nats. The proxy is not
  merely inaccurate there, it has the wrong sign;
- consequently the finite table above **cannot** adjudicate between full,
  restricted and compensated steps, and the proxy deletion column must not be
  read as a CE deletion.

What the finite pass does establish is the limit of the experiment: with
`lambda = 1` the regularised optimum is not a usable step, and any real test of
these structures has to rescale every arm to a common physical radius. That
evaluation was not run in this pass. `evaluation.csv` reports the complete raw
numbers including harm, flips and per-class detail; no acceptance gate was used.

## What is not here

- No physical parameter support was proposed, ranked or deleted. The 48-matrix
  domain is a declared family, not a discovered region.
- No new expert was trained; Val >= 80% and any EC/Pareto claim remain untested.
- No online write-back, no 17-anchor campaign, no hyper-parameter sweep.
- Binary vectors (15.2 GB) are not in git. `HASHES.sha256` covers them and
  `EXCLUDED_BINARIES.md` explains how to rebuild them from the retained
  checkpoints.

## Reproduction

```bash
python -m unittest test_conditional_schur_core test_schur_resumable
python run_step0.py \
  --output <dir> --inventory <INVENTORY.json> --probes <probes.json> \
  --trace <DTD_DYNAMIC_TRACE_20261109/trace> --flow-source <self_fiber_flow_DTD_20261119>
```

Every solve checkpoints to `solves/<label>.npz` and resumes; every result file is
written atomically and content-addressed.
