# Pareto balance pilot — DTD/Cars (2026-11-21)

Test of the hypothesis that per-coordinate gradient-cancellation positions
`lambda*_j` in the two-task preference simplex form a stable, natural grouping of
physical parameters.

**Result: it does not. The hypothesis is retired.** Balance positions are
approximately what independent co-ordinate pairing predicts, the distribution is
middle-heavy rather than edge-heavy, and positions move when the images change.

This does **not** speak against joint multi-parameter motion with preserved
prediction — nothing here tested joint motion.

## What was run

| run | data | coordinates |
|---|---|---|
| `runs/archived304_scale{0.5,1,2.0}/anchor_{0,1,2}/` | archived local Jacobians and gradients at three fusion anchors | 304 expert-recombination coordinates |
| `runs/physical_run_scale{0.5,1,2.0}/` | exported DTD/Cars CE gradients and expert deltas at one anchor | 86,192,640 physical coordinates |
| `runs/SCALE_SHAPE_CHECK.json` | re-analysis of the exported gradients | 86,192,640 |

The 304D runs are a pipeline check only: those coordinates are derivatives with
respect to a 304-dim expert-recombination dictionary, not physical weights, and
"cross-anchor" consistency is not independent-probe reproducibility.

`pareto_balance_probe.py` is the supplied script, unmodified
(sha256 `a44a517f…`). `export_physical.py` builds the physical inputs;
`scale_shape_check.py` is the correction analysis below.

## Physical-parameter result

Anchor = the pretrained visual tower `theta_0` (the base that task arithmetic is
defined against, `theta_0 + sum_k (theta_k - theta_0)`). Gradients: mean CE on
each task's primary probe; held-out gradients: a disjoint probe at the same
anchor; deltas: expert visual weights minus `theta_0`. All six arrays flat
float64 in one key order, manifest with keys, shapes, offsets and sha256.

At `scale_d = 1`:

| quantity | value |
|---|---|
| coordinates | 86,192,640 |
| opposite-sign gradients | 43,167,830 (**50.08%**) |
| opposite in both probes | 22,332,402 (51.7% of opposite) |
| of those, `abs(delta lambda*) <= 0.10` | 5,287,906 = **23.7%** |
| weighted `lambda*` MAE among the stable set | **0.244** |
| edge weight fraction (`lambda<0.1` or `>0.9`) | 0.046 |
| middle weight fraction (`0.4..0.6`) | **0.319** |
| real vs within-chunk permutation null, L1 | 0.119 |
| the null's own edge fraction | 0.072 (higher than the real 0.046) |

1. **Opposition is 50.08%**, i.e. the sign-independence baseline. Mixed-sign
   gradients are not enriched.
2. **The weighted distribution is middle-heavy**, the opposite of the
   "two peaks at the ends" picture, and it is *less* edge-concentrated than a
   random re-pairing of the two gradients.
3. **Positions do not reproduce across probes.** Counting from all opposite
   coordinates rather than from the doubly-opposite ones, only
   `0.517 * 0.237 = 12.3%` keep `abs(delta lambda*) <= 0.10`, i.e. about **6.1% of
   all physical coordinates**. The weighted MAE of 0.244 is 2.4x the script's own
   stability tolerance.

## Correction: loss scale translates, it does not restructure

`lambda*_j(a) = |g_C| / (a|g_D| + |g_C|)`, so the log-odds

    t_j = log(lambda*/(1-lambda*)) = log(|g_C|/|g_D|) - log a

is translated by exactly `-log a` when the DTD loss is scaled by `a`. Measured on
the exported gradients:

| quantity | measured | predicted |
|---|---|---|
| median `t(0.5) - t(1)` | +0.6931471805599454 | +0.6931471805599453 |
| median `t(2.0) - t(1)` | −0.6931471805599454 | −0.6931471805599453 |
| histogram shape L1, before / after removing the shift | 0.3899 / **0.0007** | 0 / 0 |

So the edge and middle fractions reported above are **not scale invariants**:
they change because fixed `lambda` edges sit on a translated distribution. The
shape of the distribution does not change. The pilot's three-scale sweep tests the
binning, not the structure; a scale check must be done on centred `t`.

The negative conclusions do not rest on the scale sweep — they rest on the
opposition fraction and the cross-probe instability, neither of which is affected.

## A signal that is not a finding

Per unit of the script's `weight`, edge coordinates carry about 5x more
`cross_harm` than middle ones (3.97 vs 0.789). This is confounded: `harmonic` is
governed by the smaller gradient while `cross_harm` can be governed by the larger
one, so the ratio is inflated at the edges by construction. It would need equal
parameter displacement, equal target-task gain or equal Fisher displacement to be
interpreted. Not evidence of anything yet.

## What this does and does not establish

- Retired: per-coordinate `lambda*`, from one probe, as a parameter-ownership
  label or as the basis for grouping coordinates for EC.
- Retired: reading the edge/middle histogram as a natural clustering.
- Not tested here: whether several coordinates moving **together** can preserve
  the protected task's predictions while improving the other task. Equal ratios
  `g_D,j/g_C,j` do not imply `J_D e_j ≈ J_D e_l`, so the coordinate route could
  not answer that question even with infinite data.
- Not tested here: any real merge evaluation.

## Boundaries

- One anchor, one probe pair, two tasks.
- The permutation null is within-chunk (chunk 500k coordinates, which crosses
  tensor boundaries); the supplied README notes this is a weak control. A matched
  null needs chunk boundaries aligned with physical tensors.
- All quantities are local first-order ones. No CLIP merge was evaluated.

## Files

| path | content |
|---|---|
| `code/` | the supplied probe (unmodified), the physical exporter, the scale-shape check |
| `runs/archived304_scale*/` | 304D summaries, histograms, figures, SCOPE notes |
| `runs/physical_run_scale*/` | physical-parameter summaries, histograms, figures, SCOPE notes |
| `runs/physical_manifest/MANIFEST.json` | key order, offsets, per-array sha256 and norms |
| `runs/SCALE_SHAPE_CHECK.json` | the translation measurement |

`EXCLUDED_BINARIES.sha256` covers the six `.npy` arrays (4.1 GB) that are not in
git; `HASHES.sha256` covers everything here.
