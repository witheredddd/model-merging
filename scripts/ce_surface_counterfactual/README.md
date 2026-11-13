# CE Surfaces counterfactual diagnostics: actual delivery status

Requested on 2026-11-13, based on frozen commit `6c78fee73670f651b06495d5da305955fe9495e8`. This is an independent diagnostic module; it does not replace the old surface implementation or train a new fiber algorithm.

## Completed

- Two coefficient-space tests pass: overlap/rank-preserving orthogonalization and common parameter-radius/rejection-category checks.
- `historical.py` read the local original `resume_step432.pt` archive using metadata-only placement, without modifying it. [historical_37_216.json](historical_37_216.json) contains the retained step37–216 records, actual rejection categories identifiable from them, and conditional tangent coefficients at milestone216.
- The A100 host `10.159.23.24` has neither `/dev/shm/dtd-ce-surfaces-20261110` nor `results/DTD_CE_SURFACES_20261110`. The original WestC connection was refused. No step216 full state has been recovered. Local archive432 is from the 36-step-warmup original run, not the early A/B run.

## Implemented, not yet executed on A100

`run.py` performs three fixed-state paired replay contexts at step432, comparing:

1. A cloned AdamW reference from saved moments, not a theoretical descent upper bound.
2. Its projection onto the actual current tangent span, with parameter coverage, functional coverage and norm-matched CE.
3. Four-round real CE optimization in the current linear tangent span.
4. The original quadratic solver, plus a same-coordinate linear/quadratic comparison with a common radius shrink.
5. Fit failure, cross-batch failure and FP32-reconstruction gate flip, using both virtual and exact canonical held CE.
6. Gradient-residual versus AdamW-residual tangent rotations, fit-only selection, independent held CE and response coverage.

A same-rank greedy residual-axis replacement additionally probes direction-learning limitations. It is not an exhaustive optimal-subspace solver; lack of benefit cannot by itself establish inadequate rank.

All cases preserve the actual checkpoint model, use the same fit/held data and AdamW-norm radius, save partial results after completed cases, and bound PyTorch memory allocation to35% of the A100. Common iteration/radius budgets do not imply equal realized calls or FLOPs. Saved replay images here are24 per batch rather than the original live32/32 gate. IDs are training-subset indices. Results at432 cannot attribute individual historical refusals at37–216.

The 7.7GB archive transfer was incomplete when the user redirected work to the new functional-pullback training direction. **No GPU counterfactual results, successful A100 memory test or completed six-arm experiment are claimed here.** Partial input copies remain separate from original artifacts and must be fully verified before execution.

## Mathematical boundaries

The old parameter-orthogonal frame does not imply functional orthogonality. Nonzero Jv does not prove ambient horizontality. U-perp is not kerJ. Small singular values are only approximate probe-local near-null evidence; rank-capped active directions are not vertical. Milestone216's logged z/q do not support asserting tangent collapse. Nonlinear fiber compensation requires measurable reachability; no global inverse, exact geodesic or guaranteed fiber/generalization benefit is asserted.

## Reproduction

```bash
python -m unittest discover -s scripts/ce_surface_counterfactual -v
python scripts/ce_surface_counterfactual/historical.py --checkpoint /verified/resume_step432.pt --output /new/historical.json
python scripts/ce_surface_counterfactual/run.py --root /home/shangxueyuan/merging/tsv --frozen /path/to/extracted/6c78fee --checkpoint /verified/resume_step432.pt --output /new/counterfactual-output
```

Large model archives are not uploaded to Git. Historical source/input identities and limits are explicit; no old experiment results are overwritten.
