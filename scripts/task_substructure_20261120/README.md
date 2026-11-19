# Task-driven parameter substructure diagnostic

This is a **local mechanism experiment**, independent of the previous functional-flow and multi-task WLS jobs. It distinguishes sufficiency of a joint motion space, conditional substitution by the full parameter complement, and signed DTD/Cars compatibility. Low dimension and irreplaceability are hypotheses, not admission requirements.

## Pilot contract

- Real normal-training parameter anchors: step 0 and step 25. Original checkpoints are verified and never changed. All 86,192,640 visual parameters participate.
- Locked construction probe p0; a class-stratified split of the disjoint original p1 gives selection p2 and final evaluation p3. DTD counts are 200/106/94 and Cars counts are 300/196/104. These are development images from the training side, not official test images or claimed unseen by the trained experts.
- Eight predeclared positive label-weight demands generate the gradient matrix. All machine-resolved ranks are reported, with no 99% cutoff or selected dimension. Mean-CE-gradient one-dimensional control is a baseline. New frequency-4 demands and independent images are used to choose controls. Frozen anchor-0 and freshly constructed anchor-25 spaces are distinct.
- Existing Fisher directions are copied from the original same-anchor flow run after exact anchor identity verification. Its four available directions permit rank-matched comparisons only where available; original discovery costs are not assumed equal.
- Every native tensor receives Euclidean gradient-energy and signed DTD/Cars diagnostics. Finite physical-region tests use the largest construction-gradient region, the union of negatively signed regions, and a seeded contiguous size-matched control. They are diagnostic supports, **not automatically discovered necessary groups**. Entire remaining parameters can adjust.
- Euclidean sufficiency is compared against all corresponding complements in the supplement. Fisher tests run independently of whether Euclidean deletion seems important.

## Quadratic substitution

For the fixed DTD Jacobian of `2 sqrt(p)/sqrt(N)`, solve

`q(v) = g^T v + 0.5 ||v||^2 + 0.5 beta ||Jv||^2`.

`beta=alpha/kappa`, `alpha in {0.1,1,10}`, `kappa=||Jg_mean||^2/||g_mean||^2`. A restricted domain is represented by an orthogonal projector `P`; the output-space solve is

`(I + beta J P J^T) u = J P g`, `v=-P g + beta P J^T u`.

No input-space Fisher, inverse or preselected coordinate dictionary is formed. The full native parameter complement is used. CG saves its recurrence every iteration and continues from checkpoints; four iterations are a work quantum, not the total solve budget. Fresh projected stationarity is checked. Report `||r||`, `||r||/||Pg||`, parameter-error bound `||r||`, optimum-gap bound `||r||^2/2` and functional bound `||r||/sqrt(beta)` for the ideal fixed operator. These algebraic bounds do not enclose floating-point operator error, sample uncertainty or error of the nonlinear proxy.

A numerically unresolved `Pg` produces a recorded zero action instead of an infinite attempt to reduce relative error of arithmetic noise. The arithmetic floor is a numerical heuristic. In particular, forbidding the very span containing a construction gradient trivially removes that gradient; this cannot establish irreplaceable physical parameters. Independent demands, samples and positions are essential.

Deletion is `q(reoptimized complement)-q(full)`, with residual-based interval, Schur energy consistency, naive freezing, compensation gain and compensation norm. Proxy deletion is not real CE loss.

## Finite interventions and complete metrics

Three common physical radii are calibrated from construction-only full mean-gradient functional radii 0.0025/0.01/0.04. Every direction records its actual norm and functional change. Negative controls are included. No CE or accuracy acceptance gate is used.

Two Fisher comparisons remain separate:

1. Normalize each direction to the same physical radius: a direction comparison, not a constrained-quadratic optimality claim.
2. Use the same `tau=rho/||g||` for full and restricted solutions. Both are budget-feasible and minimize the same physical quadratic with `lambda=1/tau`, `beta_physical=beta/tau`. Actual finite CE is then evaluated without separately normalizing the solutions. The physical deletion interval is `tau` times the original interval.

All constructed, selection and final-evaluation metrics are retained: mean/weighted CE and changes, Top-1 and correct counts, prediction flips, logit margins and quantiles, per-class counts/CE/accuracy, paired CE standard errors and approximate 95% intervals, probability change, product-sphere Log/JVP errors, parameter costs and norms, solver costs and source hashes. Intervals have no multiple-comparison correction.

Signed compatibility uses DTD/Cars gradient inner products, cosine, all-region contributions, exact **first-order** two-task reachable Gram ellipses and nine finite mixture controls per domain. The supplement adds nine ordinary expert-increment coefficient controls at matched parameter radius, plus a separately labelled original-scale family. These are sampled local comparisons, not EC search or a global nonlinear Pareto proof.

The expert supplement includes both a nine-point convex-average line and an independent 3-by-3 two-coefficient grid over `{0,0.5,1}^2`. Each family is compared separately with the same number of evaluations. Radius normalization can yield repeated directions; a zero increment at an expert endpoint is recorded as a no-op. Neither duplicates nor zero motions are counted as new controllable freedoms.

## Unattended execution

The companion `task_substructure_campaign_20261120` controller owns subsequent jobs. It finishes the pilot, runs `task_substructure_supplement.py`, generates all tables, then schedules the larger profile. Source/input hashes are frozen; explicit migrations are recorded. Each evaluation is atomic and can be skipped on resume. Failed jobs have up to three recorded attempts; unresolved failure stops with `FAILED`, not success.

Raw `.pt` vectors stay on the A100. Original trajectories, model weights and image IDs are not published. Reports and numeric array indexes are generated automatically. This diagnostic does not train a new expert or establish Val >=80%, and does not establish an EC Pareto improvement. Those remain separate end-to-end tests.

Pilot operator counters in status/solver files are per process and can reset after resume; do not treat their final values as total campaign cost. Gradient timings and phase logs remain. The expanded profile additionally records process IDs, per-evaluation wall time/JVP/VJP costs and append-only `TIMING_HISTORY.jsonl`, so subsequent restarts can be accounted for explicitly.
