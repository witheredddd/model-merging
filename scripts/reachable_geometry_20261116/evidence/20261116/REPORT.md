# Functional reachability: first structural experiment

Completed using the existing three anchored 304D expert-recombination caches. No expert retraining, Jacobian recomputation, adaptive support growth, dynamic updates or EC. All follow-up analyses are exploratory; p4/p5 were already consumed. p6 and official Test remain unopened.

## What is optimized

The norm-only reachable set is {J_C v: ||J_D v|| <= epsilon, ||v|| <= R}. We solve its exact CE-linear support direction, and separately the predictive Fisher/Gauss-Newton CE objective. A third variant retains the prior h_D^T v <= 0 constraint for direct comparison with the original experiment. GN is not the exact CE Hessian. Solver residuals and failed cases are retained in programs.json.

We compare full304, global expert span G, and the union of the eight previously p0-selected conditional planes at each anchor. The latter is explicitly a cached conditional family, not an upper bound on every possible 2D space. New directions are fitted only with p0 geometry; existing finite evaluation is reused, not relabeled as fresh validation.

## Matched primary-radius capacity

Entries are fractions of the full304 optimal predicted GN gain, with the same radius, DTD-response budget and DTD CE slope constraint. These are not actual accuracy gains.

|Anchor|tau|Global2D / full304|Cached conditional / full304|
|---|---:|---:|---:|
|0|0.01|0.0207|0.2382|
|0|0.1|0.0675|0.3869|
|0|0.3|0.0997|0.3689|
|0|1.0|0.0829|0.5829|
|1|0.01|0.0077|0.2327|
|1|0.1|0.0255|0.3861|
|1|0.3|0.0428|0.4284|
|1|1.0|0.0337|0.6052|
|2|0.01|0.0144|0.2204|
|2|0.1|0.0630|0.3606|
|2|0.3|0.1303|0.4575|
|2|1.0|0.1074|0.7118|

## Functional capacity and sampled-motion dimension

M=I/R^2+K_D/epsilon^2 defines an inner feasible ellipsoid. Every point in the exact two-budget intersection belongs to the ellipsoid with squared radius 2. Consequently, the largest achievable Cars squared response lies between sigma_1(J_C M^(-1/2))^2 and twice that value. All 304 singular values and cumulative energy are recorded without Fisher trace truncation. These spectra describe bounded local controllability, not an exact ellipsoidal representation of the reachable set.

optimal_motion_spectra.json separately records parameter/DTD/Cars/joint spectra of sampled full-space optimal updates. This is a one-Cars-objective optimal-motion family across budgets, not the dimension of the whole reachable set, a nonlinear manifold, or a Pareto front. functional_omissions.json measures how much full-space optimal response lies outside the best cached conditional plane; projection itself need not preserve budgets.

The restricted DTD Jacobian has positive minimum singular value at all three anchors. Under this observed full-rank model, exact zero source response permits only zero motion. Near-preservation capacities do not establish exact fiber directions.

## Evidence and artifacts

experiment_lock.json fixes the problem definitions and grid. capacity_gaps.json, reachable_spectra.json, optimal_motion_spectra.json and functional_omissions.json contain the complete analysis. existing_finite_validation.json preserves actual CE, accuracy and KL for the previously evaluated GN+DTD-CE controls, including failed protection on held probes; it does not validate every newly solved norm-only direction. complete.json records solver audit counts. No automatic dimension selection or new mechanism was started.
