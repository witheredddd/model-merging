# Fixed10 single-task coordinate support and control

This follow-up remains entirely inside DTD at the old v5 theta128. B is unchanged. The user's clarification supersedes the initial attachment's fusion-based interpretation of experiment B; no EC, cross-task conflict, or fusion checkpoint is used.

## A: descriptive coordinate geometry

On each of two disjoint128-image Xprime probes, compute the actual ambient48-matrix Fisher action FB and C=B^T F B. Store ellB_i=||B_i||², ellQ_i=||Q_i||², g_i=(FB)_i C†(FB)_i^T and rho_i=sum_j B_ij(FB)_ij. For the old and external expert endpoint updates from theta0 (both restricted to48 matrices), compute signed response alignment a_i=v_i(FB)_i C†h/(h^T C†h), h=B^T Fv. These are local quantities at theta128, not finite-update performance predictions or task conflict scores.

Check sum ellB=10, sum ellQ=26, sum rho=tr C and sum a=1. Keep negative rho/a. The coordinate-scale convention is the original Euclidean parameterization; g is not invariant to arbitrary parameter rescaling. Relative leverage enrichment E=ellB/((10/26)ellQ) is bounded by2.6 and is undefined if ellQ=0. A fixed nonzero row under Haar random10 in26 has E/2.6 ~ Beta(5,8). Matrix-bin null uses1000 shared Haar draws; upper-tail empirical p-values receive Holm adjustment over48 bins. The null tests deviation from random within the old candidate range, not task benefit or novelty.

For each original matrix mask M_G, calculate all10 masked JVPs R_G=J M_G B. Save Gamma_GH=<R_G,R_H>F. Verify sum R_G=JB, sum Gamma=||JB||F² and direction-removal energy change Delta_G=2<R_G,R>-||R_G||F². Matrix and layer bins are reporting choices, not newly discovered parameter groups. These identities neither establish causal formation of B nor predict CE effects. Full coordinate maps, FB and response tensors stay private on A100.

## B: hypotheses tested with real masked model changes

All masks contain exactly849346 coordinates (nominal1% of84934656). Select masks and fit response controls on Xprime indices0:128. The independent control check is indices256:512, disjoint from both A probes. It was used in older evaluations but never selects these masks or controllers. Two training-side64-image batches supply task gradients; they are disjoint from fit/held probes. No official Test is opened.

Access features are H_i=(FB)_i C^(-1/2), with all10 retained positive response eigenvalues. Candidate `joint_logdet` uses16 batches of greedy logdet marginal-score ranking, accumulating selected H_i^T H_i. Initial information ridge is0.01 * trace(H^T H)/10 * budget/D. Batch selection is a heuristic; it does not solve exact D-optimal selection and is not a causal proof. Score-only top_g and top_leverage are explicit baselines. Random-coordinate controls use seeds14,15,16. Position controls use seeds101,102,103, randomly order original matrices and take complete matrices plus a truncated boundary matrix for the same count. Matrix-matched controls use seeds41,42,43 and randomly select the candidate's exact count inside each matrix. There are12 masks total. No masks use the external high-performance expert or task-check labels.

For four seeded coefficient directions (seeds5021..5024), both signs and radii0.001/0.003, the target is the real model theta128+B*c*radius. These are four sampled directions, not exhaustive control of all ten axes. Compare two algebraic controllers per mask:

1. Minimum parameter norm subject to matching the selected linear response, delta_G=H_G(H_G^T H_G)† C^(1/2)c*radius.
2. Fit the full response in the same masked-access parameter span: T=J E_G H_G; solve min_x ||T x-(Phi_target-Phi_base)||² + lambda ||H_G x||², with lambda=1e-6*tr(T^T T)/tr(H_G^T H_G), delta_G=H_G x.

Both use relative pseudoinverse cutoff1e-8 and cap the actual proposed parameter norm at0.01. This is the same upper bound, not necessarily identical realized norms. Fit and held real forwards measure selected-target error, full-target error, outside response and target-to-control KL. Finite response is measured in the fixed base square-root probability embedding; it is not a parallel-transport comparison. The second controller is optimal only within its particular masked-access span, not among every possible masked parameter intervention.

Separately evaluate each mask's true restricted CE gradient using both training batches and radii0.001/0.003/0.01. Normalize after masking, so parameter norms are matched. All12 methods get6 task checks and32 response checks, each evaluated on fit and held images. Every point starts from the unchanged theta128. No iterative coordinate training, optimization trajectory, basis update or new learned direction is retained.

Results must distinguish fit from held behavior, selected-response matching from outside spill, and CE descent from reproducing arbitrary prediction targets. One budget, four sampled directions, one anchor and two correlated gradient batches support only a local mechanism test. Compare candidate against all baselines; equal performance to top_g does not establish an independent advantage for the joint heuristic. Poor control by these solvers does not prove that a mask has no better achievable control.

## Recovery and public evidence

A checkpoints each matrix-response cache and completed probe; B caches masks, gradients, per-mask response operators and completed method reports. An incomplete method can be replayed from its start; there is no online optimizer state. Config hashes reject changed-source reuse. Reports will link the executed source hash, input basis/base/head identity and sample IDs. Original expert checkpoints and all old experiment outputs remain intact. Publish code, aggregate numeric diagnostics, all intervention outcomes and plots; retain masks, coordinate maps, gradients, bases, raw responses and weights on A100.
