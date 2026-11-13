# Fisher Subspace Support Profile

Completed [A100 support profiles and results](evidence/20261114/REPORT.md). These diagnose how much original-coordinate support is needed along full ranking paths, rather than testing a prescribed1% controller.

The supplied Fisher space is the object of analysis, not a prescribed raw-coordinate support budget. The implementation reads its dimension from the basis file; it does not learn new directions, choose a new ten-dimensional space, refit coordinates, or keep a fixed1% mask.

For each supplied historical eigenvector in descending eigenvalue order and for the whole space, compute complete sorted cumulative parameter-mass and absolute signed-Fisher-allocation curves. Invert these exact curves at50/80/90/95/99% mass. The whole-space parameter score is leverage; its rho is the signed sum over columns before taking absolute value, preserving invariance under orthogonal basis rotation.

Evaluate original-direction masks from0.001% to100%, adding the data-dependent support counts obtained by the mass inversions. Direct JVPs measure residual error, cosine and amplitude on two128-image development probes. Rho ranking uses only probe0; the exact same mask is checked on probe1. Do not renormalize masked directions, optimize coefficients, or assume response error is monotone in support size. At100% all masks reproduce the cached full responses exactly.

Function-quality labels80/90/95/99 mean relative response error at most0.20/0.10/0.05/0.01. They are not percentages of Fisher energy retained. Report the first observed passing support count on the sampled ranking path and any later violation; this is neither an exact continuous threshold nor a globally minimum subset. Full parameter/absolute-rho mass thresholds are computed from complete sorted arrays, not a fixed budget.

Inspect history-average eigenvalue gaps and leave-one-fit-state-out direction alignment before interpreting individual modes. Direction labels are conditional on this historical operator; this is not evidence of immutable functional atoms. Whole-space leverage, signed rho sum and Frobenius profiles are basis-rotation invariant. Whole-space Frobenius error also weights stronger responses more heavily; raw records retain per-column errors under each common mask.

Input state is old v5 theta128. JVP is a local derivative, not a finite-step task-performance claim. No official Test or fusion model is used. This experiment supersedes fixed-budget control as the current research objective; older experiments are retained as separate records.

Private full coordinate rankings remain on A100. Each support point is atomically cached as JSON; completed points can be skipped on restart. Reports preserve executed-source hashes and input identities. Completed evidence and plots will be under evidence/20261114.
