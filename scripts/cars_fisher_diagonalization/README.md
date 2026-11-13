# Cars Fisher subspace diagonalization

Completed [Cars spectrum, matrices, independent-probe audit and finite interventions](evidence/20261114/REPORT.md).

Reuse the saved87.08% Cars adaptive-dynamics expert and its actual learned frames/nonlinear coordinates. Full visual scope86,192,640. No old DTD v5 fixed10, support profiling, masks, leverage or EC. No retraining or preset Cars Fisher dimension. The saved training did reach its original8-component cap; that limitation is explicit.

All learned frame axes and actual full expert/frozen-decoder displacements yield25 numerical candidate directions. The actual nonlinear-coordinate tangent has rank16. The corresponding restricted Fisher spectra and diagonals are separate outputs. Candidate rank and coordinate rank are not intrinsic Fisher dimensionality.

Two disjoint128-image training-side probes,196 classes, mean square-root probability Fisher F=J^T J. The expert saw all training data, so these probes are not unseen by the trained expert. Candidate basis and matrices do not use Test. The locked Test only reproduces87.0787% (7002/8041).

Primary diagonalization is at the Cars expert. Base and saved frozen-decoder models provide additional anchor variation; these three saved models are not a recovered training trajectory. The original DTD common-block solver is reused; the final audit requires both global leakage<=5% and normalized block-pair coupling<=10%. It fails this strict check. Independent-probe off-diagonal ratio in the expert's fitted eigenbasis is0.6410, so stable transferable diagonal coordinates have not been established.

All fitted eigenmodes receive signed finite perturbations0.001/0.003, evaluated on both probes with real KL and CE. Numerical construction, finite locality, eigenspace stability and true task conflict are separate claims. No cross-task experiment is included in this Cars-only delivery.

The initial service launch hit a module path lookup failure before config or model computations. The path was corrected to the actual frozen solver directory; the successful execution source hash and solver hash are recorded. Three math tests passed in the actual Linux environment; local Windows mixed NumPy/PyTorch tests encountered an OpenMP library conflict and are not counted as passed.

Private high-dimensional basis/model states remain on A100. Publish only code, matrices/CSV/heatmaps, aggregate spectra and finite-intervention JSON. Response caches and per-mode JSON support replay; no hard-interruption recovery test was performed.
