# DTD APML-inspired reproduction

The public [APML article](https://www.sciencedirect.com/science/article/pii/S0893608026011809) describes LTSA embeddings, blockwise maps and periodic refresh. The exact training implementation was unavailable for this run. The alignment follows the [Zhang–Zha LTSA formulation](https://arxiv.org/abs/cs/0212008). This experiment uses clean DTD labels and the same CLIP ViT-B/16 and DTD head as CE Surfaces.

## Matched continuation

- Replay the original seed-42 warmup on the 3384/376 split. Verify every visual parameter bitwise against the saved step-36 CE Surfaces anchor. Retain the replayed AdamW moments and 19 snapshots at steps 0, 2, ..., 36.
- Continue AdamW, local PCA with ranks 4 and 8, and LTSA with ranks 4 and 8 through step 270. All five arms update the same 48 transformer matrices, one attention input matrix, one attention output matrix, and two MLP matrices per transformer block. The original CE Surfaces run uses the full visual encoder and serves as a separate historical reference. All five continuation arms share sample order and the original cosine schedule through step 216, then use LR 1e-6.
- Record step 36 and each completed epoch after it. Select the lowest holdout CE checkpoint up to step 270, including the common anchor. Lock all five selections before evaluating historical Test. The CE Surfaces step-270 comparison uses the saved holdout measurements.

## Projection details

Each block uses the latest 19 snapshots, a stride of two steps, a 12-point nearest-neighbor neighborhood, and a refresh every 12 steps. PCA computes local principal axes around the most recent snapshot. LTSA estimates the tangent coordinates of every neighborhood and sums the alignment residual matrices \(I-\mathbf1\mathbf1^\top/k-Q_iQ_i^\top\). The lowest nonconstant eigenvectors define the embedding. Local least-squares regression maps these aligned coordinates back to high-dimensional weights near the latest snapshot, followed by orthonormalization and numerical rank truncation. Neighborhood size, local regression, refresh cadence, and optimizer state transport are engineering choices shared where applicable.

For each block with row basis \(U\), use \(Ug\) for the latent AdamW step and lift only its resulting displacement through \(U^\top\). Decay is also restricted to this span. Initialize the moments as \(Um\) and \(U^{\circ 2}v\). At refresh use the old/new overlap matrix for the first moment and its elementwise square for the diagonal second moment. The latter is a diagonal covariance approximation. Refresh leaves the actual weights untouched. The refresh can rearrange the observed local span, but once the rolling window consists entirely of already-projected moves, those snapshots alone supply no new off-span information in exact arithmetic.

Per-step CE gains are evaluated in eval mode on the first 32 images of that training batch. Gradient cosine refers to the eligible 48 matrices. Per-epoch Train CE is sample weighted, with the first continuation epoch covering only steps 37–54. Timings are wall time and can be affected by the existing CE Surfaces process on the shared GPU. The logs retain actual ranks, local LTSA ranks, alignment eigenvalues, principal-angle cosines, basis memory, and sample-order fingerprints.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./run-python -u scripts/fr_hst/train_dtd_apml.py \
  --surface-best /dev/shm/dtd-ce-surfaces-20261110/best.pt \
  --output results/DTD_APML_20261110
```
