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

## Results at the 270-step budget

| Method | Total rank | Val accuracy at 216 | Val accuracy at 270 | Val CE at 270 | Locked step | Historical Test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | Ambient | 80.85% | 81.12% | 0.80093 | 81 | 78.94% |
| PCA-4 | 192 | 77.13% | 76.86% | 0.84456 | 270 | 74.68% |
| LTSA-4 | 192 | 76.86% | 76.86% | 0.84338 | 270 | 74.36% |
| PCA-8 | 384 | 77.13% | 76.86% | 0.84098 | 270 | 74.63% |
| LTSA-8 | 384 | 76.86% | 76.86% | 0.83772 | 270 | 74.68% |
| CE Surfaces fixed benchmark | 4 | 77.39% | 79.26% | 0.79451 | 270 | 77.59% |

CE Surfaces uses the historical full-visual training protocol. Its step-270 checkpoint was fixed before evaluating its benchmark Test and is separate from any later validation selection in its original continuation. The step-36 anchor is 76.33% on holdout with CE 0.85668. The 4- and 8-dimensional projection bases occupy 1.27 and 2.53 GiB of host memory, respectively. Per-arm peak allocated CUDA memory was 10.72 GiB for projected arms and 11.68 GiB for AdamW. Median actual step norms were 0.00557 for AdamW and 0.0000206–0.0000298 for the four projections. The fixed-rate comparison therefore includes a major displacement-scale difference. Concurrent processes reached the container memory limit during the first PCA-4 run. The completed AdamW arm was retained, and the projected arms were restarted serially after verifying the same anchor.

## Two early diagnostic arms

`--early` runs just early LTSA and early LTSA with task correction. Replay through step 36 still validates provenance, while both arms start from the same step-12 AdamW weights and moments. Geometry starts from snapshots at every step from 0 to 12. Both methods use rank 8 per block and the same data, 12-step refresh cadence, post-warmup snapshot stride, and schedule. A shadow AdamW step on the current arm supplies the displacement norm. The actual latent AdamW direction is lifted through its basis and rescaled to this budget, with CE choosing among fractions 1, 1/2, 1/4, 1/8, and zero on the first 32 training images. The next up to 32 images measure the actual CE gain.

The task arm sums gradients from the preceding refresh window. At each refresh it removes the new trajectory basis's coverage of that sum, then rotates the axis with the smallest projected gradient magnitude by 15 degrees toward the residual. The new span has the same dimensionality. The initial refresh has no post-warmup gradient history, so both arms start with identical geometry. Neither correction nor refresh changes model weights directly. Comparing the two new arms isolates the addition of task correction. Comparing early LTSA against the original LTSA changes both intervention timing and step selection.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
./run-python -u scripts/fr_hst/train_dtd_apml.py \
  --surface-best /dev/shm/dtd-ce-surfaces-20261110/best.pt \
  --output results/DTD_APML_20261110
```

For the early experiments, add `--early` and use `--output results/DTD_APML_EARLY_20261110`.

### Early diagnostic results

| Method | Val accuracy at 216 | Val accuracy at 270 | Val CE at 270 | Locked step | Historical Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early LTSA | 71.01% | 71.81% | 0.98686 | 270 | 71.33% |
| Early LTSA with task correction | 81.65% | 80.59% | 0.68845 | 243 | 79.95% |

The common step-12 anchor has holdout accuracy 53.46% and CE 1.58513. Both arms have identical data order and step records through step 24, excluding timing. The shared requested total rank is 384. Actual ranks over refreshes span 374–384 for early LTSA and 371–384 with task correction due to numerical truncation in the trajectory fit. No step exceeds its own shadow AdamW radius. Median realized norms are 0.00385 and 0.00559, respectively. The pure trajectory arm accepts 219 nonzero updates and the corrected arm accepts 239. The correction rotates one axis in each eligible block on all 21 refreshes after the initial chart, for 1,008 block rotations. The two early arms took 9.58 and 11.20 minutes, with peak allocated CUDA memory of 11.68 GiB each.

Adding task correction improves the locked historical Test by 8.62 percentage points at the same requested rank. The pure early arm underperforms the original late LTSA-8 even with the larger CE-searched displacement. Since the timing and step-selection changes are combined in that comparison, it does not isolate the effect of timing alone. The two early runs use seed 42. They establish a within-run comparison, without a cross-seed uncertainty estimate.
