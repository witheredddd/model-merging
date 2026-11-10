# DTD early CE Surfaces A/B

The original CE Surfaces v1 is the starting point. Both arms use seed 42, the same fixed 3384/376 split, full visual encoder and frozen DTD head. AdamW warmup ends at step 12 and the joint two-coordinate surface writes begin at step 13. Each run checks its step-12 weights against the verified early LTSA replay. The common v1 initializer uses five segments of the 12 warmup steps. Its initial frames and scale are recorded so the two arms can be compared before their first refresh at step 25.

A applies the original v1 surface fitting and birth rules. B adds one bounded correction after v1 has optimized the incumbent chart and before evaluating births. Each subsequent refresh is 12 steps apart. Both use the same cosine schedule through step 216, LR 1e-6 through step 270, and the v1 six-group resource cap. Validation CE selects the locked checkpoint for each arm.

## B correction

At a refresh, sum the preceding 12 clipped training gradients and remove the part covered by the incumbent joint tangent matrix \(T\). The residual is \(g-T(T^\top T)^\dagger T^\top g\). Select the column whose removal reduces projected gradient energy the least, accounting for all other columns. Rotate its group's ambient frame in the plane of that unit tangent and the residual. The orthogonal plane rotation preserves within-group axis orthonormality and keeps its two coordinates and curvature coefficients. Candidates at 5°, 10°, and 15° must satisfy the original per-window axis cap. Require nonnegative CE gains on all three confirmation batches and mean gain above 0.0002, with the zero-angle incumbent as a candidate. Candidate evaluation reoptimizes the existing coordinates with the same radius and budget. The current next-batch gradient is used only to log coverage before and after the choice.

The accepted frame remains weight-free. The stored cumulative group updates and re-anchoring offsets follow the v1 write ledger. Candidate births retain the v1 confirmation rule with identical old-versus-expanded optimization budgets in both arms. Conditional extra CE gains at birth measure whether the older groups could substitute for the new group at that window's anchor.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
anchor=$(./run-python -c 'import json; print(json.load(open("results/DTD_APML_EARLY_20261110/early_anchor.json"))["anchor_sha256"])')
for arm in a b; do
  ./run-python -u scripts/fr_hst/train_dtd_surfaces.py --early "$arm" \
    --expected-anchor "$anchor" --output "results/DTD_CE_SURFACES_EARLY_20261111/$arm" \
    --scratch "/dev/shm/dtd-ce-surfaces-early-20261111/$arm"
done
```
