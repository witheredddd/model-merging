# DTD-first correction v2.2.0

The v2.1 append-only dictionary is retained as a failed experiment. It does not continuously learn existing directions and must not be described as a complete task-functional joint geometry algorithm. This corrective run first isolates DTD task learning. It makes no multi-task functional-discovery or EC claim.

Train `theta=theta0+Bz`, with one globally orthonormal rank-12 basis. At every minibatch compute the full task gradient g at the current reconstructed model, then use the SAME old B,z for both derivatives:

```
h = B.T @ g
r = (I - B B.T) @ g
grad_B_horizontal = r @ z.T
z_next = z - lr_z * h
u = r / ||r|| ; v = z / ||z||
angle = min(0.05, lr_B * ||r|| * ||z||)
B_next = B + ((cos(angle)-1) * B@v - sin(angle)*u) @ v.T
```

The Grassmann geodesic continuously changes span(B); this is task learning, so the model is intentionally changed. It is NOT claimed to preserve the current expert. Exact angular and parameter movement are logged, together with measured basis movement on sampled rows and global orthogonality checks. At z=0, the first step correctly updates coordinates only. Subsequent minibatches must update the basis; an assertion checks execution counts each epoch. B has no dense Adam states, but its millions of trainable entries are still explicitly acknowledged.

Initialization reuses the prior proven CLIP/DTD pipeline: one training-only warmup epoch at SGD .001 collects 24 normalized gradient snapshots, then global PCA/Gram orthogonalization initializes 12 columns. Reset all selected weights exactly to theta0 and set z=0 before formal training. No old expert or failed v2.1 basis is loaded. Heads, frozen parameters, 48 selected matrices and the 3384/376 validation split remain identical.

Formal budget: 20 epochs, batch64/micro32, coordinate SGD .01 and basis step .1, one-epoch warmup and cosine schedule. Existing B is updated every nonzero-z step throughout training. No rank growth, artificial component grouping, early freeze, multi-task fusion, or EC occurs here. Save complete B,z pairs for best/resume; coordinates from different learned bases cannot be interchanged.

Tests: analytic derivatives vs autograd, exact geodesic orthogonality, joint directional finite difference, zero-z behavior; actual CLIP horizontal CE derivative vs finite difference on four training images. These checks do not prove task performance or a useful frozen low-dimensional manifold.

Performance policy: if the best observed validation accuracy after epoch5 is below65%, stop that failed configuration and emit `failed_gate.json` for diagnosis. At completion, the selected minimum-validation-CE model must reach at least80%; primary reference target81.7%. Below80%, do not access test or advance to EC. Test the single locked model only after passing, and report the historically exposed test split as such rather than claiming a new blind test.

Preserve all previous artifacts in their original directories. New outputs: `/home/shangxueyuan/merging/tsv/results/functional_geometry_dtd_v2_2`. A separate systemd user service supervises runtime/OOM recovery and hourly diagnosis. This code reuses `scripts/dtd_grassmann4.py` and `scripts/tfpml/runtime.py` for model/data utilities; the training algorithm is in this directory.
