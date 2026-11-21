# Excluded binaries

Six flat float64 arrays of length 86,192,640 (about 690 MB each, 4.1 GB total).
They are derivable from the manifest's recipes and the checkpoints; their sha256 is in
`EXCLUDED_BINARIES.sha256`, paths relative to the experiment output directory.

| array | L2 norm |
|---|---:|
| g_c_heldout.npy | 28.038978 |
| g_c_train.npy | 27.920851 |
| g_d_heldout.npy | 38.259149 |
| g_d_train.npy | 28.437015 |
| u_c.npy | 1.995992 |
| u_d.npy | 1.034323 |
