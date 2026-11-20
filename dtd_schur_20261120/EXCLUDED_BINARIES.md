# Excluded binaries (15.2 GB)

These are the raw parameter-space payload and are not in git. Each line of
`EXCLUDED_BINARIES.sha256` covers one file, relative to the experiment output
directory on the machine that produced it. They can be rebuilt locally:
- `solves/*.npz` hold the dual CG vector x; v = -Pg/lambda + (beta/lambda^2) P J^T x.
- `vectors/*.pt`: float64 parameter directions, length 86,192,640.
- `motions.pt`, `candidate_modes.pt`, `compensation.pt` concatenate the above.

| bytes | path |
|---:|---|
| 2068625897 | candidate_modes.pt |
| 4826791154 | compensation.pt |
| 2758166749 | motions.pt |
| 198723 | solves/deletion_g0_restricted.npz |
| 198733 | solves/driver_g0_at_half_x.npz |
| 198718 | solves/driver_g0_at_x.npz |
| 219079 | solves/g0_full.npz |
| 219279 | solves/g1_full.npz |
| 219194 | solves/g2_full.npz |
| 219218 | solves/g3_full.npz |
| 219314 | solves/select0_candidate_r1.npz |
| 219322 | solves/select0_candidate_r2.npz |
| 219287 | solves/select0_candidate_r3.npz |
| 196553 | solves/select0_existing_fisher_r1.npz |
| 196260 | solves/select0_existing_fisher_r2.npz |
| 219251 | solves/select0_full.npz |
| 197256 | solves/select0_mean_gradient_1d_r1.npz |
| 197409 | solves/select0_random_coordinates_r1.npz |
| 197309 | solves/select0_random_coordinates_r2.npz |
| 689542838 | vectors/complement_at_zero_5797fe019613.pt |
| 689542845 | vectors/joint_amplitude0.25_5e6ba5f2ed8d.pt |
| 689542838 | vectors/joint_amplitude0.5_9c3e6ecd4e05.pt |
| 689542760 | vectors/joint_amplitude1_9e5e159601ee.pt |
| 689542711 | vectors/motion_g0_031d1da5accf.pt |
| 689542711 | vectors/motion_g1_043a15c33af8.pt |
| 689542711 | vectors/motion_g2_1191793aa7d6.pt |
| 689542711 | vectors/motion_g3_abc328da4cc7.pt |
