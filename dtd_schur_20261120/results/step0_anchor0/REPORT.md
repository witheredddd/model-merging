# DTD Schur conditional compensation, step 0

Anchor 0; active domain 48 attention/MLP weight matrices (84934656 of 86192640 coordinates).

## Numerical gates

- JVP/VJP adjointness relative error: 6.136e-15
- kappa = ||Jg||^2/||g||^2 = 553.6407; beta = 1.806e-03
- Candidate modes: 3; max orthonormality error 2.168e-18
- Compensation linearity relative error: 2.181e-07

## Independent rank selection (probe 2; probe 3 never used)

| demand | family | rank | proxy deletion | / abs(q_full) | Schur identity error |
|---|---|---:|---:|---:|---:|
| 0 | candidate | 1 | 4.20504 | 9.254e-03 | 5.933e-05 |
| 0 | candidate | 2 | 6.35414 | 1.398e-02 | 1.334e-04 |
| 0 | candidate | 3 | 7.1923 | 1.583e-02 | 1.376e-04 |
| 0 | mean_gradient_1d | 1 | 11.5805 | 2.548e-02 | 3.071e-04 |
| 0 | existing_fisher | 1 | 0.918974 | 2.022e-03 | -5.066e-05 |
| 0 | existing_fisher | 2 | 1.208 | 2.658e-03 | -1.040e-04 |
| 0 | random_coordinates | 1 | 2.26605e-05 | 4.987e-08 | 4.766e-10 |
| 0 | random_coordinates | 2 | 2.36622e-05 | 5.207e-08 | 3.875e-10 |

Rank advantages against dimension-matched baselines:

| rank | candidate deletion | best matched baseline | advantage |
|---:|---:|---:|---:|
| 1 | 4.20504 | 11.5805 | -7.37544 |
| 2 | 6.35414 | 1.208 | 5.14614 |
| 3 | 7.1923 | n/a | n/a |

Selected rank: 2. Deletion grows monotonically with the size of the forbidden subspace, so the rank is chosen by the advantage over matched alternatives, not by deletion itself. The H-spectrum of the construction motions is flat (see MODES.json), so nothing here supports a low-rank movement claim.

## Schur deletion on the construction demand

- q(full) = -490.092165; q(restricted) = -449.329860
- proxy deletion = 40.762305
- Schur identity error = 5.975e-04

## Conditional compensation

| amplitude | driver norm | compensation norm | compensation/driver | q(joint) - q(full) |
|---|---:|---:|---:|---:|
| 0.25 | 2.04823 | 0.442019 | 0.2158 | 22.9289 |
| 0.5 | 4.09647 | 0.884039 | 0.2158 | 10.1907 |
| 1 | 8.19294 | 1.76808 | 0.2158 | -8.51102e-08 |

## Finite DTD outcomes (delta against the anchor)

| state | probe | CE change | Top-1 change | flip rate | parameter norm |
|---|---:|---:|---:|---:|---:|
| full_optimum | 0 | 5.12859 | -0.41 | 0.9350 | 28.1534 |
| full_optimum | 2 | 5.1246 | -0.40566 | 0.9528 | 28.1534 |
| full_optimum | 3 | 4.95254 | -0.37234 | 0.9255 | 28.1534 |
| restricted_optimum | 0 | 4.01107 | -0.41 | 0.9350 | 27.0745 |
| restricted_optimum | 2 | 4.00785 | -0.40566 | 0.9528 | 27.0745 |
| restricted_optimum | 3 | 3.86122 | -0.37234 | 0.9362 | 27.0745 |
| joint_amplitude_0.25 | 0 | 4.27424 | -0.41 | 0.9350 | 27.1063 |
| joint_amplitude_0.25 | 2 | 4.27147 | -0.40566 | 0.9528 | 27.1063 |
| joint_amplitude_0.25 | 3 | 4.12004 | -0.37234 | 0.9255 | 27.1063 |
| driver_only_0.25 | 0 | 4.43296 | -0.385 | 0.9850 | 2.04823 |
| driver_only_0.25 | 2 | 4.43879 | -0.396226 | 0.9717 | 2.04823 |
| driver_only_0.25 | 3 | 4.53941 | -0.361702 | 0.9787 | 2.04823 |
| joint_amplitude_0.5 | 0 | 4.55734 | -0.41 | 0.9350 | 27.2994 |
| joint_amplitude_0.5 | 2 | 4.55382 | -0.40566 | 0.9528 | 27.2994 |
| joint_amplitude_0.5 | 3 | 4.3968 | -0.37234 | 0.9255 | 27.2994 |
| driver_only_0.5 | 0 | 3.77039 | -0.405 | 0.9900 | 4.09647 |
| driver_only_0.5 | 2 | 3.84713 | -0.40566 | 1.0000 | 4.09647 |
| driver_only_0.5 | 3 | 3.74829 | -0.351064 | 0.9574 | 4.09647 |
| joint_amplitude_1 | 0 | 5.12858 | -0.41 | 0.9350 | 28.1534 |
| joint_amplitude_1 | 2 | 5.1246 | -0.40566 | 0.9528 | 28.1534 |
| joint_amplitude_1 | 3 | 4.95254 | -0.37234 | 0.9255 | 28.1534 |
| driver_only_1 | 0 | 3.10715 | -0.395 | 0.9800 | 8.19294 |
| driver_only_1 | 2 | 3.1311 | -0.386792 | 1.0000 | 8.19294 |
| driver_only_1 | 3 | 3.02364 | -0.382979 | 1.0000 | 8.19294 |

Probes 0/2/3 are disjoint training-side development images. Probe 3 is used only for the table above. The quadratic proxy is not the network CE Hessian: a large proxy deletion with no held-out CE difference indicates local proxy structure, not an irreplaceable task parameter.