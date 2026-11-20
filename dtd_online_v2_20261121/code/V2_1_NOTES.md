# V2.1 — what changed relative to V2

V2 is frozen. This directory is V2 plus the changes agreed after the Stage A
review and after the homogeneity correction to the backtracking argument.

## A. The one thing that did *not* change

Backtracking still scales the existing solution (`trial = v * 2**-k`) instead of
re-solving the trust region at each level. That is exact here, because the
target is itself the budgeted step: `y_rho = J(rho * u/||u||)`, so scaling both
the radius and the target by `c` scales the optimum by exactly `c`.

    z*(c*rho, c*y) = c * z*(rho, y)

Verified to 0.0 relative difference for c = 1/2, 1/4, 1/8 (`test_v2_1_protocol`).
The alternative — keeping `y` fixed and shrinking only the radius — is a
different problem and does require a re-solve; it is not what backtracking does
here. A test loads both interpretations and shows the second deviates by 1-35%.

## B. Changes

| # | change | why |
|---|---|---|
| 1 | `Fit` carries `radius`, `multiplier`, `constrained` | the trust region's activity was previously unobservable |
| 2 | `fit_controls` answers with `Y@coeff` instead of a fresh `jvp` | `J(A z) = (J A) z` exactly; saves one JVP per fit, two per birth step |
| 3 | report `previous_fit_multiplier` / `previous_fit_constrained` | replaces the hard-coded `previous_fit_is_radius_constrained=True`, which was an assertion, and whose test asserted a constant |
| 4 | no-birth reason re-ordered: zero target → residual → capacity → frozen | if the residual is already inside tolerance no arm would birth, so blaming the freeze (or the cap) misattributes the cause; the arm policy is now recorded separately in `birth.arm_policy` |
| 5 | redundant `finite_cap` after the constrained fit removed; replaced by an explicit budget check that raises | the clip was a no-op that read as if clipping still happened |
| 6 | `inverse_response_error` renamed `unconstrained_inverse_response_error` | it belongs to the unbudgeted inverse; next to the budgeted `new_error` it invited the exact misreading that started this review |
| 7 | `executed_delivery_error` and `executed_shape_error` recorded, with `executed_scale` | delivery includes the loss from guard backtracking; shape is the scale-free structural fit. When no step executes both are `null` rather than a fabricated 1.0 |
| 8 | birth evidence is same-scale: `_same_scale_evidence` | comparing one span's accepted step with another span's *larger* accepted step mixes direction quality with step size. Both spans are now evaluated at every level `c_k`, and the record separates `same_scale_gain` from `feasibility_gain` |
| 9 | reference arm records the same two executed errors and the scale | parity with the controlled arms |
| 10 | reference arm's protection configuration unified | it used to construct the controller with `protect=None` even when a protection task was requested |

The definitions now recorded per step:

    e_delivery = ||J delta_exec - y|| / ||y||
    e_shape    = ||J delta_exec - c*y|| / ||c*y||,   c = executed_scale

`e_shape` is invariant to `c`; `e_delivery` grows as the guard backs off. Both
are computed from the final written delta, not from a fit or a candidate.

## C. Test coverage added

`tests/test_v2_1_protocol.py`:

- KKT residual on a **non-orthonormal** basis: `b - K z = nu * (A^T A) z`, with
  `nu` equal to the recorded multiplier and `nu > 0` when the constraint binds.
  This is the case where the `A^T A` metric matters; an `I` metric is feasible
  but suboptimal (measured +0.54% objective on a basis with Gram off-diagonal
  0.35).
- Homogeneity identity, plus the fixed-target contrast.
- `e_delivery > e_shape` when the guard backtracks to half.
- No-write steps report `null` for both errors.
- Reason ordering: residual inside tolerance wins over the frozen policy.
- Same-scale evidence distinguishes `same_scale_gain` from `feasibility_gain`.

27/27 tests pass. Toy end-to-end unchanged from V2: 0.5133 / 0.6335 / 0.4844.

## D. Not changed

- `solve_response_inverse` is still unbudgeted (it proposes a *direction*).
- `selectivity_eigenvalues` is still a diagnostic with no budget.
- Stage C (real one-epoch training) still needs a data adapter; `run_a100.py`
  continues to use the frozen development probes and says so.
