# Changes to the supplied package

`SHA256SUMS.txt` from the delivered archive verifies 9 of its 10 files unchanged
on this machine:

```
INTEGRATION_CHECKS.md: OK      core.py: OK        tests/test_controller.py: OK
README.md: OK                  run_toy.py: OK     tests/test_core.py: OK
controller.py: OK                                 tests/test_protected_graph_controller.py: OK
                                                  tests/test_toy_end_to_end.py: OK
run_a100.py: FAILED   <- the single edited file
```

`paired_endpoints.py` is new and is analysis only; it does not participate in any
run.

## 1. `run_a100.py` — a DTD-only arm could not start

`--protect-weight` defaults to `1.0` in `parse()`, and that value was passed into
`Config` unconditionally. With `--protect-task` absent, `protect` is `None`, so
`core.solve_response_inverse` correctly refuses:

```
ValueError: Cannot assign protection cost without a protect task
```

Every DTD-only arm therefore died on its first inverse solve. The 16 CPU tests do
not catch this because `Config`'s own default is `0.0`; only the CLI path sets
`1.0`. Evidence: `runs/online_dtd_step0/FAILED.json`.

```diff
@@ before the cfg=Config(...) construction
+    # --protect-weight defaults to 1.0 even when no protection task is requested,
+    # which makes every inverse solve refuse to run. A DTD-only arm carries no
+    # protection cost at all.
+    protection_weight = a.protect_weight if a.protect_task else 0.
     cfg=Config(radius=a.radius,fit_ridge=a.fit_ridge,inverse_ridge=a.inverse_ridge,
                 inverse_tol=a.cg_tol,inverse_iter=a.cg_maxiter,graph_iter=a.cg_maxiter,
-                protection_weight=a.protect_weight,graph_ridge=a.graph_ridge,
+                protection_weight=protection_weight,graph_ridge=a.graph_ridge,
```

## 2. Not patched, but wrong: `birth.reason` conflates two causes

`controller.OnlineController.propose` initialises

```python
birth_record=dict(proposed=False, admitted=False, reason='no_capacity_gap')
```

and only overwrites it when the birth branch is entered. The branch is guarded by
`(frozen_after is None or step_id < frozen_after)`, so a birth blocked by the
frozen arm records `no_capacity_gap`, indistinguishable from a genuine capacity
limit. `runs/online_dtd_two_frozen/steps.jsonl` step 1 shows exactly this: the
same position where the online arm births a second mode records
`proposed=False, reason='no_capacity_gap'` under `frozen_after=1`.

Left unpatched deliberately: it is a reporting label, not a mechanism, and the
supplied controller semantics are otherwise preserved.

## 3. Not patched: the default iteration budget

`--cg-maxiter 24` (and `Config.inverse_iter = 24`). Measured on this problem: the
parameter-space inverse reaches a relative residual of 8.3e-03 at 24 iterations
and 9.7e-04 at 57 (`runs/online_dtd_step0_r2` vs `_r3`). The birth gate requires
`solver.converged`, so the default silently forbids every birth. Recommended
default is 96; this run used 80 via the CLI, leaving the source untouched.
