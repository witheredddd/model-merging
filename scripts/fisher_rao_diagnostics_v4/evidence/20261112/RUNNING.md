# A100 diagnostic deployment

**Stopped on user request before further code review.** See `STOPPED_BY_USER.json`. Only the four seed-11/rank-4 primary cases completed; the 1% tolerance rerun was interrupted. The full four-seed protocol, offline geometry stage and functional-control stage did not complete. Existing results are retained. No experiment may be resumed without a new explicit user instruction.

The numerical/protocol suite passed 28 tests on A100. The exact-source small CLIP integration check passed; see `smoke_report.json`.

The full frozen protocol was launched on 2026-11-12 at 03:16 CST, using `fisher-rao-diagnostics-v4-20261112.service`. Code lives in `/home/shangxueyuan/merging/tsv/scripts/fisher_rao_diagnostics_v4`; output is `/media/shangxueyuan/data1/fisher_rao_diagnostics_v4_20261112`. This is a launch record, not a claim that the full protocol is already complete. Final evidence supersedes this status when added.

There are 32 primary four-seed/rank-4/rank-8 controls, plus eight predeclared development-seed tolerance reruns. The main stage is followed by offline geometry replay and functional/same-capacity controls. No official test, curvature, fiber optimization, dictionary refresh or EC is enabled. Source/config hashes are locked; old v3 results are preserved.
