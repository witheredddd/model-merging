# Fixed-space coordination diagnostic code

Contains synchronized nested profiling, frozen-checkpoint CG replay, learned-subspace correction comparisons and a restart-only control. The diagnostic producer must be paused at a committed checkpoint. Maintenance replays use an isolated writable cache.

```bash
python -m unittest test_audit -v
```

Model execution additionally requires the original private model/data/checkpoint resources and output protocol at the paths specified in the source. This source-only publication includes no audit reports, measured results, scheduler snapshots or CG arrays.
