# Eight-task online provisional controls

This directory contains the A100 runtime source and its numerical/recovery tests. Provisional structure learning and final KKT certification are independent. It depends on the adjacent `fullspace_conditional_20261116` and `eight_task_coupling_20261116` modules already in this branch.

```bash
python -m unittest discover -v
```

Real-model execution requires the existing datasets, model resources, execution protocol and checkpoint assets. Those assets and experiment evidence are not part of this source-only update. No production service is started by this publication.
