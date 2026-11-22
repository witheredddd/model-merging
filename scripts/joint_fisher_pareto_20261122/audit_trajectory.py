"""Compare local and fixed-anchor prediction responses of real optimizer steps."""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch

from audit_projections import EmpiricalJoint
from calibrate_updates import snapshot


def main(source, calibration, output):
    root = Path("/home/shangxueyuan/merging/tsv/scripts")
    sys.path[:0] = [str(root / "eight_task_coupling_20261116"),
                    str(root / "fullspace_conditional_20261116")]
    from precision import Float64Ops
    from prepare import sha, write
    from run import ideal
    from task_runtime import load_eight_geometry

    protocol = json.loads((source / "protocol.json").read_text())
    lock = protocol["source"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "runtime").mkdir()
    shutil.copy2(source / "runtime/probes.json", output / "runtime/probes.json")
    g, provenance = load_eight_geometry(output / "runtime", lock["inventory_file"], lock["checkpoint"])
    for key in ("checkpoint_sha256", "inventory_sha256", "probe_sha256"):
        if provenance[key] != lock[key]:
            raise ValueError(f"Mismatched {key}")
    joint = EmpiricalJoint(g, protocol)
    rows = json.loads((calibration / "updates.json").read_text())
    selected = [row for row in rows if row["kind"] == "task_training" and row["steps"] == [0, 1]]
    if {row["task"] for row in selected} != {"DTD", "Cars"} or len(selected) != 2:
        raise ValueError("Expected two genuine single-step increments")
    write(output / "protocol.json", dict(provenance=provenance, step_records=selected,
          comparison_coordinate="Common ambient sqrt-probability embedding; local JVP at actual step start",
          source_sha256=sha(__file__)))
    results = {}
    for row in selected:
        source_task = row["task"]
        left, right = map(Path, row["endpoints"])
        if [sha(left), sha(right)] != row["actual_hashes"]:
            raise ValueError("Trajectory endpoint hash mismatch")
        theta = snapshot(left, g.c.keys).to(g.point.device)
        after = snapshot(right, g.c.keys).to(g.point.device)
        v = after - theta
        local = Float64Ops(g.c, theta)
        metrics = {metric: {} for metric in joint.weights}
        for i, task in enumerate(joint.tasks):
            write(output / "status.json", dict(stage="auditing", source_task=source_task, response_task=task))
            start_logits, labels = g.c.baseline(task, g.probe, theta)
            end_logits, labels_after = g.c.baseline(task, g.probe, after)
            if not torch.equal(labels, labels_after):
                raise ValueError("Probe order changed")
            delta = (2 * (end_logits.softmax(-1).sqrt() - start_logits.softmax(-1).sqrt())).cuda()
            logp0 = start_logits.log_softmax(-1)
            logp1 = end_logits.log_softmax(-1)
            ce_change = (logp0 - logp1)[torch.arange(len(labels)), labels].cuda()
            ce_direction, _ = ideal(start_logits.cuda(), labels.cuda())
            local_unscaled = local.j(v, task, g.probe).reshape_as(delta) * math.sqrt(len(labels))
            anchor_unscaled = g.jvp(task, v).reshape_as(delta) * math.sqrt(len(labels))
            for metric, task_weights in joint.weights.items():
                weights = task_weights[i]
                chart = (weights[:, None] / 8).sqrt()
                finite = delta * chart
                actual = local_unscaled * chart
                anchored = anchor_unscaled * chart
                norm = float(finite.norm())
                denominator = max(norm, 1e-300)
                metrics[metric][task] = dict(finite_response_norm=norm,
                    local_jvp_norm=float(actual.norm()), anchor_jvp_norm=float(anchored.norm()),
                    local_jvp_error_relative_to_finite=float((actual - finite).norm()) / denominator,
                    anchor_jvp_error_relative_to_finite=float((anchored - finite).norm()) / denominator,
                    local_vs_anchor_jvp_difference_norm=float((actual - anchored).norm()),
                    realized_ce_change=float(weights @ ce_change),
                    local_linear_ce_change=-float((weights[:, None] * ce_direction["ce"] * local_unscaled).sum()))
        results[source_task] = dict(steps=row["steps"], step_norm=float(v.norm()), per_measure=metrics)
        write(output / "trajectory.json", results)
    write(output / "status.json", dict(stage="complete", source_tasks=list(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "calibration", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    main(args.source, args.calibration, args.output)
