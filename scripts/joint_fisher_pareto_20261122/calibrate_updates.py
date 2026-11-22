"""Measure Rayleigh quotients of saved training increments at the fixed anchor."""
import argparse
import json
import shutil
from pathlib import Path

from runtime_location import configure_runtime

import numpy as np
import torch

from audit_projections import EmpiricalJoint


def snapshot(path, keys):
    numpy_scalar = np._core.multiarray.scalar
    numeric_dtypes = [type(np.dtype(kind)) for kind in ("int32", "int64", "float32", "float64", "bool")]
    with torch.serialization.safe_globals([numpy_scalar, np.dtype, *numeric_dtypes]):
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "full_visual_flat" in payload:
        return payload["full_visual_flat"].double()
    visual = payload["visual"]
    return torch.cat([visual[key].reshape(-1) for key in keys]).double()


def main(source, baseline, output):
    configure_runtime()
    from prepare import sha, write
    from task_runtime import load_eight_geometry

    output.mkdir(exist_ok=True, parents=True)
    (output / "runtime").mkdir(exist_ok=True)
    destination_lock = output / "runtime/probes.json"
    if not destination_lock.exists():
        shutil.copy2(source / "runtime/probes.json", destination_lock)
    protocol = json.loads((source / "protocol.json").read_text())
    lock = protocol["source"]
    g, provenance = load_eight_geometry(output / "runtime", lock["inventory_file"], lock["checkpoint"], expected=lock)
    baseline_source = json.loads((baseline / "provenance.json").read_text())
    for key in ("checkpoint_sha256", "probe_sha256", "inventory_sha256", "parameter_dimension"):
        if provenance[key] != lock[key] or provenance[key] != baseline_source[key]:
            raise ValueError(f"Mismatched source {key}")
    joint = EmpiricalJoint(g, protocol)
    charts = {metric: joint.scaled(metric)[1] for metric in joint.weights}
    scale = json.loads((baseline / "scale.json").read_text())
    trajectories = source.parent / "trajectory_resolved_geometry_20261114"
    sources = []
    for task in ("DTD", "Cars"):
        manifest = json.loads((trajectories / task / "trajectory_manifest.json").read_text())
        records = manifest["snapshots"]
        # Deterministic early/middle/late consecutive saved intervals.
        for index in (0, (len(records) - 2) // 2, len(records) - 2):
            left, right = records[index:index + 2]
            sources.append(dict(task=task, kind="task_training", steps=[left["step"], right["step"]],
                           endpoints=[left["checkpoint"], right["checkpoint"]],
                           expected_hashes=[left["checkpoint_sha256"], right["checkpoint_sha256"]]))
    anchor = Path(lock["checkpoint"])
    for step in (2, 3):
        sources.append(dict(task=None, kind="anchor_adaptive", steps=[step, step + 1],
                            endpoints=[str(anchor.with_name(f"step{n:03d}.pt")) for n in (step, step + 1)]))
    plan = dict(sources=sources, evaluation_point=lock["checkpoint"],
                task_coverage=["DTD", "Cars"], verified_probe_sha256=lock["probe_sha256"])
    plan_file = output / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError("Training increment plan changed")
    write(plan_file, plan)
    done = output / "updates.json"
    records = json.loads(done.read_text()) if done.exists() else []
    for row, accepted in zip(sources, records):
        if any(accepted[key] != value for key, value in row.items()):
            raise ValueError("Existing increment record differs from plan")
    for row in sources[len(records):]:
        paths = list(map(Path, row["endpoints"]))
        hashes = [sha(path) for path in paths]
        if "expected_hashes" in row and hashes != row["expected_hashes"]:
            raise ValueError(f"Snapshot source changed: {paths}")
        direction = (snapshot(paths[1], g.c.keys) - snapshot(paths[0], g.c.keys)).to(g.point.device)
        magnitude = float(direction.norm())
        if magnitude <= 0 or direction.numel() != g.dimension:
            raise ValueError(f"Invalid training increment: {paths}")
        direction /= magnitude
        raw = joint.raw_jvp(direction)
        responses = {metric: {task: float((factor * img).square().sum())
                              for task, factor, img in zip(joint.tasks, factors, raw)}
                     for metric, factors in charts.items()}
        records.append(dict(**row, actual_hashes=hashes, parameter_update_norm=magnitude,
                            rayleigh_by_response_task=responses,
                            rayleigh={metric: sum(value.values()) for metric, value in responses.items()}))
        write(output / "updates.json", records)
        print(row["task"], row["steps"], records[-1]["rayleigh"], flush=True)
    centers = {}
    for task in ("DTD", "Cars"):
        expert = np.sqrt(np.prod([m["expert"][task] for m in scale["rayleigh_raw"].values()]))
        local = np.median([np.sqrt(np.prod(list(row["rayleigh"].values())))
                           for row in records if row["task"] == task])
        centers[task] = dict(expert=float(expert), local=float(local),
                             equal_family_center=float(np.median([expert, local])))
    reference = float(np.median([val["equal_family_center"] for val in centers.values()]))
    write(output / "reference.json", dict(provisional=True, matched_tasks=list(centers),
          missing_task_specific_local_updates=[task for task in joint.tasks if task not in centers],
          anchor_adaptive_used_for_reference=False, centers=centers, R_ref=reference,
          convention="geometric mean across measures, median within task/family, equal family and matched task weights",
          epsilon_scan=[reference * factor for factor in (.1, .3, 1., 3., 10.)],
          gradient_baseline=scale["epsilon_raw"], provenance=provenance, script_sha256=sha(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.source, args.baseline, args.output)
