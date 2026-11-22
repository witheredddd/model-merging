"""Compare two locked runs and export source-matched public reports."""
import argparse
import hashlib
import json
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def positive(scores):
    return bool(scores.min() > 1e-12 * max(abs(scores)))


def joint_mix(scores_a, scores_b):
    """Maximize the minimum of the 16 slopes over two fixed directions."""
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    slope = a - b
    knots = [0., 1.]
    for i, j in combinations(range(len(a)), 2):
        if slope[i] != slope[j]:
            t = (b[j] - b[i]) / (slope[i] - slope[j])
            if 0 < t < 1:
                knots.append(float(t))
    alpha = max(knots, key=lambda t: float((b + t * slope).min()))
    margins = b + alpha * slope
    return dict(weights_on_fixed_A_B=[alpha, 1 - alpha], scores=margins.tolist(),
                minimum_score=float(margins.min()), positive_all=positive(margins),
                interpretation="Feasible two-direction witness for the 16 slopes; scores in single-probe units")


def public_protocol(protocol):
    """Include protocol evidence while withholding filesystem paths and asset locations."""
    public = dict(protocol)
    source = public["source"]
    public["source"] = {key: source[key] for key in (
        "parameter_dimension", "visual_tensors", "tasks", "checkpoint_sha256",
        "probe_sha256", "inventory_sha256", "transformed_probe_sha256",
        "runtime_source_sha256", "versions") if key in source}
    return public


def maximum_change(old, new, keys):
    change = {}
    for field in keys:
        x, y = np.asarray(old[field]), np.asarray(new[field])
        change[field] = float(np.max(np.abs(x - y)))
    return change


def compare(legacy, rows_a, rows_b):
    cases = {}
    for name, a_rows in rows_a.items():
        comparisons = []
        if len(legacy[name]) != len(a_rows) or len(a_rows) != len(rows_b[name]):
            raise ValueError("Mismatched baseline or probe row counts")
        for old, a, b in zip(legacy[name], a_rows, rows_b[name]):
            if not old["epsilon"] == a["epsilon"] == b["epsilon"]:
                raise ValueError("Mismatched epsilon grid")
            for row in (a, b):
                if (not row["galerkin_passed"] or not row["interval_consistent"]
                        or row["orthogonality_residual_relative"] != row["projected_full_residual_relative"]):
                    raise ValueError("Scan failed certificate output validation")
            ab, ba = a["transfer"], b["transfer"]
            record = dict(epsilon=a["epsilon"],
                          legacy_alias_max_discrepancy=float(np.max(np.abs(
                              np.asarray(old["orthogonality_residual_relative"])
                              - old["projected_full_residual_relative"]))),
                          legacy_A_max_abs_changes=maximum_change(old, a, (
                              "graph_gram", "unit_gram", "residual_relative",
                              "projected_full_residual_relative", "reachability")),
                          legacy_A_bound_max_abs_changes=maximum_change(old["certification"], a["certification"], (
                              "exact_raw_distance_bounds", "exact_gram_absolute_error_upper",
                              "exact_unit_gram_lower", "exact_unit_gram_upper",
                              "exact_reachability_lower", "exact_reachability_upper")),
                          legacy_A_qp_weights_max_abs_changes={
                              key: maximum_change(old[key], a[key], ("weights",))["weights"]
                              for key in ("raw_hull", "unit_hull")},
                          A_to_B=ab, B_to_A=ba)
            witnesses = [label for label, result in (("A_to_B", ab), ("B_to_A", ba))
                         if result["source_positive_all"] and result["positive_all"]]
            record["joint_witness_from_transfer"] = witnesses
            if not witnesses:
                record["joint_two_direction_witness"] = joint_mix(
                    ab["source_scores"] + ab["target_scores"], ba["target_scores"] + ba["source_scores"])
            comparisons.append(record)
        cases[name] = comparisons
    return cases


def main(legacy, folder_a, folder_b, output, public=None):
    a, b = [read(folder / "protocol.json") for folder in (folder_a, folder_b)]
    if ([a["probe_index"], b["probe_index"]] != [0, 1] or a["tasks"] != b["tasks"]
            or a["epsilons"] != b["epsilons"] or a["script_sha256"] != b["script_sha256"]
            or not a["disjoint_id_check_passed"] or not b["disjoint_id_check_passed"]
            or a["per_class_counts"] != b["per_class_counts"]
            or a["source"]["checkpoint_sha256"] != b["source"]["checkpoint_sha256"]
            or a["source"]["transformed_probe_sha256"] != b["peer_transformed_probe_sha256"]
            or b["source"]["transformed_probe_sha256"] != a["peer_transformed_probe_sha256"]):
        raise ValueError("The two protocols do not describe matched, disjoint probe runs")
    x, y = [read(folder / "preview.json") for folder in (folder_a, folder_b)]
    required = {"original_geo", "original_ce", "balanced_geo", "balanced_ce"}
    if set(x) != required or set(y) != required or any(len(rows) != len(a["epsilons"])
                                                      for scan in (x, y) for rows in scan.values()):
        raise ValueError("Both runs must contain the complete four-case grid")
    comparison = dict(tasks=a["tasks"], cases=compare(read(legacy), x, y))
    write(output, comparison)
    if public is None:
        return
    manifest = {}
    for label, folder, protocol in (("probe_a", folder_a, a), ("probe_b", folder_b, b)):
        destination = public / label
        destination.mkdir(exist_ok=True, parents=True)
        for name in ("preview.json", "stability.json", "protocol.json"):
            source = folder / name
            target = destination / name
            if name == "protocol.json":
                write(target, public_protocol(protocol))
            else:
                shutil.copyfile(source, target)
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest[f"{label}/{name}"] = dict(original_sha256=digest(source), public_sha256=digest(target),
                                              identical=source.read_bytes() == target.read_bytes())
    shutil.copyfile(output, public / "transfer_comparison.json")
    write(public / "source_map.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("legacy", "folder_a", "folder_b", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--public", type=Path)
    args = parser.parse_args()
    main(args.legacy, args.folder_a, args.folder_b, args.output, args.public)
