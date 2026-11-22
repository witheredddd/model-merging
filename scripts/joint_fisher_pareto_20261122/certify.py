"""Bound undamped reachability from the response bank emitted by run.py."""
import argparse
import json
from pathlib import Path

import numpy as np

from run import polytope


def certify(folder):
    protocol = json.loads((folder / "protocol.json").read_text())
    if "undamped" not in protocol.get("method", "") or "zero-start" not in protocol["method"]:
        raise ValueError("Expected undamped CGLS output from run.py")
    records = json.loads((folder / "results.json").read_text())
    datasets = {}
    for case in records:
        with np.load(folder / f"{case}_vertices.npz") as data:
            datasets[case] = (data["vertices"], data["targets"])

    result = {}
    for case, (vertices, targets) in datasets.items():
        metric = case.split("_")[0]
        candidates = []
        bank_order = []
        for other, (other_vertices, _) in datasets.items():
            other_metric = other.split("_")[0]
            if metric == other_metric:
                candidates.append(other_vertices)
                bank_order.append(other)
                continue
            # Two metrics weight the same coordinates with positive row scales.
            # Transfer an actual Jv into the current metric's chart.
            same_objective = f"{other_metric}_{case.split('_')[1]}"
            if same_objective not in datasets:
                continue
            original_target = datasets[same_objective][1].sum(axis=0)
            target = targets.sum(axis=0)
            lengths = protocol["output_lengths"]
            ratio_blocks = []
            start = 0
            for i, length in enumerate(lengths):
                classes = protocol["sample_summary"][protocol["tasks"][i]]["classes"]
                block = slice(start, start + length)
                a = original_target[block].reshape(-1, classes)
                b = target[block].reshape(-1, classes)
                ia = np.linalg.norm(a, axis=1)
                ib = np.linalg.norm(b, axis=1)
                if np.any(ia == 0) or np.any(ib == 0):
                    break
                ratio_blocks.append(np.repeat(ib / ia, classes))
                start += length
            if len(ratio_blocks) == len(lengths):
                ratio = np.concatenate(ratio_blocks)
                if not np.allclose(original_target * ratio, target, atol=1e-13, rtol=1e-12):
                    raise ValueError("Probe coordinates do not agree under reweighting")
                candidates.append(other_vertices * ratio)
                bank_order.append(other)

        bank = np.concatenate(candidates, axis=0)
        _, singular_values, vt = np.linalg.svd(bank, full_matrices=False)
        keep = singular_values > singular_values[0] * 1e-10
        basis = vt[keep]
        coordinates = targets @ basis.T
        lower = polytope(coordinates @ coordinates.T)
        mixture = np.asarray(lower["weights"]) @ coordinates
        norm = np.linalg.norm(mixture)
        scores = coordinates @ mixture / norm if norm else np.zeros(len(targets))
        # A normalized feasible direction certifies min_k <B_k,w> irrespective
        # of the optimizer's residual or the convergence of any individual solve.
        bound = max(0., float(scores.min()))
        target_norms = np.linalg.norm(targets, axis=1)
        upper_qp = polytope(targets @ targets.T)
        upper_weights = np.asarray(upper_qp["weights"])
        upper = float(np.linalg.norm(upper_weights @ targets))
        eta_lower = np.divide(np.linalg.norm(coordinates, axis=1), target_norms,
                              out=np.zeros_like(target_norms), where=target_norms > 0)
        vertex_error_upper = np.sqrt(np.maximum(0, target_norms ** 2 - (coordinates ** 2).sum(axis=1)))
        active = np.flatnonzero((np.sqrt(8) * target_norms > protocol["tau_b"])
                                & (eta_lower > protocol["tau_eta"]))
        normalized = coordinates[active] / np.linalg.norm(coordinates[active], axis=1, keepdims=True)
        unit_distance_lower = (max(0., float(np.min(scores[active] / target_norms[active])))
                               if len(active) == len(targets) else None)
        result[case] = dict(
            tasks=protocol["tasks"],
            candidate_count=len(bank), numerical_bank_rank=int(keep.sum()),
            distance_lower_bound=bound, distance_upper_bound=upper,
            reaches_all_eight=bool(bound > 1e-10 * max(target_norms)),
            normalized_common_direction_margins=scores.tolist(),
            reachability_lower_bound=eta_lower.tolist(),
            reachability_upper_bound=[1. if n else None for n in target_norms],
            vertex_error_upper_bound=vertex_error_upper.tolist(),
            projected_bank_candidate_gram=(coordinates @ coordinates.T).tolist(),
            projected_bank_unit_direction_candidate=polytope(normalized @ normalized.T),
            certified_active_tasks=active.tolist(),
            exact_unit_distance_lower_bound_if_all_active=unit_distance_lower,
            candidate_model="Feasible responses from full shared-parameter JVPs",
            witness_bank_coefficients=np.linalg.lstsq(bank.T, mixture @ basis, rcond=1e-10)[0].tolist(),
            witness_bank_order=bank_order,
            qp=lower,
        )
    for certificate in result.values():
        certificate["scope"] = "undamped run.py only; reachability_lower_bound is a subspace lower bound"
    (folder / "certificates.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    certify(parser.parse_args().folder)
