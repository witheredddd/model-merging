"""Shared block Krylov Galerkin solves with full-space residual certificates."""
import argparse
from copy import copy
import json
import math
import shutil
from itertools import combinations
from pathlib import Path

from runtime_location import configure_runtime

import numpy as np
import scipy
import torch

from audit_projections import EmpiricalJoint
from run import polytope


def weight_stability(gram, error, qp, eigenvalue_lower):
    """Enclose optimizer weights using strong convexity and elementwise Gram bounds."""
    if not qp["qp_passed"] or eigenvalue_lower <= 0:
        return dict(certified=False, reason="Candidate QP or positive curvature not certified")
    a = np.asarray(qp["weights"])
    scores = gram @ a
    gradient_error = np.linalg.norm(error @ a)
    gap = qp["dual_gap"]
    radius = min(np.sqrt(2), (gradient_error + np.sqrt(
        gradient_error ** 2 + 2 * eigenvalue_lower * gap)) / eigenvalue_lower)
    lo, hi = np.maximum(0, a - radius), np.minimum(1, a + radius)
    active = np.flatnonzero(lo > 0)
    inactive = []
    for i in np.flatnonzero(lo == 0):
        for j in active:
            if i == j:
                continue
            margin = scores[i] - scores[j]
            uncertainty = (np.linalg.norm(gram[i] - gram[j]) * radius
                           + float(np.max(error[i] + error[j])))
            if margin > uncertainty:
                hi[i] = 0
                inactive.append(i)
                break
    complete = len(active) + len(inactive) == len(a)
    return dict(certified=complete, weight_intervals=np.column_stack((lo, hi)).tolist(),
                weight_l2_radius=float(radius), definitely_positive=active.tolist(),
                definitely_zero=inactive, minimum_face=active.tolist() if complete else None)


def gram_bounds(gram, correction, squared_error_bounds):
    center = gram + correction
    radius = np.sqrt(np.outer(squared_error_bounds, squared_error_bounds))
    low, high = center - radius, center + radius
    np.fill_diagonal(low, np.maximum(0, np.diag(center)))
    error = np.maximum(np.abs(low - gram), np.abs(high - gram))
    # E^T E is PSD, with spectral norm bounded by its trace.
    min_eig = np.linalg.eigvalsh(center)[0]
    eig = [max(0., float(min_eig)), max(0., float(min_eig + squared_error_bounds.sum()))]
    d_low, d_high = np.diag(low), np.maximum(0, np.diag(high))
    denom_min = np.sqrt(np.outer(d_low, d_low))
    denom_max = np.sqrt(np.outer(d_high, d_high))
    valid = (denom_min > 0) & (denom_max > 0)
    unit_low, unit_high = np.full_like(gram, -1.), np.ones_like(gram)
    unit_low[valid] = np.maximum(-1, np.minimum(low[valid] / denom_min[valid], low[valid] / denom_max[valid]))
    unit_high[valid] = np.minimum(1, np.maximum(high[valid] / denom_min[valid], high[valid] / denom_max[valid]))
    np.fill_diagonal(unit_low, 1)
    np.fill_diagonal(unit_high, 1)
    return error, eig, unit_low, unit_high, d_low, d_high


def solve_grid(parameter_gram, prediction_gram, rhs, rhs_norms, target_norms, shifts, residual,
               rtol=1e-5, transfer_rhs=None):
    rows = []
    for epsilon in shifts:
        system = prediction_gram + epsilon * parameter_gram
        coefficients = np.linalg.solve(system, rhs)
        reduced = system @ coefficients - rhs
        reduced_relative = np.linalg.norm(reduced, axis=0) / rhs_norms
        condition = float(np.linalg.cond(system))
        gram = coefficients.T @ system @ coefficients
        gram = (gram + gram.T) / 2
        parameter = epsilon * coefficients.T @ parameter_gram @ coefficients
        norms = np.sqrt(np.diag(gram))
        unit = gram / np.outer(norms, norms)
        errors, projected_residual = residual(epsilon, coefficients)
        residual_relative = np.linalg.norm(projected_residual, axis=0) / rhs_norms
        consistency = np.linalg.norm(projected_residual + reduced, axis=0) / rhs_norms
        galerkin_ok = bool(np.isfinite(condition) and condition * np.finfo(float).eps < 1e-5 and max(
            np.max(reduced_relative), np.max(residual_relative), np.max(consistency)) <= 1e-9)
        raw_hull, unit_hull = polytope(gram), polytope(unit)
        row = dict(epsilon=float(epsilon), graph_gram=gram.tolist(), unit_gram=unit.tolist(),
                   reachability=(norms / target_norms).tolist(),
                   parameter_fraction=(np.diag(parameter) / np.diag(gram)).tolist(),
                   graph_gram_eigenvalues=np.linalg.eigvalsh(gram).tolist(),
                   residual_relative=errors, orthogonality_residual_relative=residual_relative.tolist(),
                   reduced_residual_relative=reduced_relative.tolist(),
                   projected_full_residual_relative=residual_relative.tolist(),
                   reduced_full_consistency_relative=consistency.tolist(),
                   reduced_condition_number=condition if np.isfinite(condition) else None,
                   galerkin_passed=galerkin_ok, raw_hull=raw_hull, unit_hull=unit_hull,
                   exact_projection_tolerance_met=galerkin_ok and max(errors) <= rtol)
        rows.append(row)
        if transfer_rhs is not None:
            mixture = np.asarray(raw_hull["weights"])
            motion = coefficients @ mixture
            norm = math.sqrt(max(0., float(motion @ system @ motion)))
            source_scores, target_scores = rhs.T @ motion, transfer_rhs.T @ motion
            if norm > 0:
                source_scores, target_scores = source_scores / norm, target_scores / norm
            margin = float(target_scores.min())
            row["transfer"] = dict(source_graph_norm_before_normalization=norm,
                source_scores=source_scores.tolist(), target_scores=target_scores.tolist(),
                minimum_target_score=margin,
                source_positive_all=bool(norm > 0 and source_scores.min() > 1e-12 * max(abs(source_scores))),
                positive_all=bool(norm > 0 and margin > 1e-12 * max(abs(target_scores))))
        if not galerkin_ok:
            row["certification"] = None
            continue

        squared_error_bounds = (np.asarray(errors) * rhs_norms) ** 2 / epsilon
        # Keep the measured cross term even when Galerkin orthogonality is
        # numerically very accurate. r = H - G V and V = Q C.
        cross = projected_residual.T @ coefficients
        correction = cross + cross.T
        bound, eig, unit_lo, unit_hi, diagonal_lo, diagonal_hi = gram_bounds(
            gram, correction, squared_error_bounds)
        unit_error = np.maximum(np.abs(unit_lo - unit), np.abs(unit_hi - unit))
        unit_eig_lo = max(0., float(eig[0] / diagonal_hi.max()))
        norm_lo = float(diagonal_lo.min())
        unit_eig_hi = min(1., float(eig[1] / norm_lo)) if norm_lo > 0 else 1.
        mixture = np.asarray(raw_hull["weights"])
        candidate_distance = raw_hull["distance"]
        distance_lower = (max(0., float(((gram + cross) @ mixture).min() / candidate_distance))
                          if candidate_distance > 0 else 0.)
        distance_upper = min(
            float(np.sqrt(max(0., mixture @ (gram + correction) @ mixture
                              + (mixture @ np.sqrt(squared_error_bounds)) ** 2))),
            float(1 / np.linalg.norm(1 / target_norms)))
        numerical_floor = float(64 * np.finfo(float).eps * max(1., condition) * max(target_norms))
        consistent = (distance_lower <= distance_upper + numerical_floor
                      and np.all(unit_lo <= unit_hi + 64 * np.finfo(float).eps)
                      and np.all(diagonal_lo <= diagonal_hi + numerical_floor * max(target_norms))
                      and eig[0] <= eig[1] and unit_eig_lo <= unit_eig_hi)
        row["interval_consistent"] = bool(consistent)
        if not consistent:
            row["certification"] = None
            continue
        # Round-off size inversions are conservatively widened, never swapped.
        distance_upper = max(distance_upper, distance_lower)
        unit_hi = np.maximum(unit_hi, unit_lo)
        diagonal_hi = np.maximum(diagonal_hi, diagonal_lo)
        row["certification"] = dict(
                         exact_reachability_lower=(np.sqrt(diagonal_lo) / target_norms).tolist(),
                         exact_reachability_upper=np.minimum(1, np.sqrt(diagonal_hi) / target_norms).tolist(),
                         exact_raw_distance_bounds=[distance_lower, distance_upper],
                         raw_distance_interval_width=distance_upper - distance_lower,
                         distance_numerical_floor_heuristic=numerical_floor,
                         distance_lower_to_numerical_floor=distance_lower / numerical_floor,
                         exact_gram_absolute_error_upper=bound.tolist(),
                         exact_gram_min_eigenvalue_bounds=eig,
                         exact_unit_gram_lower=unit_lo.tolist(), exact_unit_gram_upper=unit_hi.tolist(),
                         exact_unit_gram_min_eigenvalue_bounds=[unit_eig_lo, unit_eig_hi],
                         raw_weight_stability=weight_stability(gram, bound, raw_hull, eig[0]),
                         unit_weight_stability=weight_stability(unit, unit_error, unit_hull, unit_eig_lo))
    return rows


def cross_shift_audit(rows):
    certs = [row["certification"] for row in rows]
    if not all(cert is not None for cert in certs):
        return None
    pairs = []
    for i, j in combinations(range(len(rows)), 2):
        a, b = certs[i], certs[j]
        lo = np.array(a["exact_unit_gram_lower"]) - np.array(b["exact_unit_gram_upper"])
        hi = np.array(a["exact_unit_gram_upper"]) - np.array(b["exact_unit_gram_lower"])
        pairs.append(dict(epsilons=[rows[i]["epsilon"], rows[j]["epsilon"]],
                          maximum_element_change_bounds=[
                              float(np.maximum(np.maximum(lo, -hi), 0).max()),
                              float(np.maximum(np.abs(lo), np.abs(hi)).max())]))
    faces = {}
    for key in ("raw", "unit"):
        reports = [cert[f"{key}_weight_stability"] for cert in certs]
        known = [tuple(report["minimum_face"]) for report in reports if report["certified"]]
        faces[key] = ("changed" if len(set(known)) > 1 else
                      "same" if len(known) == len(reports) else "undetermined")
    return dict(unit_gram_pairwise_bounds=pairs, minimum_face=faces)


def orthogonal_block(q, actions):
    new = []
    for action in actions:
        candidate = action.clone()
        baseline = float(candidate.norm())
        if baseline == 0:
            continue
        for _ in range(2):
            for basis in (*q, *new):
                candidate.add_(basis, alpha=-float(basis @ candidate) / float(basis @ basis))
        size = float(candidate.norm())
        if size > 1e-10 * baseline:
            new.append(candidate / size)
    return new


def independent_probe_lock(primary, historical, sample_summary):
    """Keep the original IDs and match its per-class sample count on split 1."""
    for task, record in primary["tasks"].items():
        source = historical["tasks"][task]
        old_ids = record["ids"][0]
        if not set(old_ids).issubset(source["ids"][0]):
            raise ValueError(f"Primary {task} IDs do not match the historical split")
        count = sample_summary[task]["samples"]
        ids = source["ids"][1][:count]
        if len(set(ids)) != count or set(ids) & set(source["ids"][0]):
            raise ValueError(f"Missing or overlapping independent IDs for {task}")
        record["ids"] = [old_ids, ids]
    return primary


def main(source, baseline, calibration, output, max_blocks=1, rtol=1e-5, selected=(),
         independent_probe_source=None, source_probe=None):
    configure_runtime()
    from prepare import sha, write
    from task_runtime import load_eight_geometry, probe_signatures

    output.mkdir(exist_ok=False, parents=True)
    (output / "runtime").mkdir()
    protocol = json.loads((source / "protocol.json").read_text())
    grid = json.loads((calibration / "reference.json").read_text())
    lock = protocol["source"]
    base = json.loads((baseline / "provenance.json").read_text())
    for key in ("checkpoint_sha256", "probe_sha256", "inventory_sha256", "parameter_dimension"):
        if len({lock[key], base[key], grid["provenance"][key]}) != 1:
            raise ValueError(f"Mismatched source {key}")
    primary_file = source / "runtime/probes.json"
    if sha(primary_file) != lock["probe_sha256"]:
        raise ValueError("Primary probe lock digest mismatch")
    destination = output / "runtime/probes.json"
    expected = dict(lock)
    probe = 0
    if independent_probe_source is not None:
        if sha(independent_probe_source) != protocol["original_probe_sha256"]:
            raise ValueError("Historical probe split digest mismatch")
        new_lock = independent_probe_lock(json.loads(primary_file.read_text()),
                                          json.loads(independent_probe_source.read_text()),
                                          protocol["sample_summary"])
        write(destination, new_lock)
        expected.pop("transformed_probe_sha256", None)
        expected["probe_sha256"] = sha(destination)
        probe = 1 if source_probe is None else source_probe
    else:
        shutil.copy2(primary_file, destination)
    g, provenance = load_eight_geometry(output / "runtime", lock["inventory_file"],
                                         lock["checkpoint"], probe=probe, expected=expected)
    if provenance["parameter_dimension"] != lock["parameter_dimension"]:
        raise ValueError("Parameter dimension mismatch")
    joint = EmpiricalJoint(g, protocol)
    other = None
    if independent_probe_source is not None:
        peer = copy(g)
        peer.probe = 1 - probe
        other = EmpiricalJoint(peer, protocol)
        if joint.class_counts != other.class_counts:
            raise ValueError("The probe batches differ in per-class sample counts")
    shifts = [*grid["epsilon_scan"], grid["gradient_baseline"]]
    write(output / "protocol.json", dict(epsilons=shifts, tasks=joint.tasks, source=provenance,
          calibration_sha256=sha(calibration / "reference.json"), script_sha256=sha(__file__),
          source_protocol_sha256=sha(source / "protocol.json"), probe_index=probe,
          independent_probe_source_sha256=sha(independent_probe_source) if other else None,
          sample_counts={t: len(w) for t, w in zip(joint.tasks, joint.weights["original"])},
          per_class_counts=joint.class_counts,
          peer_per_class_counts=other.class_counts if other else None,
          disjoint_id_check_passed=other is not None,
          peer_transformed_probe_sha256=probe_signatures(g.c, peer.probe) if other else None,
          scipy_version=scipy.__version__,
          direct_source_sha256={name: sha(Path(__file__).with_name(name)) for name in
                                ("run.py", "audit_projections.py")},
          max_blocks=max_blocks, full_relative_tolerance=rtol, reduced_relative_tolerance=1e-9,
          method="Shared block Krylov seeded by current full gradients; shifts reuse every JVP/VJP"))
    cases, stability = {}, {}
    storage = g.device if max_blocks == 1 and other is None else torch.device("cpu")
    for metric in joint.weights:
        charts, multipliers, _, jvp, vjp = joint.scaled(metric)
        peer_charts, peer_multipliers, *_ = other.scaled(metric) if other else (None, None)
        for objective in ("geo", "ce"):
            name = f"{metric}_{objective}"
            if selected and name not in selected:
                continue
            seeds, images, fisher_columns, rhs_norms, target_norms, peer_gradients = [], [], [], [], [], []
            for idx, task in enumerate(joint.tasks):
                target = charts[idx] * joint.ideals[idx][objective].flatten()
                grad = g.vjp(task, multipliers[idx] * target)
                rhs_norms.append(float(grad.norm()))
                if rhs_norms[-1] == 0:
                    raise ValueError(f"Degenerate zero gradient for {name}/{task}")
                seeds.append((grad / rhs_norms[-1]).to(storage))
                target_norms.append(float(target.norm()))
                if other:
                    peer_target = peer_charts[idx] * other.ideals[idx][objective].flatten()
                    peer_gradients.append(peer.vjp(task, peer_multipliers[idx] * peer_target).to(storage))
            q = orthogonal_block([], seeds)
            history = []
            start = 0
            for block in range(1, max_blocks + 1):
                for idx in range(start, len(q)):
                    write(output / "status.json", dict(stage="evaluating_basis", block=block,
                          basis_index=idx, case=name, finished=list(cases)))
                    image = jvp(q[idx].to(g.device))
                    images.append(image)
                    fisher_columns.append(vjp(image).to(storage))
                pg = np.array([[float(left @ right) for right in q] for left in q])
                z = torch.stack(images)
                fg = (z @ z.T).cpu().numpy()
                rhs = np.array([[float(direction @ seed) * norm for seed, norm in zip(seeds, rhs_norms)]
                                for direction in q])
                transfer_rhs = (np.array([[float(direction @ grad) for grad in peer_gradients] for direction in q])
                                if other else None)

                def residual(epsilon, coefficients):
                    errors = []
                    projected = np.empty((len(q), len(rhs_norms)))
                    for i, norm in enumerate(rhs_norms):
                        vector = norm * seeds[i]
                        for j in range(len(q)):
                            vector.add_(q[j], alpha=-float(epsilon * coefficients[j, i]))
                            vector.add_(fisher_columns[j], alpha=-float(coefficients[j, i]))
                        errors.append(float(vector.norm()) / norm)
                        projected[:, i] = [float(direction @ vector) for direction in q]
                    return errors, projected

                result = solve_grid(pg, fg, rhs, np.array(rhs_norms), np.array(target_norms),
                                    shifts, residual, rtol, transfer_rhs)
                history.append(dict(block=block, dimension=len(q),
                    max_full_residual=[max(row["residual_relative"]) for row in result]))
                cases[name] = result
                stability[name] = dict(history=history, five_shift_comparison=cross_shift_audit(result[:-1]))
                write(output / "preview.json", cases)
                write(output / "stability.json", stability)
                if block == max_blocks or all(row["exact_projection_tolerance_met"] for row in result):
                    break
                new = orthogonal_block(q, fisher_columns[start:])
                if not new:
                    break
                start = len(q)
                q.extend(new)
            print(name, [(row["epsilon"], max(row["residual_relative"])) for row in cases[name]], flush=True)
    write(output / "status.json", dict(stage="complete", finished=list(cases),
          passed={case: all(row["exact_projection_tolerance_met"] for row in rows) for case, rows in cases.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "baseline", "calibration", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--max-blocks", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--case", action="append", default=[],
                        choices=("original_geo", "original_ce", "balanced_geo", "balanced_ce"))
    parser.add_argument("--independent-probe-source", type=Path)
    parser.add_argument("--source-probe", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.max_blocks < 1 or not 0 < args.rtol < 1:
        parser.error("positive block budget and 0 < rtol < 1 required")
    if args.source_probe is not None and args.independent_probe_source is None:
        parser.error("--source-probe requires --independent-probe-source")
    main(args.source, args.baseline, args.calibration, args.output, args.max_blocks, args.rtol,
         args.case, args.independent_probe_source, args.source_probe)
