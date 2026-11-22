"""One shared full-gradient Galerkin step for all six shifts. Audit true residuals."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from audit_projections import EmpiricalJoint
from run import polytope


def solve_grid(parameter_gram, prediction_gram, rhs, rhs_norms, target_norms, shifts, residual):
    rows = []
    for epsilon in shifts:
        system = prediction_gram + epsilon * parameter_gram
        coefficients = np.linalg.solve(system, rhs)
        gram = coefficients.T @ system @ coefficients
        gram = (gram + gram.T) / 2
        parameter = epsilon * coefficients.T @ parameter_gram @ coefficients
        norms = np.sqrt(np.diag(gram))
        unit = gram / np.outer(norms, norms)
        errors = residual(epsilon, coefficients)
        raw_hull, unit_hull = polytope(gram), polytope(unit)
        squared_error_bounds = (np.asarray(errors) * rhs_norms) ** 2 / epsilon
        mixture = np.asarray(raw_hull["weights"])
        candidate_distance = raw_hull["distance"]
        distance_lower = (max(0., float((gram @ mixture).min() / candidate_distance))
                          if candidate_distance > 0 else 0.)
        distance_upper = min(
            float(np.sqrt(candidate_distance ** 2 + (mixture @ np.sqrt(squared_error_bounds)) ** 2)),
            float(1 / np.linalg.norm(1 / target_norms)))
        rows.append(dict(epsilon=float(epsilon), graph_gram=gram.tolist(), unit_gram=unit.tolist(),
                         reachability=(norms / target_norms).tolist(),
                         exact_reachability_upper=np.minimum(
                             1, np.sqrt(norms ** 2 + squared_error_bounds) / target_norms).tolist(),
                         exact_raw_distance_bounds=[distance_lower, distance_upper],
                         exact_gram_diagonal_error_upper=squared_error_bounds.tolist(),
                         parameter_fraction=(np.diag(parameter) / np.diag(gram)).tolist(),
                         graph_gram_eigenvalues=np.linalg.eigvalsh(gram).tolist(),
                         residual_relative=errors, raw_hull=raw_hull, unit_hull=unit_hull,
                         exact_projection_tolerance_met=max(errors) <= 1e-5))
    return rows


def main(source, baseline, calibration, output):
    root = Path("/home/shangxueyuan/merging/tsv/scripts")
    sys.path[:0] = [str(root / "eight_task_coupling_20261116"),
                    str(root / "fullspace_conditional_20261116")]
    from prepare import sha, write
    from task_runtime import load_eight_geometry

    output.mkdir(exist_ok=False, parents=True)
    (output / "runtime").mkdir()
    shutil.copy2(source / "runtime/probes.json", output / "runtime/probes.json")
    protocol = json.loads((source / "protocol.json").read_text())
    grid = json.loads((calibration / "reference.json").read_text())
    lock = protocol["source"]
    base = json.loads((baseline / "provenance.json").read_text())
    g, provenance = load_eight_geometry(output / "runtime", lock["inventory_file"], lock["checkpoint"])
    for key in ("checkpoint_sha256", "probe_sha256", "inventory_sha256", "parameter_dimension"):
        if len({provenance[key], lock[key], base[key], grid["provenance"][key]}) != 1:
            raise ValueError(f"Mismatched source {key}")
    joint = EmpiricalJoint(g, protocol)
    shifts = [*grid["epsilon_scan"], grid["gradient_baseline"]]
    write(output / "protocol.json", dict(epsilons=shifts, tasks=joint.tasks, source=provenance,
          calibration_sha256=sha(calibration / "reference.json"), script_sha256=sha(__file__),
          stage="exploratory one-block Krylov step seeded by the full gradients",
          statement="Subspace Galerkin candidates; each shift has a measured full-space residual"))
    cases = {}
    for metric in joint.weights:
        charts, _, _, jvp, vjp = joint.scaled(metric)
        for objective in ("geo", "ce"):
            name = f"{metric}_{objective}"
            q, images, fisher_columns, rhs_norms, target_norms = [], [], [], [], []
            for idx, task in enumerate(joint.tasks):
                write(output / "status.json", dict(stage="evaluating_basis", task=task, case=name, finished=list(cases)))
                grad = torch.load(baseline / "gradients" / f"{name}_{task}.pt",
                                  weights_only=True, map_location="cpu", mmap=True).cuda()
                rhs_norms.append(float(grad.norm()))
                q.append(grad / rhs_norms[-1])
                target_norms.append(float((charts[idx] * joint.ideals[idx][objective].flatten()).norm()))
                images.append(jvp(q[-1]))
                fisher_columns.append(vjp(images[-1]))
            pg = np.array([[float(left @ right) for right in q] for left in q])
            z = torch.stack(images)
            fg = (z @ z.T).cpu().numpy()
            rhs = pg * np.array(rhs_norms)[None, :]

            def residual(epsilon, coefficients):
                errors = []
                for i, norm in enumerate(rhs_norms):
                    vector = -norm * q[i]
                    for j in range(len(q)):
                        vector.add_(q[j], alpha=float(epsilon * coefficients[j, i]))
                        vector.add_(fisher_columns[j], alpha=float(coefficients[j, i]))
                    errors.append(float(vector.norm()) / norm)
                return errors

            cases[name] = solve_grid(pg, fg, rhs, np.array(rhs_norms), np.array(target_norms), shifts, residual)
            write(output / "preview.json", cases)
            print(name, [(row["epsilon"], max(row["residual_relative"])) for row in cases[name]], flush=True)
    write(output / "status.json", dict(stage="complete", finished=list(cases),
          passed={case: all(row["exact_projection_tolerance_met"] for row in rows) for case, rows in cases.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("source", "baseline", "calibration", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    main(args.source, args.baseline, args.calibration, args.output)
