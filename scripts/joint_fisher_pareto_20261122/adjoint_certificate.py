"""Certify strict simultaneous descent through the exact adjoint identity."""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from run import ideal, polytope


def main(source, output):
    root = Path("/home/shangxueyuan/merging/tsv/scripts")
    sys.path[:0] = [str(root / "eight_task_coupling_20261116"),
                    str(root / "fullspace_conditional_20261116")]
    from prepare import write, sha
    from task_runtime import load_eight_geometry

    start = time.monotonic()
    protocol = json.loads((source / "protocol.json").read_text())
    provenance = protocol["source"]
    if sha(source / "runtime/probes.json") != provenance["probe_sha256"]:
        raise ValueError("Probe lock changed")
    output.mkdir(exist_ok=False, parents=True)
    (output / "runtime").mkdir()
    shutil.copy2(source / "runtime/probes.json", output / "runtime/probes.json")
    g, point = load_eight_geometry(output / "runtime", provenance["inventory_file"], provenance["checkpoint"])
    for key in ("checkpoint_sha256", "probe_sha256", "inventory_sha256", "parameter_dimension"):
        if point[key] != provenance[key]:
            raise ValueError(f"Incompatible {key}")
    tasks = list(g.tasks)
    targets, factors = {}, {"original": {}, "balanced": {}}
    for task in tasks:
        z, y = g.c.baseline(task, 0, g.point)
        targets[task], _ = ideal(z.cuda(), y.cuda())
        current = torch.bincount(y, minlength=z.shape[1]).double()
        counts = torch.tensor(protocol["sample_summary"][task]["training_class_counts"], dtype=torch.double)
        if len(counts) != len(current) or not bool((current > 0).all()):
            raise ValueError(f"Invalid class counts for {task}")
        factors["original"][task] = (len(y) / 8 * counts[y] / counts.sum() / current[y]).sqrt().cuda()[:, None]
        factors["balanced"][task] = (len(y) / (8 * len(counts)) / current[y]).sqrt().cuda()[:, None]

    cases = {}
    for metric, scale in factors.items():
        for objective in ("geo", "ce"):
            name = f"{metric}_{objective}"
            targets_whitened, gradients = [], []
            gram = np.zeros((8, 8))
            for i, task in enumerate(tasks):
                write(output / "status.json", dict(stage="adjoint", case=name, task=task, completed=list(cases)))
                b = targets[task][objective]
                target = b * scale[task] / math.sqrt(len(b))
                targets_whitened.append(target)
                gradient = g.vjp(task, (target * scale[task]).flatten())
                gradients.append(gradient)
                for j, previous in enumerate(gradients):
                    gram[i, j] = gram[j, i] = float(gradient @ previous)
            norms = np.sqrt(np.diag(gram))
            normalized_gram = gram / np.outer(norms, norms) if np.all(norms > 0) else gram
            qp = polytope(normalized_gram)
            coeff = np.asarray(qp["weights"])
            if np.all(norms > 0):
                coeff /= norms
            direction = torch.zeros_like(g.point)
            for weight, gradient in zip(coeff, gradients):
                direction.add_(gradient, alpha=float(weight))
            del gradients, gradient, previous
            slopes = gram @ coeff
            actual_slopes = []
            response_energy = 0.
            for task, target in zip(tasks, targets_whitened):
                image = g.jvp(task, direction).reshape_as(target) * scale[task]
                response_energy += float(image.flatten() @ image.flatten())
                actual_slopes.append(float((image * target).sum()))
            adjoint_error = float(np.max(np.abs(slopes - actual_slopes))
                                  / max(np.max(np.abs(slopes)), 1e-300))
            response_norm = math.sqrt(response_energy)
            margins = np.asarray(actual_slopes) / response_norm if response_norm else np.zeros(8)
            targets_norm = np.array([float(target.norm()) for target in targets_whitened])
            lower = max(0., float(margins.min()))
            upper = float(np.sum(targets_norm ** -2) ** -0.5) if np.all(targets_norm > 0) else 0.
            cases[name] = dict(tasks=tasks, parameter_gradient_gram=gram.tolist(),
                               normalized_gram_eigenvalues=np.linalg.eigvalsh(normalized_gram).tolist(),
                               adjoint_max_relative_error=adjoint_error, weights=coeff.tolist(),
                               normalized_common_direction_margins=margins.tolist(),
                               reaches_all_eight=bool(lower > 1e-10 * max(targets_norm) and adjoint_error < 1e-8),
                               distance_lower_bound=lower, distance_upper_bound=upper,
                               seconds=time.monotonic() - start)
            write(output / "certificates.json", cases)
            print(name, cases[name]["reaches_all_eight"], lower, upper, flush=True)
    write(output / "status.json", dict(stage="complete", cases=list(cases), seconds=time.monotonic() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.source, args.output)
