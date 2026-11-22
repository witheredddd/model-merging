"""Matrix-free empirical joint Fisher–Rao projection at one model checkpoint."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize


def ideal(logits, labels):
    half_log_p = logits.log_softmax(-1) / 2
    root = half_log_p.exp()
    index = labels[:, None]
    correct = root.gather(1, index)
    one = torch.zeros_like(root).scatter_(1, index, 1)
    sine = torch.linalg.vector_norm(root * (1 - one), dim=1, keepdim=True)
    angle = torch.atan2(sine, correct)
    # Write the true-class component as sin²(angle) to avoid cancellation.
    tangent = (-correct * root).scatter_(1, index, sine.square())
    scale = torch.where(sine > 0, 2 * angle / sine, 2)
    ce = (-root).scatter_(1, index, sine.square() / correct)
    return {"geo": scale * tangent, "ce": ce}, root


def polytope(gram):
    n = len(gram)
    if n == 0:
        return None
    scale = max(float(np.max(np.diag(gram))), 1e-300)
    q = gram / scale
    fit = minimize(lambda a: 0.5 * a @ q @ a, np.full(n, 1 / n),
                   jac=lambda a: q @ a, method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1,
                                 "jac": lambda a: np.ones_like(a)}],
                   options={"ftol": 1e-14, "maxiter": 500})
    a = np.maximum(fit.x, 0)
    a /= a.sum()
    scores = gram @ a
    energy = float(a @ scores)
    centered = np.eye(n) - np.ones((n, n)) / n
    eigenvalues = np.linalg.eigvalsh(centered @ gram @ centered)
    return dict(distance=math.sqrt(max(energy, 0)), weights=a.tolist(),
                aff_dim_numerical=int(np.sum(eigenvalues > 1e-10 * scale)),
                active_coefficients=np.flatnonzero(a > 1e-8).tolist(),
                dual_gap=max(0., energy - float(scores.min())), success=bool(fit.success))


def project(jvp, vjp, target, steps, rtol):
    """Zero-start CGLS on A, with no damping and no parameter support filter."""
    x = torch.zeros_like(target)
    normal = vjp(target)
    direction = normal.clone()
    energy = float(normal @ normal)
    initial = math.sqrt(energy)
    history = []
    for _ in range(steps):
        if energy == 0:
            break
        image = jvp(direction)
        denom = float(image @ image)
        if denom <= 0:
            break
        x.add_(image, alpha=energy / denom)
        normal = vjp(target - x)
        next_energy = float(normal @ normal)
        history.append(math.sqrt(next_energy) / initial)
        if history[-1] <= rtol:
            break
        direction.mul_(next_energy / energy).add_(normal)
        energy = next_energy
    relative = history[-1] if history else (0. if initial == 0 else 1.)
    return x, dict(iterations=len(history), relative_normal_residual=relative,
                   meets_normal_tolerance=relative <= rtol, residual_trace=history,
                   relative_data_residual=float((target - x).norm() / target.norm())
                   if float(target.norm()) else 0.)


def summarize(vertices, targets, lengths, fits, tau_b, tau_eta, rtol):
    norms = np.linalg.norm(vertices, axis=1)
    ideal_norms = np.linalg.norm(targets, axis=1)
    eta = np.divide(norms, ideal_norms, out=np.zeros_like(norms), where=ideal_norms > 0)
    active = np.flatnonzero((ideal_norms * np.sqrt(8) > tau_b) & (eta > tau_eta) & (norms > 0))
    gram = vertices @ vertices.T
    raw = polytope(gram)
    unit = vertices[active] / norms[active, None]
    candidate = np.asarray(raw["weights"]) @ vertices
    candidate_norm = np.linalg.norm(candidate)
    margins = targets @ candidate
    lower_bound = float(margins.min() / candidate_norm) if candidate_norm else 0.
    blocks = np.split(vertices, np.cumsum(lengths)[:-1], axis=1)
    # Each computed vertex is in range(A). Its actual first-order effects use B.
    return dict(raw=raw, direction=polytope(unit @ unit.T), direction_tasks=active.tolist(),
                normal_tolerance=rtol, all_meet_normal_tolerance=all(f["meets_normal_tolerance"] for f in fits),
                ideal_norm=ideal_norms.tolist(), vertex_norm=norms.tolist(),
                reachability_candidate=eta.tolist(), gram=gram.tolist(), fits=fits,
                task_response_norms=np.stack([np.linalg.norm(b, axis=1) for b in blocks], axis=1).tolist(),
                ideal_dot_vertex=(targets @ vertices.T).tolist(),
                common_descent_candidate_margins=margins.tolist(),
                strict_common_descent=bool(lower_bound > 1e-12 * max(ideal_norms)),
                exact_distance_lower_bound_from_candidate=max(0., lower_bound))


def main(args):
    root = Path("/home/shangxueyuan/merging/tsv/scripts")
    sys.path[:0] = [str(root / "eight_task_coupling_20261116"),
                    str(root / "fullspace_conditional_20261116")]
    from prepare import write, sha
    from task_runtime import load_eight_geometry

    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    inventory = json.loads(args.inventory.read_text())["tasks"]
    prior = json.loads(args.probes.read_text())
    for row in inventory:
        locked = prior["tasks"][row["task"]]
        # Existing probes were emitted by successive complete class rounds.
        locked["ids"] = [
            locked["ids"][0][:len(row["classes"]) * args.per_class], []
        ]
    write(out / "runtime/probes.json", prior)
    write(out / "status.json", dict(stage="loading"))
    g, provenance = load_eight_geometry(out / "runtime", args.inventory, args.checkpoint)
    outputs = {}
    weights = {"original": {}, "balanced": {}}
    summary = {}
    for row in inventory:
        task = row["task"]
        z, labels = g.c.baseline(task, g.probe, g.point)
        outputs[task], square_root = ideal(z.cuda(), labels.cuda())
        current = torch.bincount(labels, minlength=z.shape[1]).double()
        counts = torch.zeros_like(current)
        counts[row["classes"]] = torch.tensor(row["class_counts"], dtype=torch.float64)
        if not torch.all(current[row["classes"]] > 0) or not torch.all(
                current[row["classes"]] == args.per_class):
            raise ValueError(f"{task} probe is not {args.per_class} examples per class")
        n = len(labels)
        weights["original"][task] = (n / 8 * counts[labels] / counts.sum() /
                                      current[labels]).sqrt().cuda()[:, None]
        weights["balanced"][task] = (n / (8 * len(row["classes"])) /
                                      current[labels]).sqrt().cuda()[:, None]
        summary[task] = dict(samples=n, classes=z.shape[1],
                             training_class_counts=row["class_counts"],
                             max_sphere_error=float((2 * square_root.norm(dim=1) - 2).abs().max()),
                             ideal_tangency_max={k: float((square_root * v).sum(1).abs().max())
                                                  for k, v in outputs[task].items()})
    tasks = list(g.tasks)
    lengths = [g.output_dimensions[task] for task in tasks]
    protocol = dict(source=provenance, tasks=tasks, sample_summary=summary,
                    original_probe_sha256=sha(args.probes), output_lengths=lengths,
                    runtime_sources={str(root / directory / name): sha(root / directory / name)
                                     for directory, name in [
                                         ("eight_task_coupling_20261116", "task_runtime.py"),
                                         ("fullspace_conditional_20261116", "geometry.py"),
                                         ("fullspace_conditional_20261116", "precision.py")]},
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    per_class=args.per_class, source_split="existing locked training-side development probe",
                    task_weight="1/8", original_weight="training class frequencies on stratified probe",
                    balanced_weight="equal class frequencies on the same inputs",
                    method="zero-start undamped matrix-free CGLS, FP64",
                    max_iterations=args.steps, normal_tolerance=args.rtol,
                    tau_b=args.tau_b, tau_eta=args.tau_eta)
    write(out / "protocol.json", protocol)
    results = {}
    for metric, factors in weights.items():
        embedding_scale = {
            task: factors[task].expand_as(outputs[task]["geo"]).ravel() for task in tasks
        }

        def jvp(v):
            return torch.cat([embedding_scale[t] * g.jvp(t, v) for t in tasks])

        def vjp(u):
            result = None
            for t, block in zip(tasks, u.split(lengths)):
                if not bool(block.any()):
                    continue
                contribution = g.vjp(t, embedding_scale[t] * block)
                if result is None:
                    result = contribution
                else:
                    result.add_(contribution)
            return torch.zeros_like(g.point) if result is None else result

        for objective in ("geo", "ce"):
            name = f"{metric}_{objective}"
            targets = torch.zeros((8, sum(lengths)), device=g.point.device, dtype=g.point.dtype)
            start = 0
            for i, task in enumerate(tasks):
                raw = outputs[task][objective]
                targets[i, start:start + lengths[i]] = (
                    raw * factors[task] / math.sqrt(raw.shape[0])).flatten()
                start += lengths[i]
            vertices, fits = [], []
            for i, task in enumerate(tasks):
                write(out / "status.json", dict(stage="projecting", case=name, task=task,
                      finished_cases=list(results), done_in_case=i, seconds=time.monotonic() - begin))
                x, fit = project(jvp, vjp, targets[i], args.steps, args.rtol)
                vertices.append(x.cpu().numpy())
                fits.append(dict(task=task, **fit))
                print(name, task, fit, flush=True)
            vertices = np.stack(vertices)
            target_np = targets.cpu().numpy()
            results[name] = summarize(vertices, target_np, lengths, fits, args.tau_b, args.tau_eta, args.rtol)
            np.savez_compressed(out / f"{name}_vertices.npz", vertices=vertices, targets=target_np)
            write(out / "results.json", results)
    write(out / "status.json", dict(stage="complete", cases=list(results),
          seconds=time.monotonic() - begin, peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("output", "checkpoint", "inventory", "probes"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--tau-b", type=float, default=1e-8)
    parser.add_argument("--tau-eta", type=float, default=1e-6)
    options = parser.parse_args()
    if options.steps < 1 or options.per_class < 1 or options.rtol <= 0:
        parser.error("positive steps, samples per class and relative tolerance required")
    main(options)
