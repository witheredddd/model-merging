"""Audit matrix-free graph projections on the locked eight-task checkpoint."""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

from runtime_location import configure_runtime

import numpy as np
import torch

from graph_geometry import GraphSpace
from run import ideal


class EmpiricalJoint:
    def __init__(self, geometry, protocol):
        self.g = geometry
        self.tasks = list(geometry.tasks)
        self.lengths = [geometry.output_dimensions[t] for t in self.tasks]
        self.weights = {"original": [], "balanced": []}
        self.ideals = []
        self.class_counts = {}
        for task in self.tasks:
            logits, labels = geometry.c.baseline(task, geometry.probe, geometry.point)
            directions, _ = ideal(logits.cuda(), labels.cuda())
            self.ideals.append(directions)
            counts = torch.tensor(protocol["sample_summary"][task]["training_class_counts"], dtype=torch.float64)
            seen = torch.bincount(labels, minlength=len(counts))
            if len(counts) != logits.shape[-1] or not bool((seen > 0).all()):
                raise ValueError(f"Incomplete class coverage for {task}")
            self.class_counts[task] = seen.tolist()
            class_mass = {"original": counts / counts.sum(),
                          "balanced": torch.full_like(counts, 1 / len(counts))}
            for metric, mass in class_mass.items():
                self.weights[metric].append((mass[labels] / seen[labels]).cuda())

    def independent_forward(self, point):
        blocks = []
        for task in self.tasks:
            logits, _ = self.g.c.baseline(task, self.g.probe, point)
            blocks.append((2 * logits.softmax(-1).sqrt()).cuda())
        return blocks

    def direct_chart(self, blocks, metric):
        return torch.cat([(block * (weights[:, None] / 8).sqrt()).flatten()
                          for block, weights in zip(blocks, self.weights[metric])])

    def raw_jvp(self, direction):
        return [self.g.jvp(task, direction) for task in self.tasks]

    def scaled(self, metric):
        charts = [(w[:, None].expand_as(ideals["ce"]) / 8).sqrt().flatten()
                  for w, ideals in zip(self.weights[metric], self.ideals)]
        multipliers = [chart * math.sqrt(len(w)) for chart, w in zip(charts, self.weights[metric])]

        def embed_images(images):
            return torch.cat([scale * image for scale, image in zip(multipliers, images)])

        def jvp(v):
            return embed_images(self.raw_jvp(v))

        def vjp(z):
            result = torch.zeros_like(self.g.point)
            for task, scale, block in zip(self.tasks, multipliers, z.split(self.lengths)):
                result.add_(self.g.vjp(task, scale * block))
            return result

        return charts, multipliers, embed_images, jvp, vjp


def audit_derivative(joint, direction, generator):
    anchor = joint.g.point
    raw = joint.raw_jvp(direction)
    audit = {}
    steps = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    center = joint.independent_forward(anchor)
    pairs = [(joint.independent_forward(anchor + step * direction),
              joint.independent_forward(anchor - step * direction)) for step in steps]
    for metric in joint.weights:
        _, _, embed, _, vjp = joint.scaled(metric)
        image = embed(raw)
        z = torch.randn(image.shape, device=image.device, dtype=image.dtype, generator=generator)
        z /= z.norm()
        lhs, rhs = float(image @ z), float(direction @ vjp(z))
        adjoint_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
        curve = []
        for step, (plus_unweighted, minus_unweighted) in zip(steps, pairs):
            plus = joint.direct_chart(plus_unweighted, metric)
            minus = joint.direct_chart(minus_unweighted, metric)
            delta = (plus - minus) / (2 * step) - image
            blocks = [float(a.norm() / b.norm().clamp_min(1e-30))
                      for a, b in zip(delta.split(joint.lengths), image.split(joint.lengths))]
            curve.append(dict(step=step, joint=float(delta.norm() / image.norm()), per_task=blocks))
        # Also independently confirm the normalization directly in both charts.
        norm = float(joint.direct_chart(center, metric).norm())
        audit[metric] = derivative_acceptance(curve, norm, adjoint_error)
        audit[metric]["adjoint_pair"] = [lhs, rhs]
    return audit


def derivative_acceptance(curve, norm, adjoint_error, tolerance=1e-5):
    finite = [row for row in curve if np.isfinite(row["joint"]) and np.isfinite(row["per_task"]).all()]
    common = min(finite, key=lambda row: row["joint"], default=None)
    passed = (common is not None and adjoint_error < 1e-9 and abs(norm - 2) < 1e-10
              and common["joint"] < tolerance and max(common["per_task"]) < tolerance)
    return dict(adjoint_relative_error=adjoint_error, prediction_norm=norm,
                finite_difference=curve, best_common_step=common, tolerance=tolerance,
                passed=bool(passed))


def solve(tangent, rhs, target, rtol, max_steps, iteration_status):
    """CG with explicit residual replacement and a final independent graph audit."""
    v = torch.zeros_like(rhs)
    r = rhs.clone()
    direction = r.clone()
    rr = float(r @ r)
    rhs_norm = math.sqrt(rr)
    trace = []
    for iteration in range(1, max_steps + 1):
        if rr == 0:
            break
        gd = tangent.pullback(direction)
        denominator = float(direction @ gd)
        if denominator <= 0 or not math.isfinite(denominator):
            raise RuntimeError(f"CG curvature invalid: {denominator}")
        alpha = rr / denominator
        v.add_(direction, alpha=alpha)
        r.add_(gd, alpha=-alpha)
        next_rr = float(r @ r)
        relative = math.sqrt(next_rr) / rhs_norm
        trace.append(relative)
        iteration_status(iteration, relative)
        if relative <= rtol / 2:
            r = rhs - tangent.pullback(v)
            next_rr = float(r @ r)
            if math.sqrt(next_rr) / rhs_norm <= rtol:
                break
            direction.copy_(r)
        else:
            direction.mul_(next_rr / rr).add_(r)
        rr = next_rr
    parameter, image = tangent.lift(v)
    normal = tangent.vjp(image)
    g_residual = normal + tangent.epsilon * v - rhs
    # Evaluate the residual adjoint directly, rather than reusing the solve residual.
    orth = tangent.adjoint(-parameter, target - image)
    graph_energy = float(parameter @ parameter + image @ image)
    b_energy = float(target @ target)
    residual = target - image
    residual_energy = float(parameter @ parameter + residual @ residual)
    relative_solve = float(g_residual.norm()) / max(rhs_norm, 1e-300)
    relative_orth = float(orth.norm()) / max(rhs_norm, 1e-300)
    pythagoras = abs(b_energy - graph_energy - residual_energy) / max(b_energy, 1e-300)
    parameter_energy = float(v @ v)
    prediction_energy = float(image @ image)
    report = dict(iterations=len(trace), recursive_residuals=trace, solve_residual=relative_solve,
                  orthogonality_residual=relative_orth, pythagorean_relative_error=pythagoras,
                  reachability=math.sqrt(graph_energy / b_energy) if b_energy else None,
                  rayleigh_raw=prediction_energy / parameter_energy if parameter_energy else None,
                  parameter_energy_fraction=float(parameter @ parameter) / graph_energy if graph_energy else None,
                  passed=relative_solve <= rtol and relative_orth <= rtol and pythagoras <= 5 * rtol)
    return v, image, report


def main(args):
    configure_runtime()
    from prepare import sha, write
    from task_runtime import load_eight_geometry

    begin = time.monotonic()
    protocol = json.loads((args.source / "protocol.json").read_text())
    lock = protocol["source"]
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "runtime").mkdir()
    shutil.copy2(args.source / "runtime/probes.json", out / "runtime/probes.json")
    g, source = load_eight_geometry(out / "runtime", lock["inventory_file"], lock["checkpoint"], expected=lock)
    for key in ("checkpoint_sha256", "probe_sha256", "inventory_sha256", "parameter_dimension"):
        if source[key] != lock[key]:
            raise ValueError(f"Provenance differs for {key}")
    joint = EmpiricalJoint(g, protocol)
    source.update(gpu=torch.cuda.get_device_name(), audit_code_sha256=sha(__file__),
                  graph_code_sha256=sha(Path(__file__).with_name("graph_geometry.py")),
                  source_protocol_sha256=sha(args.source / "protocol.json"),
                  rtol=args.rtol, max_steps=args.max_steps)
    write(out / "provenance.json", source)

    def status(stage, **details):
        write(out / "status.json", dict(stage=stage, seconds=time.monotonic() - begin, **details))

    generator = torch.Generator(device=g.point.device).manual_seed(221127)
    status("derivative")
    _, multipliers, _, _, _ = joint.scaled("original")
    # A fixed CE gradient plus an independent random component exercises both regimes.
    grad = g.vjp(joint.tasks[0], (joint.ideals[0]["ce"].flatten() * multipliers[0].square()
                                / math.sqrt(len(joint.weights["original"][0]))))
    direction = torch.randn(g.point.shape, device=g.point.device, dtype=g.point.dtype, generator=generator)
    direction /= direction.norm()
    direction.add_(grad / grad.norm())
    direction /= direction.norm()
    del grad
    derivatives = audit_derivative(joint, direction, generator)
    write(out / "derivative.json", derivatives)
    if not all(x["passed"] for x in derivatives.values()):
        raise RuntimeError("Independent prediction derivative audit failed")

    scaled = {metric: joint.scaled(metric) for metric in joint.weights}
    point_rms_squared = float(g.point @ g.point) / g.dimension
    scales = {metric: {"expert": {}, "ce_gradient": {}, "geo_gradient": {}} for metric in scaled}
    for index, task in enumerate(joint.tasks):
        status("calibrating_experts", task=task)
        expert = (g.c.experts[index] - g.c.base).to(g.point.device)
        expert /= expert.norm()
        images = joint.raw_jvp(expert)
        for metric, (_, _, embed, _, _) in scaled.items():
            image = embed(images)
            scales[metric]["expert"][task] = float(image @ image)
        del expert, images

    gradient_dirs = out / "gradients"
    gradient_dirs.mkdir()
    for metric, (charts, mul, embed, _, _) in scaled.items():
        for objective in ("ce", "geo"):
            for i, task in enumerate(joint.tasks):
                status("calibrating_gradients", metric=metric, objective=objective, task=task)
                block = charts[i] * joint.ideals[i][objective].flatten()
                h = g.vjp(task, mul[i] * block)
                norm = float(h.norm())
                if norm <= 0 or not math.isfinite(norm):
                    raise RuntimeError(f"Invalid task gradient for {task}")
                image = embed(joint.raw_jvp(h / norm))
                scales[metric][f"{objective}_gradient"][task] = float(image @ image)
                torch.save(h.cpu(), gradient_dirs / f"{metric}_{objective}_{task}.pt")
                del h, image
    # Shared audit epsilon: median task-gradient Rayleigh scale across both
    # measures and objectives. Uniform RMS normalization is a units change.
    gradient_rayleigh = [r for metric in scales.values() for key, group in metric.items()
                         if key.endswith("_gradient") for r in group.values()]
    epsilon = float(np.median(gradient_rayleigh))
    if epsilon <= 0 or not math.isfinite(epsilon):
        raise RuntimeError("Nonpositive empirical damping scale")
    write(out / "scale.json", dict(rayleigh_raw=scales, epsilon_raw=epsilon,
          epsilon_in_anchor_rms_units=epsilon * point_rms_squared,
          anchor_rms=math.sqrt(point_rms_squared),
          parameter_coordinate="u = v / RMS(anchor); raw graph epsilon = epsilon_rms / RMS(anchor)^2",
          selection_rule="Pooled median of 32 positive full task-gradient Rayleigh quotients; one shared epsilon",
          report_only_fixture_factors=[0.1, 1, 10]))

    cases = {}
    for metric, (charts, _, _, jvp, vjp) in scaled.items():
        tangent = GraphSpace(epsilon, g.point).at(jvp, vjp)
        for objective in ("geo", "ce"):
            name = f"{metric}_{objective}"
            case_dir = out / name
            case_dir.mkdir()
            rows, images, solutions, gradients = [], [], [], []
            for i, task in enumerate(joint.tasks):
                status("solving", case=name, task=task, completed_in_case=i)
                target = torch.zeros(sum(joint.lengths), device=g.point.device, dtype=g.point.dtype)
                offset = sum(joint.lengths[:i])
                target[offset:offset + joint.lengths[i]] = charts[i] * joint.ideals[i][objective].flatten()
                rhs = torch.load(gradient_dirs / f"{name}_{task}.pt", weights_only=True, map_location="cpu").cuda()

                def iteration_status(iteration, residual):
                    status("solving", case=name, task=task, completed_in_case=i,
                           iteration=iteration, recursive_residual=residual)

                v, image, row = solve(tangent, rhs, target, args.rtol, args.max_steps, iteration_status)
                row["task"] = task
                row["rayleigh_in_anchor_rms_units"] = (row["rayleigh_raw"] * point_rms_squared
                    if row["rayleigh_raw"] is not None else None)
                rows.append(row)
                images.append(image)
                solutions.append(v)
                gradients.append(rhs)
                tmp = case_dir / f"{task}_direction.tmp"
                torch.save(v.cpu(), tmp)
                tmp.replace(case_dir / f"{task}_direction.pt")
                write(case_dir / "tasks.json", rows)
                print(name, task, row["iterations"], row["solve_residual"], row["passed"], flush=True)
            image_matrix = torch.stack(images)
            gram_graph = (image_matrix @ image_matrix.T).cpu().numpy()
            gram_rhs = np.empty((8, 8))
            for i, v in enumerate(solutions):
                for j, h in enumerate(gradients):
                    gram_rhs[j, i] = float(h @ v)
                    gram_graph[j, i] += epsilon * float(solutions[j] @ v)
            eigenvalues = np.linalg.eigvalsh((gram_graph + gram_graph.T) / 2)
            gram_scale = max(np.linalg.norm(gram_graph), 1e-300)
            symmetry_error = np.linalg.norm(gram_graph - gram_graph.T) / gram_scale
            rhs_symmetry_error = np.linalg.norm(gram_rhs - gram_rhs.T) / gram_scale
            route_error = np.linalg.norm(gram_graph - gram_rhs) / gram_scale
            # Save the response and the much smaller sufficient Gram statistic.
            np.savez_compressed(case_dir / "prediction_responses.npz", responses=image_matrix.cpu().numpy())
            report = dict(tasks=joint.tasks, rows=rows, graph_gram=gram_graph.tolist(), rhs_gram=gram_rhs.tolist(),
                          symmetry_relative_error=float(symmetry_error), rhs_symmetry_relative_error=float(rhs_symmetry_error),
                          gram_route_relative_error=float(route_error), gram_eigenvalues=eigenvalues.tolist(),
                          gram_passed=bool(symmetry_error < 1e-12 and eigenvalues.min() >= -1e-12 * gram_scale
                                           and route_error <= 5 * args.rtol and rhs_symmetry_error <= 5 * args.rtol))
            report["passed"] = report["gram_passed"] and all(row["passed"] for row in rows)
            write(case_dir / "gram.json", report)
            cases[name] = report
            write(out / "results.json", cases)
            del solutions, gradients, image_matrix, images, v, rhs
    status("complete", case_passed={name: case["passed"] for name, case in cases.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    if args.rtol <= 0 or args.max_steps < 1:
        parser.error("positive rtol and iteration count required")
    main(args)
