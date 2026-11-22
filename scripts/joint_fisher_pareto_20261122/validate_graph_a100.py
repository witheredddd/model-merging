"""Check the graph embedding and tangent operators on the actual eight tasks."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

from graph_geometry import GraphSpace


def main(source, out):
    root = Path("/home/shangxueyuan/merging/tsv/scripts")
    sys.path[:0] = [str(root / "eight_task_coupling_20261116"),
                    str(root / "fullspace_conditional_20261116")]
    from prepare import sha, write
    from task_runtime import load_eight_geometry

    protocol = json.loads((source / "protocol.json").read_text())
    lock = protocol["source"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "runtime").mkdir()
    shutil.copy2(source / "runtime/probes.json", out / "runtime/probes.json")
    g, provenance = load_eight_geometry(out / "runtime", lock["inventory_file"], lock["checkpoint"])
    for key in ("checkpoint_sha256", "probe_sha256", "inventory_sha256", "parameter_dimension"):
        if provenance[key] != lock[key]:
            raise ValueError(f"Provenance differs for {key}")

    generator = torch.Generator(device=g.point.device).manual_seed(221126)
    v = torch.randn(g.point.shape, device=g.point.device, dtype=g.point.dtype, generator=generator)
    v /= v.norm()
    outputs, images = {}, {}
    scales = {"original": {}, "balanced": {}}
    for task in g.tasks:
        z, labels = g.c.baseline(task, g.probe, g.point)
        root_prob = (z.log_softmax(-1) / 2).exp().cuda()
        outputs[task] = root_prob
        images[task] = g.jvp(task, v).reshape_as(root_prob)
        counts = torch.as_tensor(protocol["sample_summary"][task]["training_class_counts"],
                                 dtype=torch.float64)
        seen = torch.bincount(labels, minlength=len(counts)).double()
        if not torch.all(seen > 0):
            raise ValueError(f"Missing class in {task}")
        scales["original"][task] = (counts[labels] / counts.sum() / seen[labels]).sqrt().cuda()[:, None]
        scales["balanced"][task] = (1 / (len(counts) * seen[labels])).sqrt().cuda()[:, None]

    # Positive fixture to exercise the implementation. The geometry has no default epsilon.
    fixture_epsilon = 1.
    space = GraphSpace(fixture_epsilon, g.point)
    results = {}
    for metric, factors in scales.items():
        lengths = [outputs[t].numel() for t in g.tasks]
        task_scale = {t: factors[t].expand_as(outputs[t]).flatten() for t in g.tasks}
        jacobian_scale = {t: task_scale[t] * (len(outputs[t]) / 8) ** .5 for t in g.tasks}

        def jvp(direction):
            return torch.cat([jacobian_scale[t] * g.jvp(t, direction) for t in g.tasks])

        def vjp(vector):
            result = torch.zeros_like(g.point)
            for task, block in zip(g.tasks, vector.split(lengths)):
                result.add_(g.vjp(task, jacobian_scale[task] * block))
            return result

        prediction = torch.cat([(2 / 8 ** .5) * task_scale[t] * outputs[t].flatten() for t in g.tasks])
        image = torch.cat([jacobian_scale[t] * images[t].flatten() for t in g.tasks])
        tangent = space.at(jvp, vjp)
        parameter_point, prediction_point = space.point(g.point, prediction)
        regular = tangent.adjoint(v, image)
        pullback_energy = float(v @ regular)
        coefficients = [float(image @ image), float(v @ v)]
        energy_error = abs(pullback_energy - sum(coefficients)) / sum(coefficients)
        result = dict(prediction_norm=float(prediction_point.norm()),
                      reference_parameter_component_norm=float(parameter_point.norm()),
                      task_probability_mass={t: float((factors[t] ** 2).sum()) for t in g.tasks},
                      tangent_radial_error=max(float((outputs[t] * images[t]).sum(1).abs().max())
                                               for t in g.tasks),
                      tangent_norm_squared_coefficients={"constant": coefficients[0],
                                                        "linear_in_epsilon": coefficients[1]},
                      graph_adjoint_energy_relative_error=energy_error)
        if (abs(result["prediction_norm"] - 2) > 1e-10
                or result["tangent_radial_error"] > 1e-10
                or energy_error > 1e-10
                or any(abs(s - 1.) > 1e-10 for s in result["task_probability_mass"].values())):
            raise RuntimeError(f"{metric} graph audit failed: {result}")
        results[metric] = result
    write(out / "graph_audit.json", dict(passed=True, gpu=torch.cuda.get_device_name(),
          source=provenance, graph_reference="anchor0 step004", epsilon_fixture=fixture_epsilon,
          research_epsilon=None, graph_source_sha256=sha(Path(__file__).with_name("graph_geometry.py")),
          audit_source_sha256=sha(__file__), results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.source, args.output)
