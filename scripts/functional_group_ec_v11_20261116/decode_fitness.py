"""Deterministic semantic-mode decoder and original Pareto fitness."""
import hashlib
import importlib.util
import numpy as np
from chart_state import coordinates


class RealEvaluator:
    def __init__(self, runtime, chart):
        self.runtime, self.chart = runtime, chart
        spec = importlib.util.spec_from_file_location(
            'original_compromise_numerics',
            '/home/shangxueyuan/merging/tsv/scripts/compromise_geometry_20261115/numerics.py')
        numerics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(numerics)
        self.original_fitness = numerics.fitness
        self.reference = runtime.old.all_metrics(runtime.c, runtime.point(chart.anchor_a), 0)

    def __call__(self, z):
        r, chart = self.runtime, self.chart
        a = coordinates(chart, z)
        point = r.point(a)
        metrics = r.old.all_metrics(r.c, point, 0)
        top1 = np.array([metrics[t]['Top1'] for t in r.tasks])
        F = self.original_fitness(top1, r.normalizers)
        # Hash actual FP32 reconstructed model, not only latent coordinates.
        point_cpu = point.detach().cpu().contiguous().numpy()
        model_hash = hashlib.sha256(memoryview(point_cpu).cast('B')).hexdigest()
        record = dict(z=np.array(z, copy=True), a=a, parameter_digest=model_hash,
                      top1=top1, normalized_top1=top1 / r.normalizers,
                      ec_objectives=F,
                      monitor={t: dict(CE=metrics[t]['CE'], Top1=metrics[t]['Top1'],
                          KL=r.old.prediction_kl(self.reference[t]['phi'], metrics[t]['phi']),
                          finite_response_norm=float((metrics[t]['phi'] -
                                                      self.reference[t]['phi']).norm()))
                               for t in r.tasks},
                      cost=r.cost.snapshot(r.policy))
        return F, record

