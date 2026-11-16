"""Software correctness checks, never scientific acceptance thresholds."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from chart_state import FrozenChart, coordinates, freeze
from monitor import Monitor


def chart(m=3):
    # Test sizes exercise dynamic interfaces; they do not set an experiment dimension.
    return FrozenChart('test', 'source', np.arange(6.) / 10,
                       np.arange(6*m, dtype=float).reshape(6, m) / 100,
                       tuple({} for _ in range(m)), np.arange(6.) / 10,
                       np.zeros(6, dtype=int), None, 'test')


class TestImplementation(unittest.TestCase):
    def test_overlapping_mode_decode_order_independent(self):
        c = chart()
        z, w = np.array([.2, -.5, .1]), np.array([-.1, .4, .8])
        before = c.anchor_a.copy()
        expected = before + sum((c.mode_matrix[:, j] * z[j] for j in range(3)))
        np.testing.assert_allclose(coordinates(c, z), expected)
        coordinates(c, w)
        np.testing.assert_array_equal(coordinates(c, z), coordinates(c, z))
        np.testing.assert_array_equal(c.anchor_a, before)

    def test_snapshot_isolated_from_recondition(self):
        from types import SimpleNamespace
        m = SimpleNamespace(vector=np.array([.6, .8]), driver=np.array([1., 0.]),
                            supports=['b'], map_matrix=[[-.3]])
        policy = SimpleNamespace(modes=[m])
        snap = freeze('x', 'hash', np.zeros(2), policy,
                      SimpleNamespace(column_ages=np.zeros(2)), 'protocol')
        m.vector[:] = 0
        m.map_matrix[0][0] = 9
        np.testing.assert_array_equal(snap.mode_matrix[:, 0], [.6, .8])
        self.assertEqual(snap.modes_snapshot[0]['map_matrix'], [[-.3]])

    def test_no_fixed_dimension_or_extra_fitness_filter(self):
        from ec_adapter import GroupProblem
        for m in [1, 3, 6]:
            c = chart(m)
            def evaluate(z):
                F = np.array([np.sum(z), -np.sum(z)])
                return F, {'ec_objectives': F, 'CE': 1e10}
            p = GroupProblem(c, -.1, .1, evaluate, 2)
            self.assertEqual(p.n_var, m)
            self.assertEqual(p.n_ieq_constr, 0)
            self.assertEqual(p.n_eq_constr, 0)
            z = np.zeros(m)
            np.testing.assert_array_equal(p.evaluate(z), [0., 0.])

    def test_reject_undefined_domain_not_invent_bounds(self):
        import json
        from protocol import load_protocol
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'protocol.json'
            p.write_text(json.dumps({'protocol_id': 'unspecified'}))
            with self.assertRaises(ValueError):
                load_protocol(p)

    def test_native_nsga_checkpoint_resume_and_monitor_invariance(self):
        from ec_adapter import run_ec
        c = chart()
        def evaluate(z):
            F = np.array([np.sum((z-.2)**2), np.sum((z+.3)**2)])
            return F, dict(z=np.array(z), a=coordinates(c, z), ec_objectives=F)
        p = dict(n_objectives=2, domain=dict(lower=-1., upper=1.),
                 ec=dict(pop_size=7, seed=1, termination=dict(kind='n_eval', value=21)))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            full = run_ec(c, evaluate, p, root/'full')
            part = run_ec(c, evaluate, p, root/'resumed',
                          Monitor(root/'monitor.jsonl'), stop_after_evaluations=9)
            self.assertFalse(part['complete'])
            resumed = run_ec(c, evaluate, p, root/'resumed', resume=True,
                             monitor=Monitor(root/'monitor.jsonl'))
            for a, b in zip(full['records'], resumed['records']):
                np.testing.assert_array_equal(a['z'], b['z'])
                np.testing.assert_array_equal(a['ec_objectives'], b['ec_objectives'])
            self.assertEqual(len(full['records']), len(resumed['records']))
            self.assertEqual(len(resumed['records']), 21)

    def test_joint_motion_domain_and_original_rank_rule(self):
        from control_domain import MotionDomain
        from legacy_adapter import imports
        core, _, _, _ = imports()
        B = np.array([[1., .8], [0., .6], [0., 0.]])
        metric = np.diag([1., 1.5, 1.])
        domain = MotionDomain(B, metric, .03, core.ortho)
        rng = np.random.default_rng(9)
        for genes in rng.uniform(domain.lower, domain.upper, size=(100, 2)):
            z, meta = domain.map(genes)
            self.assertLessEqual(np.linalg.norm(B@z), .03*(1+1e-14))
            self.assertLessEqual(z@domain.H@z, .03**2*(1+1e-14))
            np.testing.assert_allclose(z, genes/meta['radial_scale'])
        redundant = np.column_stack([B, B[:, 0]])
        reduced = MotionDomain(redundant, metric, .03, core.ortho)
        self.assertEqual(reduced.effective_rank, core.ortho(redundant).shape[1])
        self.assertEqual(reduced.gene_to_mode.shape, (3, 2))
        z, _ = reduced.map(np.zeros(2))
        np.testing.assert_array_equal(z, np.zeros(3))

    def test_pareto_arc_matches_original_cache_rule(self):
        from protocol import select_branch
        rows = [dict(candidate_id=str(i), ec_objectives=np.array([-x, -y]),
                     top1=np.array([x, y]))
                for i, (x, y) in enumerate([(.8,.4), (.6,.7), (.7,.6), (.7,.6)])]
        # Original cache.select: objective deduplicate, DTD ascending, first argmin.
        unique = {}
        for r in rows:
            unique.setdefault(tuple(r['ec_objectives']), r)
        ordered = sorted(unique.values(), key=lambda r: r['top1'][0])
        F = np.array([r['ec_objectives'] for r in ordered])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(F, axis=0), axis=1))]
        expected = ordered[int(np.argmin(abs(arc-.25*arc[-1])))]['candidate_id']
        selected = select_branch(rows, dict(kind='pareto_arc', fraction=.25), 0, ['DTD','Cars'])
        self.assertEqual(selected['candidate_id'], expected)

    def test_legacy_monitoring_does_not_change_learning(self):
        from legacy_adapter import imports
        core, _, _, _ = imports()
        groups = {'driver': [0], 'compensator': [1], 'other': [2]}
        rng = np.random.default_rng(10)
        g = core.Geometry(rng.normal(size=(8, 3)), rng.normal(size=(9, 3)),
                          np.array([1., -.3, .2]), np.array([-1., .2, .1]))
        scenarios = [core.Scenario('Cars', .3, .01)]
        first = core.IncrementalController(groups, core.ConstrainedSolver())
        second = copy.deepcopy(first)
        a = first.learn(g, scenarios, max_new=1)
        with tempfile.TemporaryDirectory() as d:
            observer = Monitor(Path(d)/'observations.jsonl')
            b = second.learn(g, scenarios, max_new=1)
            for e in b:
                self.assertIsNone(observer.record(e))
        self.assertEqual(a, b)
        first.recondition(g)
        second.recondition(g)
        for x, y in zip(first.modes, second.modes):
            np.testing.assert_array_equal(x.vector, y.vector)
            np.testing.assert_array_equal(x.map_matrix, y.map_matrix)


if __name__ == '__main__':
    unittest.main()

