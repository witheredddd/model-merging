"""Use original A100 functions; do not modify any historical source file."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from state_io import digest, write_json

ROOT = Path('/home/shangxueyuan/merging/tsv')
R2 = ROOT / 'scripts/adaptive_fiber_20261116_r2'
COMPROMISE = ROOT / 'scripts/compromise_geometry_20261115'
EXPECTED = {
    R2 / 'fiber_core.py': 'c32c4790dda7de7a1f2b119e995c338782e87948a27815a3423b7551910c45c4',
    R2 / 'fast_online.py': 'fbf9cbc61266f7491fcebff9e8f5f0fab52ad191145636d081925aee908194b9',
    R2 / 'run_a100.py': '5c4b04da2a8e06add7ba40241fc74d772672326bc59be9706eb7df34486e75e8',
    R2 / 'resume_state.py': '1bc876e070b329cbb37805a8c88f5c8335857d8dd2f25ecdc83ee05de77cc64a',
    COMPROMISE / 'space.py': '0a29085e1cfb601c71525e4d6cfa430146ea211a80328b3517b6809879fa77a9',
    COMPROMISE / 'baselines.py': 'a0b535757ce8fa856f65886ec6e59e93c658d2ddf9a644c082df0e099f4c4f99',
}


def imports():
    sys.path.insert(0, str(R2))
    sys.path.insert(0, '/home/shangxueyuan/merging/compromise_geometry_deps')
    import fiber_core
    import fast_online
    import run_a100
    import resume_state
    return fiber_core, fast_online, run_a100, resume_state


class Runtime:
    def __init__(self, out, checkpoint):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        hashes = {str(p): digest(p) for p in EXPECTED}
        if any(hashes[str(p)] != h for p, h in EXPECTED.items()):
            raise RuntimeError('Original source identity differs; not a scientific gate')
        self.core, self.fast, self.old, self.resume = imports()
        self.c, self.s, self.norm = self.old.setup(self.out)
        self.cost = self.fast.Costs(self.c)
        self.solver = self.fast.instrument_solver(self.cost)
        self.geom = self.old.cached_geometry(0)
        self.groups = {g['key']: g['indices'] for g in self.s.groups if g['indices']}
        state = self.resume.restore(checkpoint, self.groups, self.solver,
                                    self.geom.budget_references)
        self.a, self.policy = state['a'], state['policy']
        self.learning_next_step = state['next_step']
        self.original_records = state['records']
        self.source_hash = state['checkpoint_sha256']
        lock_path = Path(checkpoint).parent.parent / 'fast_lock.json'
        self.original_lock = json.loads(lock_path.read_text())
        self.original_lock_path = lock_path
        if not torch.equal(self.point(self.a).cpu(), state['point']):
            raise RuntimeError('Checkpoint and complete reconstructed visual parameters differ')
        self.geom.column_ages[:] = self.learning_next_step
        self.radius = .003 * self.norm(self.s.aD - self.s.aC)
        protocol = Path('/media/shangxueyuan/data1/compromise_geometry_20261115/fitness_and_scope_audit.json')
        old_config = json.loads(protocol.read_text())
        self.tasks = ('DTD', 'Cars')
        self.normalizers = np.array([old_config['normalizers'][t] for t in self.tasks])
        normalizer_path = Path(old_config['normalizer_path'])
        if digest(normalizer_path) != old_config['normalizer_sha256']:
            raise RuntimeError('Original archived normalizer source hash differs')
        values = json.loads(normalizer_path.read_text())
        if not np.array_equal(self.normalizers, [values[t + 'Val'] for t in self.tasks]):
            raise RuntimeError('Normalizer source/value mismatch')
        self.provenance = dict(source_hashes=hashes, checkpoint=str(checkpoint),
                               checkpoint_sha256=self.source_hash,
                               exact_full_visual_equality=True,
                               normalizer_protocol=str(protocol),
                               normalizer_protocol_sha256=digest(protocol),
                               normalizer_path=str(normalizer_path),
                               normalizers=self.normalizers.tolist(),
                               P=int(state['point'].numel()), d_E=len(self.a),
                               inherited_learning_next_step=self.learning_next_step,
                               inherited_actual_responses=len(self.policy.actual_responses),
                               original_run_lock=str(lock_path), original_run_lock_sha256=digest(lock_path))
        write_json(self.out / 'SOURCE_AUDIT.json', self.provenance)

    def point(self, a):
        return self.s.point(torch.as_tensor(a, device='cuda', dtype=torch.float64))

    def current_chart_geometry(self, extend_births=False):
        """Extract original fast_online refresh/recondition/learn ordering unchanged."""
        core, fast = self.core, self.fast
        taus = self.original_lock['taus']
        scenarios = [core.Scenario(t, tau, self.radius)
                     for t in ['Cars', 'DTD'] for tau in taus]
        keys = fast.shortlist(self.policy, self.geom, scenarios, self.groups)
        B = self.policy.basis(self.geom.dim)
        before = np.r_[self.geom.JD @ B, self.geom.JC @ B]
        self.geom, ids = fast.refresh(self.c, self.s, self.geom, self.a, keys, self.groups)
        self.geom, _ = fast.exact_basis(self.c, self.s, self.geom, self.a, B)
        after = np.r_[self.geom.JD @ B, self.geom.JC @ B]
        discrepancy = float(np.linalg.norm(after - before) / max(np.linalg.norm(after), 1e-8))
        self.policy.recondition(self.geom)
        # This is the original r2 reliability policy, not a new decision criterion.
        reliability = self.original_lock['reliability_tolerance']
        if discrepancy > reliability:
            keys |= fast.shortlist(self.policy, self.geom, scenarios, self.groups, extra=2)
            self.geom, more = fast.refresh(self.c, self.s, self.geom, self.a, keys, self.groups)
            ids = sorted(set(ids + more))
            self.policy.recondition(self.geom)
            self.geom, _ = fast.exact_basis(self.c, self.s, self.geom, self.a,
                                             self.policy.basis(self.geom.dim))
        births = []
        if extend_births:
            radius = self.radius * (.5 if discrepancy > reliability else 1.)
            scenarios = [core.Scenario(t, tau, radius)
                         for t in ['Cars', 'DTD'] for tau in taus]
            self.policy.groups = {k: self.groups[k] for k in keys}
            try:
                births = self.policy.learn(self.geom, scenarios,
                                          max_new=1 if self.learning_next_step == 0 else 2)
            finally:
                self.policy.groups = self.groups
        self.geom, new_error = fast.exact_basis(self.c, self.s, self.geom, self.a,
                                                  self.policy.basis(self.geom.dim))
        return dict(refreshed_columns=ids, basis_drift=discrepancy,
                    new_basis_error=new_error, births=births,
                    learning_next_step=self.learning_next_step,
                    extension_to_ec_anchor=bool(extend_births),
                    original_accepted_history_unchanged=True,
                    cost=self.cost.snapshot(self.policy))

    def state(self):
        return dict(a=self.a, modes=self.policy.modes, geom=self.geom,
                    trajectories=self.policy.trajectories,
                    actual_responses=self.policy.actual_responses,
                    events=self.policy.events, learning_next_step=self.learning_next_step)

    def apply_state(self, state):
        self.a = state['a'].copy()
        self.geom = state['geom']
        self.policy.modes = state['modes']
        self.policy.trajectories = state['trajectories']
        self.policy.actual_responses = state['actual_responses']
        self.policy.events = state['events']
        self.learning_next_step = state['learning_next_step']
        self.policy.teacher_cache = {}
        self.policy.teacher_geometry = None

