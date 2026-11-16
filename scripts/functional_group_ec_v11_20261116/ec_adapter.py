"""Native pymoo NSGA-II, with resumable per-individual ask/evaluate/tell."""
import random
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from monitor import Monitor
from state_io import save_pickle, load_pickle, write_json


class GroupProblem(Problem):
    def __init__(self, chart, lower, upper, evaluator, n_obj, motion_domain=None):
        m = len(chart.modes_snapshot)
        self.motion_domain = motion_domain
        n = m if motion_domain is None else motion_domain.gene_to_mode.shape[1]
        lo = np.broadcast_to(np.asarray(lower, dtype=float), (n,)).copy()
        hi = np.broadcast_to(np.asarray(upper, dtype=float), (n,)).copy()
        if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo >= hi):
            raise ValueError('Invalid explicit amplitude domain')
        super().__init__(n_var=n, n_obj=n_obj, xl=lo, xu=hi)
        self.chart_id, self.evaluate_real = chart.chart_id, evaluator
        self.records = []

    def _evaluate(self, X, out, *args, **kwargs):
        values = []
        for genes in np.atleast_2d(X):
            if self.motion_domain is not None:
                z, domain_record = self.motion_domain.map(genes)
            else:
                z, domain_record = genes, None
            F, record = self.evaluate_real(z)
            record = dict(record, chart_id=self.chart_id,
                          proposed_genes=np.array(genes, copy=True), domain_mapping=domain_record,
                          candidate_id=f'{self.chart_id}:eval{len(self.records):06d}')
            self.records.append(record)
            values.append(F)
        out['F'] = np.asarray(values)

    def __getstate__(self):
        state = dict(self.__dict__)
        # Runtime contains a live CUDA model. It is rebound after exact state restore.
        state['evaluate_real'] = None
        return state


def rng_state():
    import torch
    return dict(numpy=np.random.get_state(), python=random.getstate(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    import torch
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint(path, algorithm, pending, index):
    save_pickle(path, dict(algorithm=algorithm, pending=pending,
                          pending_index=index, rng=rng_state()))


def run_ec(chart, evaluator, protocol, out, monitor=None, resume=False,
           initial_population=None, stop_after_evaluations=None, motion_domain=None):
    """stop_after_evaluations is exclusively a software resume-test hook."""
    out.mkdir(parents=True, exist_ok=True)
    observer = monitor or Monitor()
    path = out / 'ec_state.pkl'
    if resume:
        state = load_pickle(path)
        algorithm, pending, index = state['algorithm'], state['pending'], state['pending_index']
        problem = algorithm.problem
        if problem.chart_id != chart.chart_id:
            raise ValueError('Resuming EC state into a different chart')
        problem.evaluate_real = evaluator
        restore_rng(state['rng'])
    else:
        if motion_domain is None:
            lower, upper = protocol['domain']['lower'], protocol['domain']['upper']
        else:
            lower, upper = motion_domain.lower, motion_domain.upper
        problem = GroupProblem(chart, lower, upper, evaluator,
                               protocol['n_objectives'], motion_domain)
        options = dict(pop_size=protocol['ec']['pop_size'], eliminate_duplicates=True)
        if initial_population is not None:
            options['sampling'] = initial_population
        algorithm = NSGA2(**options)
        term = protocol['ec']['termination']
        algorithm.setup(problem, termination=(term['kind'], term['value']),
                        seed=protocol['ec']['seed'], verbose=False)
        pending, index = None, 0
        checkpoint(path, algorithm, pending, index)
    while pending is not None or algorithm.has_next():
        if pending is None:
            pending, index = algorithm.ask(), 0
            if protocol['ec']['termination']['kind'] == 'n_eval':
                remaining = protocol['ec']['termination']['value'] - algorithm.evaluator.n_eval
                pending = pending[:remaining]
            checkpoint(path, algorithm, pending, index)
        while index < len(pending):
            algorithm.evaluator.eval(problem, pending[index:index + 1])
            index += 1
            checkpoint(path, algorithm, pending, index)
            observer.record(problem.records[-1])
            write_json(out / 'progress.json', dict(chart_id=chart.chart_id,
                       evaluations=algorithm.evaluator.n_eval, generation=algorithm.n_gen,
                       pending_completed=index, pending_size=len(pending),
                       n_modes=problem.n_var, last=problem.records[-1]))
            if (stop_after_evaluations is not None and
                    algorithm.evaluator.n_eval >= stop_after_evaluations):
                return dict(complete=False, records=problem.records)
        algorithm.tell(infills=pending)
        pending, index = None, 0
        checkpoint(path, algorithm, pending, index)
    F = np.array([x['ec_objectives'] for x in problem.records])
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    result = dict(complete=True, chart_id=chart.chart_id,
                  n_modes=problem.n_var, evaluations=algorithm.evaluator.n_eval,
                  records=problem.records, front_indices=front,
                  final_population=algorithm.pop.get('X'))
    write_json(out / 'FINAL.json', result)
    return result

