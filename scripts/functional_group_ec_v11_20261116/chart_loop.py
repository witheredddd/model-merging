"""Actual model EC and discrete position-dependent compensation charts."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from chart_state import freeze, describe, coordinates
from legacy_adapter import Runtime, imports
from monitor import Monitor
from protocol import load_protocol, select_branch
from state_io import digest, write_json, save_pickle, load_pickle


def prepare(args):
    out = args.output
    if (out / 'PREPARED.json').exists():
        raise RuntimeError('Prepared state already exists; use run --prepared, not an overwrite')
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    write_json(out / 'status.json', dict(stage='restore_original_checkpoint'))
    r = Runtime(out / 'runtime', args.checkpoint)
    initial_a = r.a.copy()
    write_json(out / 'status.json', dict(stage='original_active_geometry_refresh',
              n_modes=len(r.policy.modes), source=r.provenance))
    # No unknown EC domain is needed for original endpoint geometry preparation.
    event = r.current_chart_geometry(extend_births=False)
    chart = freeze('chart000', r.source_hash, r.a, r.policy, r.geom, 'prepared_source_state')
    description = describe(chart, r.groups, r.geom)
    save_pickle(out / 'prepared.pkl', dict(runtime=r.state(), chart=chart,
                checkpoint=str(args.checkpoint), source=r.provenance,
                original_anchor_a=initial_a, geometry_refresh=event))
    temp = out / 'prepared_model.tmp'
    torch.save(dict(full_visual_flat=r.point(r.a).cpu(), coordinates=torch.from_numpy(r.a),
                    modes=chart.modes_snapshot, groups=r.s.groups), temp)
    temp.replace(out / 'prepared_model.pt')
    write_json(out / 'GROUPS.json', description)
    write_json(out / 'PREPARED.json', dict(complete=True, kind='real_A100_geometry_preparation',
                elapsed_seconds=time.monotonic() - start, source=r.provenance,
                n_modes=len(chart.modes_snapshot), geometry_refresh=event,
                new_ec_evaluations=0, control_updates=0,
                parameters_changed=False, source_checkpoint_unchanged=
                digest(args.checkpoint) == r.source_hash))
    write_json(out / 'status.json', dict(stage='prepared_waiting_for_explicit_run_protocol',
                n_modes=len(chart.modes_snapshot), cost=r.cost.snapshot(r.policy)))


def projected_initial_population(previous, next_chart, protocol):
    """Only a named, explicit reproject protocol calls this function."""
    Z = np.linalg.lstsq(next_chart.mode_matrix,
                       (np.array([r['a'] for r in previous]) - next_chart.anchor_a).T,
                       rcond=None)[0].T
    lo = np.broadcast_to(protocol['domain']['lower'], (Z.shape[1],))
    hi = np.broadcast_to(protocol['domain']['upper'], (Z.shape[1],))
    if np.any(Z < lo) or np.any(Z > hi):
        raise ValueError('Specified reproject protocol has out-of-domain seeds; no implicit clipping')
    return Z, [dict(old_candidate_id=r['candidate_id'], projected_z=z,
                    reexpression_residual=float(np.linalg.norm(coordinates(next_chart, z) - r['a'])),
                    must_be_reevaluated=True) for r, z in zip(previous, Z)]


def run(args):
    from decode_fitness import RealEvaluator
    from ec_adapter import run_ec
    from control_domain import MotionDomain, parameter_metric
    protocol = load_protocol(args.protocol)
    protocol['n_objectives'] = 2
    out = args.output
    imports()  # Original Geometry/Mode classes must exist before trusted pickle restore.
    prepared = load_pickle(args.prepared / 'prepared.pkl')
    out.mkdir(parents=True, exist_ok=True)
    protocol_hash = digest(args.protocol)
    manifest_path = out / 'experiment_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not args.resume or manifest['protocol_sha256'] != protocol_hash:
            raise ValueError('Use an explicit resume with identical run protocol or a new directory')
    else:
        manifest = dict(protocol=protocol, protocol_sha256=protocol_hash,
                 source=prepared['source'], prepared_path=str(args.prepared),
                 design_sha256=digest(Path(__file__).with_name('CODE_DESIGN_v1.1.md')),
                 code_hashes={p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                 data_roles='p0 geometry and EC; p4/p5 exploratory final only; p6/Test unopened',
                 no_added_EC_constraints=True, input_dimension=304,
                 ec_dimension='len(frozen_chart.modes_snapshot)')
        write_json(manifest_path, manifest)
    r = Runtime(out / 'runtime', Path(prepared['checkpoint']))
    Q_metric = parameter_metric(r.s)
    r.apply_state(copy.deepcopy(prepared['runtime']))
    state_path = out / 'loop_state.pkl'
    if args.resume and state_path.exists():
        loop = load_pickle(state_path)
        r.apply_state(loop['runtime'])
    else:
        loop = dict(chart_index=0, archive=[], transitions=[], phase='chart_ready',
                    parent=None, runtime=r.state(), initial_population=None)
        save_pickle(state_path, loop)
    while loop['chart_index'] < protocol['max_charts']:
        k = loop['chart_index']
        folder = out / 'charts' / f'{k:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        chart_path = folder / 'chart_state.pkl'
        if chart_path.exists():
            chart = load_pickle(chart_path)
        else:
            chart = freeze(f'chart{k:03d}', r.source_hash, r.a, r.policy, r.geom,
                           protocol['protocol_id'], loop['parent'])
            save_pickle(chart_path, chart)
            write_json(folder / 'GROUPS.json', describe(chart, r.groups, r.geom))
        evaluator = RealEvaluator(r, chart)
        domain = MotionDomain(chart.mode_matrix, Q_metric, r.radius, r.core.ortho)
        write_json(folder / 'CONTROL_DOMAIN.json', domain.describe())
        write_json(out / 'status.json', dict(stage='real_group_amplitude_EC', chart=k,
                   n_modes=len(chart.modes_snapshot), protocol=protocol['protocol_id']))
        result = run_ec(chart, evaluator, protocol, folder,
                        Monitor(folder / 'evaluations.jsonl'),
                        resume=bool(args.resume and (folder / 'ec_state.pkl').exists()),
                        initial_population=loop['initial_population'], motion_domain=domain)
        # Replace this chart's archive on resume, preserving all true evaluated models.
        loop['archive'] = [x for x in loop['archive'] if x['chart_id'] != chart.chart_id]
        loop['archive'].extend(result['records'])
        write_json(out / 'archive.json', loop['archive'])
        if k + 1 >= protocol['max_charts']:
            break
        front = [result['records'][int(i)] for i in sorted(result['front_indices'])]
        parent = select_branch(front, protocol['branch'], k, list(r.tasks))
        before = r.a.copy()
        old_history = (len(r.policy.trajectories), len(r.policy.actual_responses))
        old_modes = copy.deepcopy(r.policy.modes)
        r.a = np.array(parent['a'], copy=True)
        r.geom.column_ages += 1
        refresh_event = r.current_chart_geometry(protocol['learn_at_ec_anchors'])
        if old_history != (len(r.policy.trajectories), len(r.policy.actual_responses)):
            raise RuntimeError('EC movement must not enter original accepted-learning history')
        next_chart = freeze(f'chart{k+1:03d}', r.source_hash, r.a, r.policy, r.geom,
                            protocol['protocol_id'], parent['candidate_id'])
        initial = None
        migration = []
        if protocol['population_transition']['kind'] == 'reproject':
            # Migrate the actual final population, not an invented subset or top-mean score.
            old_pop = result['final_population']
            previous = [dict(candidate_id=f'{chart.chart_id}:population{i}',
                             a=coordinates(chart, z)) for i, z in enumerate(old_pop)]
            initial, migration = projected_initial_population(previous, next_chart, protocol)
        drift = [dict(mode_index=j,
                      vector_change=float(np.linalg.norm(m.vector - old_modes[j].vector)),
                      map_change=float(np.linalg.norm(np.asarray(m.map_matrix) -
                                                      np.asarray(old_modes[j].map_matrix))))
                 for j, m in enumerate(r.policy.modes[:len(old_modes)])]
        transition = dict(event_source='ec_chart_move', chart_from=k, chart_to=k+1,
                     parent_candidate_id=parent['candidate_id'], a_before=before,
                     a_after=r.a.copy(), actual_coordinate_increment=r.a-before,
                     branch_protocol=protocol['branch'], migration=migration,
                     geometry_refresh=refresh_event, old_mode_changes=drift,
                     accepted_learning_history_unchanged=True)
        write_json(folder / 'transition.json', transition)
        loop['transitions'].append(transition)
        loop.update(chart_index=k+1, parent=parent['candidate_id'], runtime=r.state(),
                    initial_population=initial)
        save_pickle(state_path, loop)
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    indices = NonDominatedSorting().do(np.array([x['ec_objectives'] for x in loop['archive']]),
                                        only_non_dominated_front=True)
    write_json(out / 'FINAL.json', dict(complete=True, charts=loop['chart_index']+1,
               evaluations=len(loop['archive']), pareto=[loop['archive'][int(i)] for i in indices],
               transitions=loop['transitions'], cost=r.cost.snapshot(r.policy),
               p6_and_Test_unopened=True, dimension_converged=False))
    write_json(out / 'status.json', dict(stage='complete',
               evaluations=len(loop['archive']), n_modes=len(r.policy.modes)))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--checkpoint', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    execute = sub.add_parser('run')
    execute.add_argument('--prepared', type=Path, required=True)
    execute.add_argument('--protocol', type=Path, required=True)
    execute.add_argument('--output', type=Path, required=True)
    execute.add_argument('--resume', action='store_true')
    args = p.parse_args()
    try:
        (prepare if args.command == 'prepare' else run)(args)
    except Exception:
        import traceback
        write_json(args.output / 'FAILURE.json', dict(traceback=traceback.format_exc(),
                                                     scientific_failure=False))
        raise


if __name__ == '__main__':
    main()
