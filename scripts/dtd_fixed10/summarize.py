"""Verify paired records and render a standalone scientific training figure."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    here = Path(__file__).resolve().parent
    root = here / 'evidence/20261113'
    read = lambda name: json.loads((root / name).read_text())
    config, projection = read('config.json'), read('projection.json')
    assert hashlib.sha256((here / 'run.py').read_bytes()).hexdigest() == config['source_sha256']
    assert read('complete.json')['status'] == 'complete'
    events = {arm: read(f'{arm}/events.json') for arm in ('fixed10', 'full_adamw')}
    for arm, rows in events.items():
        assert [e['step'] for e in rows] == list(range(1, 129))
        assert all(len(e['ids']) == 32 for e in rows)
        assert [e['step'] for e in rows if 'validation' in e] == list(range(8, 129, 8))
    assert [e['ids'] for e in events['fixed10']] == [e['ids'] for e in events['full_adamw']]
    split = config['split']
    assert len(split['Vcal']) == 128 and len(split['Xprime']) == 512
    assert not set(split['Vcal']) & set(split['Xprime'])
    training = {i for e in events['fixed10'] for i in e['ids']}
    for name in ('Vcal', 'Xprime', 'X'):
        assert not training & set(split[name])
    for name in ('Vcal', 'Xprime'):
        assert projection['evaluations'][name]['sample_ids'] == split[name]
        assert projection['evaluations'][name]['prediction_agreement'] == 1
    audit = dict(steps_per_arm=128, images_per_arm=4096,
                 paired_training_ids_identical=True, development_ids_disjoint=True,
                 optimizer_ids_exclude_probe_and_development=True,
                 public_source_matches_executed_source=True,
                 service_exit_status=0, official_test_opened=False,
                 basis_from_prior_complete_history=True,
                 endpoint_read_for_projection=True,
                 adapter_assets_warning=(
                     'assets.json is unmodified metadata from the frozen v5 ClipTask adapter. '
                     'Its algorithm/parameter_state/expert_endpoints_read fields describe that '
                     'adapter construction, not this fixed10 experiment. This experiment explicitly '
                     'reads historical Fisher matrices and the endpoint update; use config.json, '
                     'projection.json and this effective protocol for experiment provenance.'),
                 live_interrupt_and_replay_tested=False)
    (root / 'effective_protocol.json').write_text(json.dumps(audit, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for arm, label in [('fixed10', 'Fixed 10D (LR 0.01)'), ('full_adamw', 'Full AdamW (LR 1e-5)')]:
        rows = [e for e in events[arm] if 'validation' in e]
        axes[0].plot([e['step'] for e in rows], [e['validation']['CE'] for e in rows], 'o-', label=label)
        axes[1].plot([e['step'] for e in rows], [100*e['validation']['accuracy'] for e in rows], 'o-', label=label)
    original = projection['evaluations']['Vcal']['original']
    axes[0].axhline(original['CE'], color='grey', linestyle='--', label='Prior v5 endpoint')
    axes[1].axhline(100*original['accuracy'], color='grey', linestyle='--', label='Prior v5 endpoint')
    for ax, ylabel in zip(axes, ['Vcal cross entropy', 'Vcal accuracy (%)']):
        ax.set(xlabel='Optimizer updates (32 images each)', ylabel=ylabel)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('DTD: fixed history-informed space vs full-parameter control')
    fig.savefig(root / 'training.png', dpi=180)
    fig.savefig(root / 'training.pdf')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
