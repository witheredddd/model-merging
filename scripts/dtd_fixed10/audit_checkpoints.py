"""Read-only CPU audit of this run's saved states; does not resume training."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def audit(root):
    torch.set_num_threads(4)
    identity = json.loads((root / 'config.json').read_text())
    result = {'checkpoint_metadata_checked': True,
              'live_interrupt_and_replay_tested': False, 'arms': {}}
    basis = torch.load(root / 'fixed_basis.pt', map_location='cpu',
                       weights_only=True, mmap=True)
    assert basis['identity'] == identity
    assert tuple(basis['B_rows'].shape) == (10, 84934656)
    result['basis_shape'] = list(basis['B_rows'].shape)
    for arm in ('fixed10', 'full_adamw'):
        folder = root / arm
        events = json.loads((folder / 'events.json').read_text())
        report = json.loads((folder / 'report.json').read_text())
        assert [e['step'] for e in events] == list(range(1, 129))
        paths = sorted(folder.glob('step_*.pt'))
        assert len(paths) == 16
        expected_numel = 10 if arm == 'fixed10' else 84934656
        steps = []
        for path in paths:
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            step = int(path.stem.split('_')[-1])
            assert state['identity'] == identity and state['step'] == step
            assert state['events'] == events[:step]
            assert state['variable'].numel() == expected_numel
            optstate = next(iter(state['optimizer']['state'].values()))
            assert int(optstate['step']) == step
            assert optstate['exp_avg'].numel() == expected_numel
            assert optstate['exp_avg_sq'].numel() == expected_numel
            steps.append(step)
        assert steps == list(range(8, 129, 8))
        recovery = torch.load(folder / 'recovery.pt', map_location='cpu',
                              weights_only=True, mmap=True)
        assert recovery['step'] == 128 and recovery['identity'] == identity
        selected = torch.load(folder / 'selected.pt', map_location='cpu',
                              weights_only=True, mmap=True)
        assert selected['best'] == report['selected']
        chosen = torch.load(folder / f"step_{report['selected']['step']:06d}.pt",
                            map_location='cpu', weights_only=True, mmap=True)
        assert torch.equal(selected['variable'], chosen['variable'])
        result['arms'][arm] = {'saved_steps': steps, 'recovery_step': 128,
                               'selected_step': report['selected']['step'],
                               'selected_matches_saved_state': True,
                               'variable_numel': expected_numel}
    result['source_sha256'] = identity['source_sha256']
    result['matrix_sha256'] = hashlib.sha256(
        Path('/media/shangxueyuan/data1/DTD_fisher_blocks_expanded_20261113/block_matrices.npz').read_bytes()
    ).hexdigest()
    (root / 'checkpoint_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
