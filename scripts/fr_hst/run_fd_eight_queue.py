"""Train the eight Fisher–Rao experts sequentially, then audit their controls."""

import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys

from train_eight_expert_queue import BASE, BATCH_SIZE, HEADS, ROOT, existing, read
from task_data import TASKS, test_protocol, test_set, train_set


# Epochs at the best saved checkpoint of the preceding eight-expert run.
EXPERT_EPOCHS = dict(SUN397=12, Cars=17, EuroSAT=6, GTSRB=2, MNIST=1, RESISC45=7, SVHN=3)
WARMUP = 36
BIRTH_PENALTY = .0002
MAX_COMPONENTS = 8
HERE = Path(__file__).resolve().parent


def preflight():
    plan = []
    for task in TASKS:
        train, test = train_set(task), test_set(task)
        head = HEADS / f'{task}.pt'
        manifest = read(head.with_suffix('.json'))
        if manifest['task'] != task or manifest['sha256'] != existing.sha(head):
            raise ValueError(f'{task}: classification head manifest mismatch')
        if hasattr(train, 'class_to_idx'):
            if train.class_to_idx != test.class_to_idx or len(train.class_to_idx) != manifest['classes']:
                raise ValueError(f'{task}: train, Test and classification head classes differ')
        reference = read(ROOT / 'results/expert_references' / f'{task}.json')['test']
        if task == 'EuroSAT':
            approved = ROOT / 'results/EXPERTS8_99_20261109' / task / 'best.pt'
            if not approved.is_file():
                raise FileNotFoundError(approved)
        elif len(test) != reference['n']:
            raise ValueError(f'{task}: configured Test size differs from the original reference')
        steps = 216 if task == 'DTD' else EXPERT_EPOCHS[task] * math.ceil(len(train) / BATCH_SIZE)
        plan.append(dict(task=task, steps=steps, train_n=len(train), test=test_protocol(task,test),
                         head_sha256=manifest['sha256']))
    return dict(base_sha256=existing.sha(BASE),warmup=WARMUP,birth_penalty=BIRTH_PENALTY,
                max_components=MAX_COMPONENTS,tasks=plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    plan = preflight()
    if args.preflight_only:
        print(json.dumps(plan, indent=2))
        return

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status = output / 'queue.json'
    with (output / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if status.exists() and read(status)['plan'] != plan:
            raise ValueError('Existing queue plan differs')
        rows = {row['task']: dict(training='pending', audit='pending') for row in plan['tasks']}

        def publish():
            existing.write(status, dict(plan=plan, progress=rows))

        for entry in plan['tasks']:
            task = entry['task']
            row = rows[task]
            folder = output / task
            row['training'] = 'running'
            publish()
            print(f'{task}: starting {entry["steps"]} training steps', flush=True)
            cmd = [sys.executable, '-u', str(HERE / 'train_fd_task.py'), '--task', task,
                   '--arm', 'adaptive_dynamics', '--steps', str(entry['steps']), '--warmup', str(WARMUP),
                   '--birth-penalty', str(BIRTH_PENALTY), '--max-components', str(MAX_COMPONENTS),
                   '--output', str(folder)]
            child = subprocess.run(cmd, check=False)
            if child.returncode:
                row.update(training='failed', training_returncode=child.returncode, audit='skipped')
            else:
                result = read(folder / 'results.json')['arms']['adaptive_dynamics']
                row.update(training='complete', test=result['test'], quality=result['quality'],
                           checkpoint_sha256=result['checkpoint_sha256'], components=result['learned']['components'])
                print(f'{task}: {row["test"]["accuracy"]:.4%}, {row["quality"]["gate"]}', flush=True)
            publish()

        for task, row in rows.items():
            if row['training'] != 'complete':
                continue
            folder = output / task
            controls = folder / 'controls' / f'{task}_controls.json'
            if controls.exists():
                cached = read(controls)
                if (cached['arm']['checkpoint_sha256'] != row['checkpoint_sha256']
                        or cached['test_provenance'] != row['test']['provenance']):
                    raise ValueError(f'{task}: existing audit belongs to a different checkpoint or Test')
                row['audit'] = 'complete'
            else:
                row['audit'] = 'running'
                publish()
                print(f'{task}: starting component sensitivity audit', flush=True)
                cmd = [sys.executable, '-u', str(HERE / 'audit_task_controls.py'), '--task', task,
                       '--folder', str(folder / 'adaptive_dynamics'), '--output', str(controls.parent),
                       '--pretrained', str(BASE)]
                child = subprocess.run(cmd, check=False)
                row['audit'] = 'complete' if child.returncode == 0 else 'failed'
                if child.returncode:
                    row['audit_returncode'] = child.returncode
            publish()
    if any(row['training'] != 'complete' or row['audit'] != 'complete' for row in rows.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
