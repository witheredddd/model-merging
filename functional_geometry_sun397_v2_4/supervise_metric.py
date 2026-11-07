import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

out = Path('/root/autodl-tmp/tsv/results/SUN397_rank20_parameter_metric')
out.mkdir(parents=True, exist_ok=True)
lock = (out/'supervisor.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
micro = 32
next_report = 0
with (out/'supervisor.log').open('a', buffering=1) as supervisor_log:
    for attempt in range(4):
        if (out/'result.json').exists():
            break
        with (out/'run.log').open('ab', buffering=0) as run_log:
            start_offset = run_log.tell()
            child = subprocess.Popen([sys.executable, '-B', '-u', str(Path(__file__).with_name('train_metric.py')),
                                      '--micro', str(micro)], stdout=run_log, stderr=subprocess.STDOUT)
            supervisor_log.write(f'launch pid={child.pid} micro={micro}\n')
            while child.poll() is None:
                if time.time() >= next_report:
                    report = dict(time=time.time(), pid=child.pid, supervisor_pid=os.getpid(), micro=micro)
                    for name in ('status.json', 'validation_latest.json'):
                        try:
                            report[name] = json.loads((out/name).read_text())
                        except (ValueError, OSError):
                            pass
                    supervisor_log.write(json.dumps(report) + '\n')
                    next_report = time.time() + 3600
                time.sleep(10)
        if child.returncode == 0 and (out/'result.json').exists():
            supervisor_log.write('completed\n')
            break
        with (out/'run.log').open('rb') as run_log:
            run_log.seek(max(start_offset, (out/'run.log').stat().st_size - 16000))
            failure = run_log.read().decode(errors='replace')
        supervisor_log.write(f'exit code={child.returncode}\n')
        if 'out of memory' in failure.lower() and micro > 1:
            micro //= 2
        elif child.returncode not in (-9, -15):
            (out/'needs_repair.json').write_text(json.dumps(dict(error=failure, time=time.time()), indent=2))
            raise SystemExit(child.returncode or 1)
        time.sleep(10)
