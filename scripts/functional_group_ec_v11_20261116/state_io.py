"""Atomic state persistence. Monitoring never provides control decisions."""
import hashlib
import json
import os
import pickle
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(x):
    if hasattr(x, 'tolist'):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(jsonable(value), ensure_ascii=False,
                               indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)


def save_pickle(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def load_pickle(path):
    # Only locally generated, trusted runtime state is accepted by the runner.
    with Path(path).open('rb') as f:
        return pickle.load(f)

