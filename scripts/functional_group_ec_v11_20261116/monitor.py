"""Append-only observer: no thresholds, filtering or return decisions."""
import json
from pathlib import Path
from state_io import jsonable


class Monitor:
    def __init__(self, path=None):
        self.path = Path(path) if path else None

    def record(self, event):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf8') as f:
                f.write(json.dumps(jsonable(event), ensure_ascii=False,
                                   allow_nan=False) + '\n')
        return None

