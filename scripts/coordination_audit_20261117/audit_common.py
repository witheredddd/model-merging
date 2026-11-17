"""Diagnostic timing with nested exclusive totals and synchronized CUDA work."""
import contextlib
import functools
import time
from collections import defaultdict


class Profiler:
    def __init__(self, sync=lambda: None):
        self.sync = sync
        self.stack = []
        self.records = []
        self.phase = 'setup'

    @contextlib.contextmanager
    def measure(self, label):
        self.sync()
        frame = dict(label=label, children=0., phase=self.phase)
        self.stack.append(frame)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            seconds = time.perf_counter() - start
            self.stack.pop()
            if self.stack:
                self.stack[-1]['children'] += seconds
            self.records.append(dict(label=label, phase=frame['phase'],
                                     seconds=seconds,
                                     exclusive_seconds=seconds-frame['children']))

    def wrap(self, obj, method, label):
        original = getattr(obj, method)
        @functools.wraps(original)
        def timed(*args, **kwargs):
            with self.measure(label):
                return original(*args, **kwargs)
        setattr(obj, method, timed)
        return original

    def summary(self):
        result = defaultdict(lambda: dict(calls=0, inclusive_seconds=0., exclusive_seconds=0.))
        for r in self.records:
            row = result[r['phase']+'/'+r['label']]
            row['calls'] += 1
            row['inclusive_seconds'] += r['seconds']
            row['exclusive_seconds'] += r['exclusive_seconds']
        return dict(result)
