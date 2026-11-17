"""Fixed local-chart amplitudes, with real normalized Top-1 vector objectives."""
import numpy as np


class ModeAmplitudeProblem:
    def __init__(self,anchor,decode,measure_top1,normalizers,tasks):
        self.anchor=anchor;self.decode=decode;self.measure=measure_top1
        self.tasks=tuple(tasks);self.normalizers=normalizers
        if any(normalizers[t]<=0 for t in tasks):raise ValueError('Original positive normalization factors must be supplied explicitly')

    def evaluate(self,z):
        point=self.anchor+self.decode(np.asarray(z))
        top1=self.measure(point)
        return np.array([-top1[t]/self.normalizers[t] for t in self.tasks])
