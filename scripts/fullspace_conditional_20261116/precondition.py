"""Randomized output Jacobi scaling. It does not select or truncate directions."""
import time
import numpy as np
import torch


def output_diagonal(geometry,tasks,seconds,seed):
    if seconds<=0:raise ValueError('Explicit positive numerical estimation time required')
    generator=torch.Generator(device=geometry.device).manual_seed(seed)
    accumulated={t:np.zeros(geometry.output_dimensions[t]) for t in tasks}
    started=time.monotonic();count=0
    while count==0 or time.monotonic()-started<seconds:
        xi=torch.empty(geometry.dimension,device=geometry.device,dtype=torch.float64)
        xi.bernoulli_(.5,generator=generator).mul_(2).sub_(1)
        for task in tasks:
            y=geometry.tensor_to_numpy(geometry.jvp(task,xi))
            accumulated[task]+=y*y
        count+=1
    return {t:v/count for t,v in accumulated.items()},dict(
        probes=count,elapsed_seconds=time.monotonic()-started,
        full_parameter_dimension=geometry.dimension,
        numerical_estimation_seconds=seconds,seed=seed,
        interpretation='Only estimates diag(J J^T) for congruence scaling; no search-space truncation')
