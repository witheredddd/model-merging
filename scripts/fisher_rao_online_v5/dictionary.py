"""Tall parameter dictionaries use vector orthogonalization, not cuSOLVER SVD."""
import torch


def streaming_dictionary(directions,maximum_rank,relative_tolerance=1e-6):
    if not directions or maximum_rank<1 or relative_tolerance<=0:
        raise ValueError('Invalid parameter dictionary inputs')
    shape=directions[0].shape;device=directions[0].device;dtype=directions[0].dtype
    columns=[]
    for direction in directions:
        if direction.ndim!=1 or direction.shape!=shape or direction.device!=device:
            raise ValueError('Parameter candidate shapes/devices differ')
        residual=direction.detach().double().clone()
        norm=residual.norm()
        if not torch.isfinite(norm):raise ValueError('Nonfinite parameter candidate')
        if float(norm)==0:continue
        # FP64 dot products and a second pass maintain a stable small basis
        # without ever asking a dense linalg backend to factor D x n.
        for _ in range(2):
            for axis in columns:residual=residual-axis*torch.dot(axis,residual)
        remaining=residual.norm()
        if float(remaining)>relative_tolerance*float(norm):
            columns.append(residual/remaining)
        if len(columns)>=maximum_rank:break
    if not columns:raise ValueError('No independent parameter candidate')
    return torch.stack(columns,dim=1).to(dtype)
