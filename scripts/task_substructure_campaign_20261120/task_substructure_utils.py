"""Scalar policies shared by unattended baseline generation."""
import math

def intervention_scale(norm,radius,original_scale=False):
    if not math.isfinite(norm) or norm<0:raise ValueError('Invalid parameter norm')
    if not math.isfinite(radius) or radius<=0:raise ValueError('Invalid radius')
    if original_scale:return 1.
    return radius/norm if norm else 0.
