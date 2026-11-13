"""Small coefficient-space algebra; no tall parameter-matrix factorizations."""
import numpy as np

def orthogonal_coefficients(coefficients,gram,rtol=1e-8):
    matrix=coefficients.T@gram@coefficients
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    keep=values>max(float(values[-1])*rtol,1e-14)
    return coefficients@vectors[:,keep]/np.sqrt(values[keep])[None,:]

def trust(a,gram,radius):
    norm=float(np.sqrt(max(0.,a@gram@a)))
    return a*min(1.,radius/max(norm,1e-30))

def refusal(fit_gain,iterations,held_before,held_virtual,held_rebuilt):
    if not iterations or fit_gain<=0:return 'fit_no_strict_improvement'
    if held_virtual>=held_before:return 'cross_batch_no_improvement'
    if held_rebuilt>=held_before:return 'fp32_reconstruction_gate_flip'
    return 'accepted'
