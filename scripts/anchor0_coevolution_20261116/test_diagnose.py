import numpy as np
from diagnose import subspace,trajectory,fit,outside

def test_persistent_modes_are_separate_from_birth_rank():
    A=np.eye(4)[:,:1];B=np.eye(4)[:,:2];x=subspace(A,B)
    assert abs(x['projector_Frobenius']-1)<1e-12
    assert x['matched_sin_Frobenius']<1e-12 and x['unmatched_dimensions']==1

def test_finite_trajectory_energy_and_innovation():
    R=np.array([[1.,2.,0.],[0.,0.,1.]])
    x=trajectory(R)
    assert np.allclose(x['cumulative_energy'],[5/6,1,1])
    assert np.allclose(x['relative_innovation'],[1,0,1])
    assert x['numerical_rank']==2

def test_tangent_fit_and_new_component():
    a=np.array([1.,2.]);x=fit(a,a)
    assert x['relative_error']==0 and abs(x['cosine']-1)<1e-12
    assert abs(outside(np.array([1.,1.]),np.array([[1.],[0.]]))-np.sqrt(.5))<1e-12
