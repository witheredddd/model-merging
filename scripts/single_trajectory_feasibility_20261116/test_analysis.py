import numpy as np
from analyze import log_map,dlog,centered_ensemble_covariance

def test_single_curve_population_covariance_is_zero():
    x=np.arange(30.).reshape(1,5,6)
    assert np.array_equal(centered_ensemble_covariance(x),np.zeros((1,1)))

def test_product_sphere_log_has_correct_length_and_tangency():
    R=2/np.sqrt(2);ref=np.array([[R,0.,0.],[R,0.,0.]])
    q=np.array([[R*np.cos(.03),R*np.sin(.03),0.],[R*np.cos(.02),0.,R*np.sin(.02)]])
    L=log_map(ref,q)
    assert np.allclose(np.linalg.norm(L,axis=1),R*np.array([.03,.02]),atol=1e-12)
    assert np.linalg.norm(np.sum(L*ref,axis=1))<1e-12

def test_dlog_matches_spherical_finite_difference():
    R=2.;ref=np.array([[R,0.,0.]]);q=np.array([[R*np.cos(.02),R*np.sin(.02),0.]])
    v=np.array([[-np.sin(.02),np.cos(.02),.3]])
    h=1e-6
    def sphere(x):return x*R/np.linalg.norm(x,axis=1)[:,None]
    fd=(log_map(ref,sphere(q+h*v))-log_map(ref,sphere(q-h*v)))/(2*h)
    assert np.allclose(dlog(ref,q,v),fd,rtol=1e-6,atol=1e-8)
