"""DTD functional geometry and gauge-invariant diagnostics."""
import numpy as np

def sphere_log(base,new):
    # phi=2 sqrt(p)/sqrt(N): a product of N spheres, not one large sphere.
    base=np.asarray(base);new=np.asarray(new);radius=2/np.sqrt(len(base))
    x=base/radius;z=new/radius;dot=np.clip(np.sum(x*z,axis=1),-1.,1.)
    tangent=z-dot[:,None]*x;sin=np.linalg.norm(tangent,axis=1);angle=np.arctan2(sin,dot)
    factor=np.ones_like(angle);np.divide(angle,sin,out=factor,where=sin>np.finfo(float).eps)
    if np.any((sin<=np.finfo(float).eps)&(dot<0)):raise ArithmeticError('Antipodal Log undefined')
    return radius*factor[:,None]*tangent

def functional_innovation(existing,response,sample_classes=None):
    response=np.asarray(response)
    if existing.shape[1]:
        z=np.linalg.lstsq(existing,response,rcond=None)[0];r=response-existing@z
    else:r=response.copy()
    bound=np.finfo(float).eps*(max(existing.shape)+4)*(np.linalg.norm(response)+np.linalg.norm(existing))
    lower=None
    if sample_classes:
        energy=np.sum(r.reshape(-1,sample_classes)**2,axis=1)
        lower=float(energy.mean()-1.96*energy.std(ddof=1)/np.sqrt(len(energy))) if len(energy)>1 else None
    return dict(residual_norm=float(np.linalg.norm(r)),response_norm=float(np.linalg.norm(response)),
        independent=bool(np.linalg.norm(r)>bound and (lower is None or lower>0)),
        mean_sample_energy_approx95_lower=lower,machine_comparison_bound=float(bound))

def principal_geometry(cross,old_dim,new_dim):
    s=np.clip(np.linalg.svd(cross,compute_uv=False),0.,1.)
    return dict(principal_angles_radians=np.arccos(s).tolist(),
        projector_distance_squared=float(old_dim+new_dim-2*np.sum(s*s)),
        old_dimension=old_dim,new_dimension=new_dim,dimension_changed=old_dim!=new_dim)

def align_rotation(cross):
    U,_,Vt=np.linalg.svd(cross,full_matrices=False);return Vt.T@U.T
