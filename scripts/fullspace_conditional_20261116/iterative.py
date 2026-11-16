"""Checkpointable ordinary CG; iteration chunks do not restart its directions."""
import numpy as np


def continuing_cg(A,b,*,rtol,max_iterations,state=None,callback=None):
    b=np.asarray(b,dtype=np.float64);limit=rtol*np.linalg.norm(b)
    if state is None:
        x=np.zeros_like(b);r=b.copy();p=r.copy();rr=float(r@r);iterations=0
    else:
        x=np.asarray(state['x']).copy();r=np.asarray(state['r']).copy()
        p=np.asarray(state['p']).copy();rr=float(state['rr']);iterations=int(state['iterations'])
        if any(v.shape!=b.shape for v in (x,r,p)):raise ValueError('CG checkpoint dimensions differ')
    def payload():
        return dict(x=x.copy(),r=r.copy(),p=p.copy(),rr=rr,iterations=iterations)
    if np.sqrt(rr)<=limit:return x,0,payload()
    for _ in range(max_iterations):
        Ap=A@p;curvature=float(p@Ap)
        if curvature<=0 or not np.isfinite(curvature):return x,-1,payload()
        step=rr/curvature;x=x+step*p;r=r-step*Ap
        rr_next=float(r@r);iterations+=1
        p=r+(rr_next/rr)*p;rr=rr_next
        current=payload()
        if callback is not None:callback(x,current)
        if np.sqrt(rr)<=limit:return x,0,current
    return x,max_iterations,payload()
