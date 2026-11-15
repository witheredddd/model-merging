"""Same-cost, same-source-budget controls; no sparsity or trajectory dependence."""
import sys
import numpy as np
from scipy import sparse
sys.path.insert(0,'/home/shangxueyuan/merging/projected_fiber_1b_deps')
import clarabel


def solve(KD,KC,hD,hC,R,ell,source_factor=None):
    d=len(hC);KD=(KD+KD.T)/2;KC=(KC+KC.T)/2
    if source_factor is None:
        e,V=np.linalg.eigh(KD);scale=max(float(abs(e).max()),1e-30)
        if e.min()<-1e-8*scale:raise ValueError('Indefinite source block')
        L=np.sqrt(np.maximum(e,0))[:,None]*V.T
    else:L=source_factor
    source=np.r_[np.zeros((1,d)),-R*L/max(ell,1e-30)]
    radius=np.r_[np.zeros((1,d)),-np.eye(d)]
    ce_scale=max(R*np.linalg.norm(hD),1e-15)
    A=np.r_[source,radius,(R*hD/ce_scale)[None,:]]
    b=np.r_[1.,np.zeros(L.shape[0]),1.,np.zeros(d),0.]
    P=R*R*KC;q=R*hC
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100
    settings.tol_gap_abs=1e-11;settings.tol_gap_rel=1e-8;settings.tol_feas=1e-9
    cones=[clarabel.SecondOrderConeT(L.shape[0]+1),clarabel.SecondOrderConeT(d+1),clarabel.NonnegativeConeT(1)]
    s=clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix(P)).tocsc(),q,sparse.csc_matrix(A),b,cones,settings).solve()
    z=np.asarray(s.x);x=R*z;dual=np.asarray(s.z);slack=b-A@z
    leakage=float(np.sqrt(max(x@KD@x,0)));cost=float(np.linalg.norm(x));source_ce=float(hD@x)
    primal=max(max(leakage/max(ell,1e-30)-1,0),max(cost/R-1,0),max(source_ce/ce_scale,0))
    stat=float(np.linalg.norm(P@z+q+A.T@dual)/max(np.linalg.norm(q),1e-14))
    gap=abs(float(slack@dual));relative=gap/max(abs(float(q@z)),abs(float(z@P@z)),1e-15)
    audit=dict(status=str(s.status),primal_relative_violation=primal,stationarity_relative_residual=stat,
       complementarity_absolute=gap,complementarity_relative=relative,
       predicted_Cars_CE=float(hC@x+.5*x@KC@x),Cars_CE_slope=float(hC@x),DTD_CE_slope=source_ce,
       source_response=leakage,cost=cost,R=R,ell=ell)
    audit['passed']=bool(str(s.status)=='Solved' and primal<1e-5 and stat<1e-5 and (gap<1e-9 or relative<1e-4))
    return x,audit


def conditional(KD,im,jn):
    nn=KD[np.ix_(jn,jn)];nm=KD[np.ix_(jn,im)];e=np.linalg.eigvalsh((nn+nn.T)/2)
    largest=max(float(e.max()),1e-30)
    damping=0. if e.min()>largest*1e-10 else largest*1e-8
    if e.max()<=1e-25:T=np.zeros_like(nm)
    else:T=-np.linalg.solve(nn+damping*np.eye(len(jn)),nm)
    B=np.r_[np.eye(len(im)),T];W=np.linalg.qr(B,mode='reduced')[0]
    ids=im+jn;pair=KD[np.ix_(ids,ids)]
    schur=KD[np.ix_(im,im)]+KD[np.ix_(im,jn)]@T
    identity_error=float(np.linalg.norm(schur-B.T@pair@B-damping*T.T@T)/max(np.linalg.norm(KD[np.ix_(im,im)]),np.linalg.norm(schur),1e-25))
    return W,dict(T=T.tolist(),damping=damping,regularized_Schur_identity_relative_error=identity_error,
                   Schur_includes_damping_cost=damping>0,raw_source_energy_computed_separately=True)
