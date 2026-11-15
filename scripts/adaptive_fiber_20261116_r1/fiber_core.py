"""Online fiber-conditioned control dictionary construction in an existing expert span.

No post-hoc PCA is used to write model updates. The optional full-rank solver is a
local capacity oracle, used for *candidate birth and reporting*, never for direct
parameter write. Every actual model update is restricted to the learned modes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


def arr(x):
    return np.asarray(x, dtype=np.float64)


def ortho(A, tol=1e-10):
    A=arr(A)
    if A.ndim == 1: A=A[:,None]
    if A.shape[1] == 0: return np.zeros((A.shape[0],0),dtype=np.float64)
    U,s,_=np.linalg.svd(A,full_matrices=False)
    return U[:,s>max(float(s[0])*tol,1e-14)]


@dataclass
class Geometry:
    JD: np.ndarray
    JC: np.ndarray
    hD: np.ndarray
    hC: np.ndarray

    def __post_init__(self):
        self.JD=arr(self.JD)
        self.JC=arr(self.JC)
        self.hD=arr(self.hD)
        self.hC=arr(self.hC)
        assert self.JD.ndim==2 and self.JC.ndim==2
        assert self.JD.shape[1]==self.JC.shape[1]==len(self.hD)==len(self.hC)
        self.KD=self.JD.T@self.JD
        self.KC=self.JC.T@self.JC
        self.scaleD=max(float(np.sqrt(np.trace(self.KD))),1e-30)
        self.scaleC=max(float(np.sqrt(np.trace(self.KC))),1e-30)

    @property
    def dim(self): return len(self.hD)

    def source_reference(self, target: str):
        if hasattr(self,'budget_references'):return self.budget_references[target]
        if target=='Cars':
            h=self.hC; J=self.JD
        else:
            h=self.hD; J=self.JC
        return float(np.linalg.norm(J@(h/max(np.linalg.norm(h),1e-30))))

    def arrays(self,target):
        return ((self.KD,self.KC,self.hD,self.hC) if target=='Cars'
                else (self.KC,self.KD,self.hC,self.hD))


@dataclass(frozen=True)
class Scenario:
    target: str
    tau: float
    radius: float

    def __post_init__(self):
        assert self.target in ('Cars','DTD')
        assert self.tau>0 and self.radius>0


@dataclass
class Mode:
    driver_key: str
    driver: np.ndarray
    supports: list[str] = field(default_factory=list)
    vector: np.ndarray | None = None
    source_task: str = 'DTD'
    map_matrix: list[list[float]] | None = None
    birth_step: int = 0


class ConstrainedSolver:
    """Import existing audited Clarabel solver on A100; scipy fallback for CPU tests."""
    def __init__(self, program_dir=None):
        self._audited=None
        if program_dir:
            import sys
            sys.path.insert(0,str(program_dir))
            from program import solve
            self._audited=solve

    def __call__(self,KD,KC,hD,hC,R,ell):
        KD=arr(KD); KC=arr(KC);hD=arr(hD);hC=arr(hC)
        if hC.size==0:
            return np.zeros(0),dict(passed=True,predicted_Cars_CE=0.,source_response=0.,cost=0.)
        if self._audited:
            v,a=self._audited(KD,KC,hD,hC,R,ell)
            if not a['passed']: return None,a
            return arr(v),a
        from scipy.optimize import minimize
        d=len(hC)
        fun=lambda x: float(hC@x+.5*x@KC@x)
        jac=lambda x: hC+KC@x
        constraints=[
            {'type':'ineq','fun':lambda x: float(R*R-x@x), 'jac':lambda x:-2*x},
            {'type':'ineq','fun':lambda x: float(ell*ell-x@KD@x), 'jac':lambda x:-2*KD@x},
            {'type':'ineq','fun':lambda x:float(-hD@x),'jac':lambda x:-hD},
        ]
        x0=np.zeros(d)
        res=minimize(fun,x0,jac=jac, constraints=constraints,method='SLSQP',
                     options={'maxiter':1000,'ftol':1e-14})
        x=arr(res.x)
        violations=[float(x@x-R*R),float(x@KD@x-ell*ell),float(hD@x)]
        passed=bool(max(violations)<=max(1e-10,R*R*1e-6))
        audit=dict(passed=passed,status=str(res.message),predicted_Cars_CE=fun(x),
                   source_response=float(np.sqrt(max(x@KD@x,0))),cost=float(np.linalg.norm(x)),
                   DTD_CE_slope=float(hD@x),primal_violations=violations)
        return (x if passed else None),audit


def optimize(geom: Geometry,B,scenario: Scenario,solver:ConstrainedSolver):
    B=ortho(B)
    if B.shape[1]==0:
        return np.zeros(geom.dim),dict(passed=True,predicted_Cars_CE=0.,source_response=0.,cost=0.)
    KP,KT,hP,hT=geom.arrays(scenario.target)
    ell=scenario.radius*scenario.tau*geom.source_reference(scenario.target)
    # A controller's gain and cost are computed in orthonormal parameter coordinates.
    x,a=solver(B.T@KP@B,B.T@KT@B,B.T@hP,B.T@hT,scenario.radius,ell)
    if x is None: return None,a
    return B@x,a


def gain(geom,scenario,v):
    if v is None:return -np.inf
    _,KT,_,hT=geom.arrays(scenario.target)
    return float(-hT@v-0.5*v@KT@v)


def mixed_response(geom,v):
    v=arr(v)
    # The relative scales are chosen using the current *whole* geometry, no SVD cutoffs.
    sd=geom.scaleD
    sc=geom.scaleC
    return np.r_[geom.JD@v/sd,geom.JC@v/sc]


def functional_novelty(geom,v,old_modes: np.ndarray):
    """Fraction of candidate joint functional response outside learned modes."""
    w=mixed_response(geom,v)
    if np.linalg.norm(w)<1e-30: return 0.
    if old_modes.size==0 or old_modes.shape[1]==0: return 1.
    Y=np.stack([mixed_response(geom,old_modes[:,j]) for j in range(old_modes.shape[1])],1)
    U=ortho(Y)
    return float(np.linalg.norm(w-U@(U.T@w))/np.linalg.norm(w))


def mode_from_driver(geom: Geometry,driver: np.ndarray,driver_idx, support_idx, protected: str,
                     damping_scale=1e-8):
    """Solve an entire compensator support jointly; returns a unit-cost *one-dimensional* mode."""
    driver=arr(driver).copy()
    idx=np.asarray(list(driver_idx),dtype=np.int64)
    sd=np.asarray(list(support_idx),dtype=np.int64)
    assert not set(idx).intersection(set(sd))
    assert np.linalg.norm(driver)>0
    KP=geom.KD if protected=='DTD' else geom.KC
    v=driver.copy()
    T=np.zeros((len(sd),len(idx)))
    if len(sd):
        A=KP[np.ix_(sd,sd)]
        C=KP[np.ix_(sd,idx)]
        lam=damping_scale*max(np.linalg.norm(A,ord=2),1e-30)
        # Minimum norm if ill-conditioned; numerical regularization does not assert a fiber exists.
        T=-np.linalg.solve(A+lam*np.eye(len(sd)),C)
        v[sd]=T@driver[idx]
    v/=max(np.linalg.norm(v),1e-30)
    return v,T


def mode_basis(modes: list[Mode],dim):
    if not modes:return np.zeros((dim,0))
    return ortho(np.stack([m.vector for m in modes],axis=1))


def rank_curve(geom, modes,scenarios, solver):
    d=geom.dim; rows=[]
    for k in range(len(modes)+1):
        B=mode_basis(modes[:k],d)
        results=[]
        for s in scenarios:
            v,a=optimize(geom,B,s,solver)
            results.append({'target':s.target,'tau':s.tau,'gain':gain(geom,s,v) if v is not None else None,
                            'feasible':a['passed']})
        rows.append({'controller_rank':B.shape[1],'n_modes':k,'values':results})
    return rows


class IncrementalController:
    """Birth from *current* task residual, joint support compensation, and trajectory-informed updates."""
    def __init__(self,groups,solver,capacity_target=0.90,max_modes=16,max_compensators=8,
                 novelty_tol=1e-3, numeric_gain_tol=1e-11):
        self.groups={str(k):list(v) for k,v in groups.items()}
        self.solver=solver
        self.capacity_target=capacity_target
        self.max_modes=max_modes  # compute safety limits, never intrinsic dimensions
        self.max_compensators=max_compensators
        self.novelty_tol=novelty_tol
        self.numeric_gain_tol=numeric_gain_tol
        self.modes: list[Mode]=[]
        self.trajectories=[]
        self.actual_responses=[]
        self.events=[]

    def basis(self,dim):return mode_basis(self.modes,dim)

    def score_scenarios(self,geom,scenarios):
        B=self.basis(geom.dim)
        ranking=[]
        for s in scenarios:
            full,af=optimize(geom,np.eye(geom.dim),s,self.solver)
            part,ap=optimize(geom,B,s,self.solver)
            gf=gain(geom,s,full);gp=gain(geom,s,part)
            if part is None or not ap['passed']:continue
            if full is None or not af['passed'] or gf<=self.numeric_gain_tol:continue
            gap=max(gf-gp,0.)
            ranking.append((gap/max(gf,1e-30),gap,s,full,gf,gp))
        ranking.sort(key=lambda e:(e[0],e[1]),reverse=True)
        return ranking

    def _driver_candidates(self,geom,full):
        B=self.basis(geom.dim)
        # Remove motions already explained by the learned *functional* control directions.
        if B.shape[1]:
            K=geom.KD+geom.KC
            coeff=np.linalg.solve(B.T@K@B+1e-10*np.eye(B.shape[1]),B.T@K@full)
            residual=full-B@coeff
        else:residual=full
        scores=[]
        for key,idx in self.groups.items():
            x=residual[idx];amount=float(np.linalg.norm(x))
            if amount>1e-13:scores.append((amount,key,residual))
        scores.sort(reverse=True)
        return scores

    def _best_with(self,geom,scenario,proposed):
        B=ortho(np.column_stack((self.basis(geom.dim),proposed)))
        v,a=optimize(geom,B,scenario,self.solver)
        return gain(geom,scenario,v),v,a

    def _grow_mode(self,geom,scenario,full,driver_key):
        driver=np.zeros(geom.dim)
        driver_idx=self.groups[driver_key]
        driver[driver_idx]=full[driver_idx]
        if np.linalg.norm(driver)<1e-14:return None,None
        driver/=np.linalg.norm(driver)
        protected='DTD' if scenario.target=='Cars' else 'Cars'
        supports=[];used=set([driver_key]);
        candidate,T=mode_from_driver(geom,driver,driver_idx,[],protected)
        current_score,_,_=self._best_with(geom,scenario,candidate)
        grow_trace=[]
        for _ in range(self.max_compensators):
            best=None
            KP=geom.KD if protected=='DTD' else geom.KC
            residual_inner=KP@candidate
            preselected=[]
            for key,idx in self.groups.items():
                if key in used:continue
                A=KP[np.ix_(idx,idx)]
                si=[j for name in supports for j in self.groups[name]]
                if si:
                    cross=KP[np.ix_(idx,si)]
                    A=A-cross@np.linalg.solve(KP[np.ix_(si,si)]+1e-8*max(np.linalg.norm(KP[np.ix_(si,si)],2),1e-30)*np.eye(len(si)),cross.T)
                b=residual_inner[idx]
                # Residualized support score; approximate when the joint solve is damped.
                reduc=float(b@np.linalg.pinv(A,rcond=1e-9)@b)
                preselected.append((reduc,key))
            preselected.sort(reverse=True)
            for _,key in preselected[:min(24,len(preselected))]:
                trial_supports=supports+[key]
                comp_idx=[j for k in trial_supports for j in self.groups[k]]
                trial,trialT=mode_from_driver(geom,driver,driver_idx,comp_idx,protected)
                score,_,audit=self._best_with(geom,scenario,trial)
                if not audit['passed']:continue
                KP=geom.KD if protected=='DTD' else geom.KC
                source_cost=float(trial@KP@trial)/max(float(trial[driver_idx]@trial[driver_idx]),1e-30)
                raw_cost=float(candidate@KP@candidate)/max(float(candidate[driver_idx]@candidate[driver_idx]),1e-30)
                # Do not accept merely because more parameters are available. Extra support
                # must improve the target gain AND reduce the protected-task response.
                if score>current_score+self.numeric_gain_tol and source_cost<raw_cost*(1-1e-8):
                    record=(score,key,trial,trialT,source_cost)
                    if best is None or score>best[0]:best=record
            if best is None:break
            current_score,key,candidate,T,source_energy=best
            supports.append(key);used.add(key)
            grow_trace.append(dict(group=key,predicted_gain=current_score,protected_energy=source_energy))
        mode=Mode(driver_key=driver_key,driver=driver,supports=supports,vector=candidate,
                  source_task=protected,map_matrix=arr(T).tolist(),birth_step=len(self.trajectories))
        return mode,grow_trace

    def learn(self,geom,scenarios,max_new=2):
        """Birth decisions use current full-vs-structured gaps; only modes can write parameters."""
        added=[]
        for _ in range(max_new):
            if len(self.modes)>=self.max_modes:break
            ranked=self.score_scenarios(geom,scenarios)
            if not ranked:break
            fraction,gap,scenario,full,gf,gp=ranked[0]
            if gp/max(gf,1e-30)>=self.capacity_target or gap<=self.numeric_gain_tol:break
            best=None
            for _,key,residual in self._driver_candidates(geom,full)[:min(16,len(self.groups))]:
                mode,trace=self._grow_mode(geom,scenario,residual,key)
                if mode is None:continue
                old_basis=self.basis(geom.dim)
                nov=functional_novelty(geom,mode.vector,old_basis)
                if nov<self.novelty_tol:continue
                # Historical response trajectories also influence birth: avoid a nominally
                # new parameter mode if it duplicates already *executed* functional motion.
                if self.actual_responses:
                    past=np.stack(self.actual_responses[-12:],1).copy()
                    past[:len(geom.JD)]/=geom.scaleD;past[len(geom.JD):]/=geom.scaleC
                    response=mixed_response(geom,mode.vector);U=ortho(past)
                    hist_nov=float(np.linalg.norm(response-U@(U.T@response))/max(np.linalg.norm(response),1e-30))
                    if hist_nov<self.novelty_tol:continue
                elif self.trajectories:
                    past=np.stack([tr['parameter_update'] for tr in self.trajectories[-12:]],1)
                    hist_nov=functional_novelty(geom,mode.vector,past)
                    if hist_nov<self.novelty_tol:continue
                else:hist_nov=1.
                predicted,_,a=self._best_with(geom,scenario,mode.vector)
                increment=predicted-gp
                if increment>self.numeric_gain_tol and a['passed']:
                    rank_after=mode_basis(self.modes+[mode],geom.dim).shape[1]
                    if rank_after<=old_basis.shape[1]:continue
                    if best is None or increment>best[0]:best=(increment,mode,trace,nov,hist_nov)
            if best is None:break
            increment,mode,trace,nov,hist_nov=best
            self.modes.append(mode)
            ev=dict(event='birth',mode_index=len(self.modes)-1,driver=mode.driver_key,
                    support=mode.supports,nominal_rank=len(self.modes),effective_rank=self.basis(geom.dim).shape[1],
                    target=scenario.target,tau=scenario.tau,capacity_gap_before=gap,
                    predicted_gain_increment=increment,functional_novelty=nov,
                    trajectory_novelty=hist_nov,support_growth=trace)
            self.events.append(ev);added.append(ev)
        return added

    def recondition(self,geom):
        """Refresh learned conditional graph maps from current geometry after a *real* update."""
        updates=[]
        for m in self.modes:
            di=self.groups[m.driver_key]; si=[j for key in m.supports for j in self.groups[key]]
            v,T=mode_from_driver(geom,m.driver,di,si,m.source_task)
            change=float(np.linalg.norm(v-m.vector))
            m.vector=v;m.map_matrix=T.tolist()
            updates.append(dict(driver=m.driver_key,supports=m.supports,delta_direction_norm=change))
        self.events.append(dict(event='recondition',updates=updates))
        return updates

    def accept_trajectory(self,geom,v,scenario,ce_target,ce_protected,actual_response=None):
        v=arr(v)
        resp=mixed_response(geom,v)
        novelty=functional_novelty(geom,v,np.stack([tr['parameter_update'] for tr in self.trajectories],1)
                                      if self.trajectories else np.zeros((geom.dim,0)))
        tr=dict(step=len(self.trajectories),scenario=dict(target=scenario.target,tau=scenario.tau),
                parameter_update=v.tolist(),joint_response_norm=float(np.linalg.norm(resp)),
                trajectory_novelty=novelty,CE_target_change=ce_target,CE_protected_change=ce_protected,
                active_controls=self.basis(geom.dim).shape[1])
        self.trajectories.append(tr)
        if actual_response is not None:
            self.actual_responses.append(arr(actual_response))
            tr['actual_finite_response_norm']=float(np.linalg.norm(actual_response))
            tr['history_scope']='Measured sqrt-probability finite increments in shared output coordinates; no parallel-transport claim'
        return tr
