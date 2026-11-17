"""Full-space WLS output-dual solver and distinct parameter-support Schur action."""
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator,cg
from bootstrap import CORE
from iterative import continuing_cg


def vnorm(v):return float(torch.linalg.vector_norm(v))


class OutputDualWLS:
    """PDF (14): rho I + sum w J^T J; output reduction does not restrict parameters."""
    def __init__(self,g,weights,rho,desired,diagonals=None,complement=None):
        self.g=g;self.weights={t:float(w) for t,w in weights.items() if w>0};self.tasks=list(self.weights)
        self.rho=float(rho);self.complement=complement
        if rho<=0:raise ValueError('rho must be positive')
        self.lengths=[g.output_dimensions[t] for t in self.tasks]
        self.d=np.concatenate([np.sqrt(self.weights[t])*np.asarray(desired[t]) for t in self.tasks])
        diagonal=np.concatenate([np.ones(g.output_dimensions[t])+(self.weights[t]*diagonals[t]/rho if diagonals is not None else 0.) for t in self.tasks])
        self.scale=1/np.sqrt(diagonal);self.b=self.scale*self.d
        self.A=LinearOperator((len(self.b),len(self.b)),matvec=self.apply,dtype=float)
        self.calls=0

    def project(self,v):
        if self.complement is None:return v
        v=v.clone();v[self.complement]=0.;return v

    def transpose(self,q):
        v=torch.zeros_like(self.g.point);offset=0
        for t,n in zip(self.tasks,self.lengths):
            v+=np.sqrt(self.weights[t])*self.g.vjp(t,self.g.numpy_to_tensor(q[offset:offset+n]));offset+=n
        return self.project(v)

    def forward(self,v):
        return np.concatenate([np.sqrt(self.weights[t])*self.g.tensor_to_numpy(self.g.jvp(t,v)) for t in self.tasks])

    def apply(self,x):
        self.calls+=1;q=self.scale*x
        return self.scale*(q+self.forward(self.transpose(q))/self.rho)

    def advance(self,state,steps,rtol,callback=None):
        return continuing_cg(self.A,self.b,rtol=rtol,max_iterations=steps,state=state,callback=callback)[1:]

    def materialize(self,state):return self.transpose(self.scale*state['x'])/self.rho

    def residual_certificate(self,state):
        # The recursive output residual maps algebraically to a primal residual.
        # Fresh output residual is recomputed at observation, not every CG step.
        r=self.b-self.apply(state['x'])
        physical=self.transpose(r/self.scale)
        n=vnorm(physical)
        return dict(primal_normal_residual_norm=n,parameter_error_bound=n/self.rho,
            joint_energy_error_bound=n/np.sqrt(self.rho),objective_gap_upper_bound=n*n/(2*self.rho),
            output_recursive_fresh_discrepancy=float(np.linalg.norm(r-state['r'])/max(np.linalg.norm(self.b),np.finfo(float).tiny)),
            bound_basis='Positive lower bound rho I; absolute primal normal residual, not relative KKT alone',
            output_dual_schur_distinct_from_parameter_support_schur=True)


class ParameterSupportSchur:
    """T_{N<-M} action on ANY full driver-coordinate vector, not scalar interaction.

    A=F_protected+rho M. Practical identity-metric path uses a complement-masked
    output-dual solve. The general SPD path includes the M_NM cross block.
    Finite iterations are explicitly approximate T actions with residual bounds.
    """
    def __init__(self,g,driver,protection,rho,metric=None,diagonals=None,metric_lower_bound=None):
        self.g=g;self.driver=driver;self.protection=protection;self.rho=rho
        self.metric=metric;self.diagonals=diagonals
        self.metric_lower_bound=1. if metric is None else metric_lower_bound

    def projected(self,v):
        q=v.clone();q[self.driver]=0.;return q

    def cost_action(self,v):
        a=self.rho*(v if self.metric is None else self.metric(v))
        for t,w in self.protection.items():
            if w:a=a+w*self.g.vjp(t,self.g.jvp(t,v))
        return a

    def session(self,driver_vector):
        if self.metric is not None:raise ValueError('Use general_apply for nonidentity metric')
        d={t:self.g.tensor_to_numpy(self.g.jvp(t,driver_vector)) for t in self.protection}
        return OutputDualWLS(self.g,self.protection,self.rho,d,self.diagonals,self.driver)

    def general_apply(self,driver_vector,rtol=1e-10,max_iterations=1000):
        rhs=-self.projected(self.cost_action(driver_vector));g=self.g
        def action(x):
            v=g.numpy_to_tensor(x);n=self.projected(v)
            return g.tensor_to_numpy(self.projected(self.cost_action(n))+v-n)
        A=LinearOperator((g.dimension,g.dimension),matvec=action,dtype=float)
        x,info=cg(A,g.tensor_to_numpy(rhs),rtol=rtol,atol=0.,maxiter=max_iterations)
        compensation=g.numpy_to_tensor(x)
        residual=self.projected(self.cost_action(driver_vector+compensation))
        return compensation,dict(solver_info=int(info),compensation_stationarity_norm=vnorm(residual),
            parameter_error_bound=(vnorm(residual)/(self.rho*self.metric_lower_bound) if self.metric_lower_bound else None),
            declared_metric_lower_bound=self.metric_lower_bound,
            general_metric_cross_block_included=True)

    def lift(self,driver_vector,session,state):
        compensation=-session.materialize(state)
        cert=session.residual_certificate(state)
        cert.update(compensation_stationarity_norm=cert['primal_normal_residual_norm'],
            full_complement_parameter_count=self.g.dimension-(self.driver.stop-self.driver.start),
            driver_parameter_count=self.driver.stop-self.driver.start,
            compensation_uses_protected_tasks_only=True,
            target_curvature_in_compensation=False,
            numerical_T_action_not_exact_inverse_claim=True)
        return driver_vector+compensation,cert

    def apply(self,driver_vector,rtol=1e-10,max_iterations=1000):
        if self.metric is not None:return self.general_apply(driver_vector,rtol,max_iterations)
        session=self.session(driver_vector)
        info,state=session.advance(None,max_iterations,rtol)
        motion,cert=self.lift(driver_vector,session,state);cert['solver_info']=int(info)
        return motion-driver_vector,cert

    def effective_cost_action(self,driver_vector,compensation):
        # On an exact T action this is S_{M|N} x_M. Approximate action residual
        # remains separately measurable on N.
        a=self.cost_action(driver_vector+compensation)
        result=torch.zeros_like(a);result[self.driver]=a[self.driver]
        return result


def ce_cotangent_target(g,task):
    """A normalized functional CE descent demand with J^T y=-grad_CE/scale.

    This is an explicitly new WLS demand, distinct from historical J(-grad/||grad||).
    It need not be in Im(J), so it is never called an exact reachable response.
    """
    logits,labels=g.c.baseline(task,g.probe,g.point);p=logits.softmax(-1)
    n=len(labels);q=torch.zeros_like(p)
    q[torch.arange(n),labels]=-1/(np.sqrt(n)*p[torch.arange(n),labels].sqrt())
    root=p.sqrt();q-=root*(root*q).sum(-1,keepdim=True)
    norm=float(q.norm());y=g.numpy_to_tensor((-q/norm).reshape(-1).numpy())
    gradient=g.ce_gradient(task);lift=g.vjp(task,y)
    discrepancy=vnorm(lift+gradient/norm)/max(vnorm(gradient/norm),np.finfo(float).tiny)
    return y,gradient,dict(source='Normalized negative tangent CE cotangent in square-root probability embedding',
        cotangent_norm=norm,adjoint_CE_identity_relative_error=discrepancy,
        exact_reachability_claim=False,problem='weighted_ls',normalization_explicit=True)
