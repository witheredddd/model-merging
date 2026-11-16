"""Matrix-free full-domain conditional motion. Numerical limits are explicit inputs."""
import numpy as np
from scipy.sparse.linalg import LinearOperator,minres,cg


def norm(x):return float(np.linalg.norm(x))


def relative(numerator,denominator):
    return float(numerator/denominator) if denominator else (0. if numerator==0 else None)


class ConditionalMotion:
    def __init__(self,geometry,target,protected_weights,parameter_metric,regularization):
        if regularization<=0:raise ValueError('The stated positive parameter cost is required')
        if any(w<0 for w in protected_weights.values()):raise ValueError('Negative protection weight')
        self.geometry=geometry
        self.target=target
        self.protected=dict(protected_weights)
        self.metric=parameter_metric
        self.regularization=float(regularization)
        self.dimension=geometry.dimension

    def protected_cost(self,v):
        result=self.regularization*self.metric(v)
        for task,weight in self.protected.items():
            if weight:result=result+weight*self.geometry.fisher(task,v)
        return result

    def solve_exact_response(self,y,*,relative_tolerance,max_iterations,callback=None,
                             parameter_scale=1.,response_scale=1.):
        """[M_P J_T^T; J_T 0][v,nu]=[0,y]. No output ridge or hidden projection."""
        g=self.geometry
        target=g.tensor_to_numpy(y)
        P=self.dimension
        m=g.output_dimensions[self.target]
        if target.shape!=(m,):raise ValueError('Target functional response shape differs')
        def apply(x):
            v=g.numpy_to_tensor(x[:P]);nu=g.numpy_to_tensor(x[P:])
            first=self.protected_cost(v)/parameter_scale+g.vjp(self.target,nu)/response_scale
            second=g.jvp(self.target,v)/response_scale
            return np.r_[g.tensor_to_numpy(first),g.tensor_to_numpy(second)]
        operator=LinearOperator((P+m,P+m),matvec=apply,dtype=np.float64)
        rhs=np.r_[np.zeros(P),target/response_scale]
        solution,info=minres(operator,rhs,rtol=relative_tolerance,
                             maxiter=max_iterations,callback=callback)
        v=g.numpy_to_tensor(solution[:P]);nu=g.numpy_to_tensor(solution[P:])*parameter_scale/response_scale
        response=g.jvp(self.target,v)
        cost_v=self.protected_cost(v)
        jt_nu=g.vjp(self.target,nu)
        residual=g.tensor_to_numpy(response)-target
        stationarity=g.tensor_to_numpy(cost_v+jt_nu)
        return v,dict(solver='MINRES on full KKT',solver_info=int(info),
                      solver_reported_convergence=info==0,
                      parameter_dimension=P,output_dimension=m,
                      response_residual_norm=norm(residual),
                      response_relative_residual=relative(norm(residual),norm(target)),
                      stationarity_residual_norm=norm(stationarity),
                      stationarity_relative_residual=relative(norm(stationarity),
                          max(norm(g.tensor_to_numpy(cost_v)),norm(g.tensor_to_numpy(jt_nu)))),
                      achieved_response_norm=norm(g.tensor_to_numpy(response)),
                      target_response_norm=norm(target),parameter_norm=norm(solution[:P]),
                      regularization=self.regularization,protection_weights=self.protected,
                      numerical_block_scaling=dict(parameter=parameter_scale,response=response_scale),
                      metric_certificate=getattr(self.metric,'certificate',
                          'Identity SPD' if getattr(self.metric,'is_identity',False) else 'Caller-supplied SPD contract; not globally certified by finite probes'),
                      primal_tolerance_requested=relative_tolerance,
                      target_constraint_numerically_satisfied=bool(
                          relative(norm(residual),norm(target)) is not None and
                          relative(norm(residual),norm(target))<=relative_tolerance),
                      single_protected_tradeoff_ratio=(self.regularization/next(iter(self.protected.values()))
                          if len(self.protected)==1 and next(iter(self.protected.values()))>0 else None),
                      objective=.5*float(solution[:P]@g.tensor_to_numpy(cost_v)),
                      reachability='Residual is reported; solver exhaustion is not proof of unreachability',
                      search_domain='All common visual parameters, not an expert dictionary')

    def solve_exact_response_schur_euclidean(self,y,*,relative_tolerance,max_iterations,
                    protected_diagonal_scale,target_diagonal_scale,callback=None):
        """Exact elimination of rho I; no target ridge, support restriction or rank cutoff."""
        if not getattr(self.metric,'is_identity',False):
            raise ValueError('This Schur identity requires explicit M=I; use full KKT for general M')
        g=self.geometry;rho=self.regularization
        tasks=[(t,np.sqrt(w)) for t,w in self.protected.items() if w>0]
        lengths=[g.output_dimensions[t] for t,_ in tasks]
        protected_size=sum(lengths);m=g.output_dimensions[self.target]
        scales=np.r_[1/np.sqrt(np.broadcast_to(protected_diagonal_scale,(protected_size,))),
                     1/np.sqrt(np.broadcast_to(target_diagonal_scale,(m,)))]
        def transpose(z):
            offset=0;value=None
            for (task,weight),length in zip(tasks,lengths):
                term=weight*g.vjp(task,g.numpy_to_tensor(z[offset:offset+length]));offset+=length
                value=term if value is None else value+term
            term=g.vjp(self.target,g.numpy_to_tensor(z[protected_size:]))
            return term if value is None else value+term
        def apply(x):
            z=scales*x
            v=transpose(z)/rho
            pieces=[weight*g.tensor_to_numpy(g.jvp(task,v)) for task,weight in tasks]
            p=np.concatenate(pieces) if pieces else np.empty(0)
            return scales*np.r_[z[:protected_size]+p,g.tensor_to_numpy(g.jvp(self.target,v))]
        rhs=np.r_[np.zeros(protected_size),-g.tensor_to_numpy(y)]*scales
        A=LinearOperator((len(rhs),len(rhs)),matvec=apply,dtype=np.float64)
        x,info=cg(A,rhs,rtol=relative_tolerance,atol=0.,maxiter=max_iterations,callback=callback)
        z=scales*x;v=-transpose(z)/rho
        nu=g.numpy_to_tensor(z[protected_size:])
        Av=self.protected_cost(v);jt=g.vjp(self.target,nu)
        primal=g.tensor_to_numpy(g.jvp(self.target,v)-y)
        dual=g.tensor_to_numpy(Av+jt)
        return v,dict(solver='CG exact output Schur for explicit Euclidean M',
            solver_info=int(info),parameter_dimension=self.dimension,
            numerical_system_dimension=len(rhs),response_relative_residual=relative(norm(primal),norm(g.tensor_to_numpy(y))),
            response_residual_norm=norm(primal),stationarity_residual_norm=norm(dual),
            stationarity_relative_residual=relative(norm(dual),max(norm(g.tensor_to_numpy(Av)),norm(g.tensor_to_numpy(jt)))),
            target_constraint_numerically_satisfied=bool(relative(norm(primal),norm(g.tensor_to_numpy(y))) is not None and
                relative(norm(primal),norm(g.tensor_to_numpy(y)))<=relative_tolerance),
            parameter_norm=norm(g.tensor_to_numpy(v)),metric_certificate='Identity SPD',
            target_ridge=0.,structural_rank_truncation=False,
            note='Smaller output linear system represents a full-P parameter solution; no fixed parameter dictionary')

    def solve_response_least_squares(self,y,output_weight,*,relative_tolerance,
                                     max_iterations,callback=None):
        """PDF (14)-(15); explicit caller choice, never automatic KKT fallback."""
        g=self.geometry
        rhs=g.vjp(self.target,output_weight(y))
        def apply(x):
            v=g.numpy_to_tensor(x)
            z=self.protected_cost(v)+g.vjp(self.target,output_weight(g.jvp(self.target,v)))
            return g.tensor_to_numpy(z)
        A=LinearOperator((self.dimension,self.dimension),matvec=apply,dtype=np.float64)
        result,info=cg(A,g.tensor_to_numpy(rhs),rtol=relative_tolerance,atol=0.,
                       maxiter=max_iterations,callback=callback)
        residual=apply(result)-g.tensor_to_numpy(rhs)
        target_residual=g.tensor_to_numpy(g.jvp(self.target,g.numpy_to_tensor(result))-y)
        return g.numpy_to_tensor(result),dict(solver='CG full-space weighted response fitting',
                    solver_info=int(info),parameter_dimension=self.dimension,
                    equation_residual_norm=norm(residual),
                    equation_relative_residual=relative(norm(residual),norm(g.tensor_to_numpy(rhs))),
                    response_residual_norm=norm(target_residual),
                    response_relative_residual=relative(norm(target_residual),norm(g.tensor_to_numpy(y))),
                    output_metric_status=getattr(output_weight,'certificate','Caller-supplied PSD contract'),
                    exact_target_matching_claim=False)

