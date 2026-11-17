"""Identical full-space Schur equations, with independent step, materialize and audit."""
import numpy as np
from scipy.sparse.linalg import LinearOperator
from iterative import continuing_cg


def relative(a,b):return float(a/b) if b else (0. if not a else None)


class SchurSession:
    def __init__(self,g,target,weights,rho,y,pd,td):
        if rho<=0 or any(w<0 for w in weights.values()):raise ValueError('Invalid positive cost contract')
        self.g=g;self.target=target;self.rho=rho;self.y=y
        self.tasks=[(t,np.sqrt(w)) for t,w in weights.items() if w>0]
        self.lengths=[g.output_dimensions[t] for t,_ in self.tasks]
        self.n=sum(self.lengths);m=g.output_dimensions[target]
        self.scale=np.r_[1/np.sqrt(np.broadcast_to(pd,(self.n,))),1/np.sqrt(np.broadcast_to(td,(m,)))]
        if not np.all(np.isfinite(self.scale)) or not np.all(self.scale>0):raise ValueError('Invalid numerical scaling')
        self.b=np.r_[np.zeros(self.n),-g.tensor_to_numpy(y)]*self.scale
        self.A=LinearOperator((len(self.b),len(self.b)),matvec=self.apply,dtype=np.float64)

    def transpose(self,z):
        offset=0;v=None
        for (task,weight),length in zip(self.tasks,self.lengths):
            term=weight*self.g.vjp(task,self.g.numpy_to_tensor(z[offset:offset+length]));offset+=length
            v=term if v is None else v+term
        term=self.g.vjp(self.target,self.g.numpy_to_tensor(z[self.n:]))
        return term if v is None else v+term

    def apply(self,x):
        z=self.scale*x;v=self.transpose(z)/self.rho
        p=[w*self.g.tensor_to_numpy(self.g.jvp(t,v)) for t,w in self.tasks]
        joined=np.concatenate(p) if p else np.empty(0)
        return self.scale*np.r_[z[:self.n]+joined,self.g.tensor_to_numpy(self.g.jvp(self.target,v))]

    def step(self,state,rtol,iterations,callback):
        return continuing_cg(self.A,self.b,rtol=rtol,max_iterations=iterations,state=state,callback=callback)

    def materialize(self,state):
        z=self.scale*state['x']
        return -self.transpose(z)/self.rho,self.g.numpy_to_tensor(z[self.n:])

    def full_motion_duals(self,state):
        z=self.scale*state['x'];offset=0;result={t:np.zeros(self.g.output_dimensions[t]) for t in self.g.tasks}
        for (task,weight),length in zip(self.tasks,self.lengths):
            result[task]=-weight*z[offset:offset+length]/self.rho;offset+=length
        result[self.target]=-z[self.n:]/self.rho
        return result

    def audit(self,state):
        g=self.g;v,nu=self.materialize(state)
        response=g.jvp(self.target,v);Av=self.rho*v
        energies={}
        for task,w in self.tasks:
            j=g.jvp(task,v);energies[task]=float(np.linalg.norm(g.tensor_to_numpy(j))**2)
            Av=Av+w*w*g.vjp(task,j)
        jt=g.vjp(self.target,nu)
        primal=float(np.linalg.norm(g.tensor_to_numpy(response-self.y)))
        dual=float(np.linalg.norm(g.tensor_to_numpy(Av+jt)))
        return dict(response_relative_residual=relative(primal,np.linalg.norm(g.tensor_to_numpy(self.y))),
            stationarity_relative_residual=relative(dual,max(np.linalg.norm(g.tensor_to_numpy(Av)),np.linalg.norm(g.tensor_to_numpy(jt)))),
            response_residual_norm=primal,stationarity_residual_norm=dual,
            protected_response_energies=energies,parameter_norm=float(np.linalg.norm(g.tensor_to_numpy(v))),
            objective=.5*float(g.tensor_to_numpy(v)@g.tensor_to_numpy(Av)),
            numerical_system_dimension=len(self.b),parameter_dimension=g.dimension,
            no_target_ridge=True,structural_rank_truncation=False)

    def response_residual_estimate(self,state):
        # Recursive Krylov residual; useful immediately for provisional exploration,
        # but explicitly not a fresh original-system optimality certificate.
        residual=-state['r']/self.scale
        return dict(estimated_target_relative_residual=relative(np.linalg.norm(residual[self.n:]),
            np.linalg.norm(self.g.tensor_to_numpy(self.y))),recursive_residual=True,
            original_stationarity_certified=False)
