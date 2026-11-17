"""User-specified M-residual control space; no hard blocks or energy cutoffs.

This module learns residual directions, solves the original restricted conditional
problem, and seeds the unrestricted numerical solve with its actual decoded motion.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import eigh
from checkpoint import atomic_json,digest


def plain(x):
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x


class ConditionalControls:
    def __init__(self,g,folder,metric,metadata=None):
        self.g=g;self.root=Path(folder);self.root.mkdir(parents=True,exist_ok=True)
        self.metric=metric
        self.state=metadata if metadata is not None else dict(modes=[],events=[],gram=[])

    @property
    def size(self):return len(self.state['modes'])

    @property
    def gram(self):return np.asarray(self.state['gram'],dtype=float).reshape(self.size,self.size)

    def vector(self,index):
        item=self.state['modes'][index];path=self.root/item['vector']
        if digest(path)!=item['vector_sha256']:raise ValueError('Control vector checksum mismatch')
        value=torch.load(path,map_location='cpu',mmap=True,weights_only=True)['direction']
        return self.g.numpy_to_tensor(value.numpy())

    def duals(self,index):
        item=self.state['modes'][index];path=self.root/item['duals']
        if digest(path)!=item['duals_sha256']:raise ValueError('Control dual representation differs')
        with np.load(path) as x:return {t:x[t].copy() for t in x.files}

    def decode(self,z):
        result=self.g.numpy_to_tensor(np.zeros(self.g.dimension))
        for j,value in enumerate(z):
            if value:result+=float(value)*self.vector(j)
        return result

    def responses(self,g=None):
        g=g or self.g
        signature=hashlib.sha256(json.dumps(g.output_coordinate_ids,sort_keys=True).encode()).hexdigest()
        values={t:[] for t in g.tasks}
        for j,mode in enumerate(self.state['modes']):
            path=self.root/mode['folder']/('responses_'+signature+'.npz')
            if not path.exists():
                b=self.vector(j)
                arrays={t:g.tensor_to_numpy(g.jvp(t,b)) for t in g.tasks}
                temp=Path(str(path)+'.tmp')
                with temp.open('wb') as f:np.savez(f,**arrays)
                temp.replace(path)
            with np.load(path) as x:
                for t in g.tasks:values[t].append(x[t].copy())
        return {t:np.column_stack(columns) if columns else np.empty((g.output_dimensions[t],0))
                for t,columns in values.items()}

    def solve(self,task,y,weights,rho,rtol,g=None):
        g=g or self.g;Y=self.responses(g);A=Y[task];target=g.tensor_to_numpy(y)
        H=rho*self.gram.copy()
        for t,w in weights.items():H+=w*(Y[t].T@Y[t])
        if not self.size:
            z=np.empty(0);singular=np.empty(0);rank=0;cutoff=0.
        else:
            U,singular,Vt=np.linalg.svd(A,full_matrices=A.shape[0]<self.size)
            cutoff=np.finfo(A.dtype).eps*max(A.shape)*(singular[0] if len(singular) else 0.)
            rank=int(np.count_nonzero(singular>cutoff))
            z=Vt[:rank].T@((U[:,:rank].T@target)/singular[:rank])
            Z=Vt[rank:].T
            if Z.shape[1]:z+=Z@np.linalg.solve(Z.T@H@Z,-Z.T@H@z)
        predicted=A@z;residual=predicted-target;target_norm=float(np.linalg.norm(target))
        relative=float(np.linalg.norm(residual)/target_norm) if target_norm else float(np.linalg.norm(residual))
        feasible=bool(relative<=rtol)
        return dict(z=z,response_relative_residual=relative,feasible_at_requested_numeric_accuracy=feasible,
            response_is_projection_not_exact_constraint=not feasible,
            objective=float(.5*z@H@z),protected_energies={t:float(np.linalg.norm(Y[t]@z)**2) for t in weights},
            parameter_metric_cost=float(z@self.gram@z),coefficient_norm=float(np.linalg.norm(z)),
            constraint_singular_values=singular,machine_rank=rank,machine_cutoff=cutoff,
            machine_rule='eps * max(output_rows,control_columns) * largest_singular_value',
            control_count=self.size,gram_eigenvalues=np.linalg.eigvalsh(self.gram),
            objective_is_constrained_optimum_only_if_response_feasible=feasible)

    def observe_certified(self,v,duals,certificate,provenance):
        if not certificate['certified']:raise ValueError('Uncertified candidates are not conditional-optimum references')
        return self._observe_motion(v,duals,provenance)

    def observe_provisional(self,v,duals,quality,provenance):
        return self._observe_motion(v,duals,dict(provenance,quality=quality,
            numerical_status='provisional',conditional_optimum_claim=False))

    def _observe_motion(self,v,duals,provenance):
        original=v.clone();r=v.clone();coeff=np.zeros(self.size)
        # Two-pass M reorthogonalization implements the stated projection; the
        # second pass corrects floating-point loss, not a new learning objective.
        for _ in range(2):
            if not self.size:break
            Mr=self.metric(r)
            rhs=np.array([float(torch.dot(self.vector(j),Mr)) for j in range(self.size)])
            step=np.linalg.solve(self.gram,rhs);coeff+=step;r-=self.decode(step)
        squared=float(torch.dot(r,self.metric(r)))
        if squared<0:raise ArithmeticError('M quadratic cost became negative')
        norm=np.sqrt(squared)
        R=np.zeros((self.size+1,self.size+1))
        if self.size:
            factor=np.linalg.cholesky(self.gram).T
            R[:-1,:-1]=factor;R[:-1,-1]=factor@coeff
        R[-1,-1]=norm
        singular=np.linalg.svd(R,compute_uv=False)
        cutoff=np.finfo(np.float64).eps*max(self.g.dimension,self.size+1)*(singular[0] if len(singular) else 0.)
        independent=bool(norm>0 and singular[-1]>cutoff)
        event=dict(source=provenance,projection_coefficients=coeff,residual_metric_norm=norm,
            augmented_singular_values=singular,machine_cutoff=cutoff,
            machine_rule='eps * max(full_parameter_dimension,augmented_columns) * largest_singular_value',
            independent_at_machine_precision=independent,intrinsic_dimension_claim=False)
        if not independent:
            self.state['events'].append(plain(event));return event
        b=r/norm;q={t:np.asarray(duals[t]).copy() for t in self.g.tasks}
        for j,value in enumerate(coeff):
            old=self.duals(j)
            for t in q:q[t]-=value*old[t]
        q={t:value/norm for t,value in q.items()}
        serial=self.state.get('next_serial',self.size)
        self.state['next_serial']=serial+1
        folder=Path('modes')/f'{serial:08d}';destination=self.root/folder
        destination.mkdir(parents=True,exist_ok=True)
        torch.save(dict(direction=b.detach().cpu()),destination/'direction.pt')
        with (destination/'duals.npz').open('wb') as f:np.savez(f,**q)
        G=np.zeros((self.size+1,self.size+1));G[:-1,:-1]=self.gram
        Mb=self.metric(b)
        for j in range(self.size):G[j,-1]=G[-1,j]=float(torch.dot(self.vector(j),Mb))
        G[-1,-1]=float(torch.dot(b,Mb))
        parts=[];context=getattr(self.g,'c',None)
        if context is not None:
            offset=0
            for key,size in zip(context.keys,context.sizes):
                parts.append(dict(tensor=key,continuous_euclidean_energy=float(b[offset:offset+size].square().sum())))
                offset+=size
        mode=dict(folder=str(folder),vector=str(folder/'direction.pt'),vector_sha256=digest(destination/'direction.pt'),
            duals=str(folder/'duals.npz'),duals_sha256=digest(destination/'duals.npz'),
            source=provenance,continuous_tensor_participation=parts,hard_membership_threshold=None)
        self.state['modes'].append(mode);self.state['gram']=G.tolist()
        # Verify the dual lift used for actual full-solver warm starts, rather than
        # assuming a coefficient vector encodes the same parameter motion.
        lift=self.g.numpy_to_tensor(np.zeros(self.g.dimension))
        for t in self.g.tasks:lift+=self.g.vjp(t,self.g.numpy_to_tensor(q[t]))
        mode['dual_lift_relative_error']=float((lift-b).norm()/b.norm())
        if not hasattr(self.g,'_control_ce_gradients'):self.g._control_ce_gradients={}
        for task in self.g.tasks:
            if task not in self.g._control_ce_gradients:self.g._control_ce_gradients[task]=self.g.ce_gradient(task)
        mode['CE_signed_slopes']={t:float(torch.dot(self.g._control_ce_gradients[t],b)) for t in self.g.tasks}
        self.responses()
        event['control_index']=self.size-1;self.state['events'].append(plain(event))
        atomic_json(self.root/'CONTROL_STATE.json',self.state)
        return event

    def seed(self,session,z,rtol):
        if session.g.output_coordinate_ids!=self.g.output_coordinate_ids:
            return None,dict(used=False,reason='Different probe/window Jacobian; old dual coordinates are not transported')
        if not getattr(self.metric,'is_identity',False):
            return None,dict(used=False,reason='Dual Schur seed requires the explicit Euclidean baseline')
        if not self.size:return None,dict(used=False,reason='No learned controls yet')
        # A numerical representation failure must not remove a learned direction.
        if any(m['dual_lift_relative_error']>rtol for m in self.state['modes']):
            return None,dict(used=False,reason='Dual lift not accurate enough for a reliable numerical seed')
        q={t:np.zeros(self.g.output_dimensions[t]) for t in self.g.tasks}
        for j,value in enumerate(z):
            if value:
                item=self.duals(j)
                for t in q:q[t]+=value*item[t]
        pieces=[-session.rho*q[t]/w for t,w in session.tasks]
        pieces.append(-session.rho*q[session.target]);x=np.concatenate(pieces)/session.scale
        residual=session.b-session.apply(x)
        cg=dict(x=x,r=residual,p=residual.copy(),rr=float(residual@residual),iterations=0)
        return cg,dict(used=True,control_count=self.size,coefficients=np.asarray(z),
            role='Decoded restricted control motion initializes the unrestricted full-space solve',
            parameter_search_space_restricted=False)

    def spectrum(self,task,weights,rho,g=None):
        g=g or self.g
        Y=self.responses(g);FT=Y[task].T@Y[task];FP=np.zeros_like(FT)
        for t,w in weights.items():FP+=w*(Y[t].T@Y[t])
        denominator=FP+rho*self.gram
        values,axes=eigh(FT,denominator) if self.size else (np.empty(0),np.empty((0,0)))
        if g.output_coordinate_ids==self.g.output_coordinate_ids:
            slopes=np.array([m['CE_signed_slopes'][task] for m in self.state['modes']])
        else:
            gradient=g.ce_gradient(task)
            slopes=np.array([float(torch.dot(gradient,self.vector(j))) for j in range(self.size)])
        return dict(all_generalized_eigenvalues=values,control_coordinate_axes=axes,
            CE_signed_derivatives=slopes@axes,target_effective_Fisher=FT,
            protected_effective_Fisher=FP,parameter_metric_pullback=self.gram,
            spectral_cutoff=None,all_modes_reported=True)

    def compare(self,result,vstar,weights,rho,certificate,g=None):
        g=g or self.g
        if not result['feasible_at_requested_numeric_accuracy'] or not certificate['certified']:
            return dict(comparable_as_exact_constrained_optima=False,cost_gap=None,
                reason='Restricted response infeasible or full reference uncertified')
        vhat=self.decode(result['z']);difference=vhat-vstar
        cost=float(rho*torch.dot(difference,self.metric(difference)))
        for t,w in weights.items():cost+=w*float(g.jvp(t,difference).square().sum())
        gap=result['objective']-certificate['objective']
        return dict(comparable_as_exact_constrained_optima=True,cost_gap=gap,
            half_A_distance=.5*cost,identity_discrepancy=gap-.5*cost,
            exact_identity_requires_exact_KKT_and_response_constraints=True,
            full_reference_numerical_audit=certificate)
