"""Growing functional controller of full-space motions; no preselected rank/support."""
from pathlib import Path
import numpy as np
import torch


class FunctionalControls:
    def __init__(self,geometry,folder):
        self.g=geometry
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.columns=[]
        self.responses={t:[] for t in geometry.tasks}
        self.parameter_gram=np.empty((0,0))
        self.events=[]

    def _read(self,index):
        x=torch.load(self.columns[index]['file'],map_location='cpu',mmap=True,weights_only=True)['direction']
        return self.g.numpy_to_tensor(x.numpy())

    def add_full_motion(self,v,provenance):
        if v.numel()!=self.g.dimension:raise ValueError('Must be a complete parameter motion')
        magnitude=float(v.norm())
        if magnitude==0:
            self.events.append(dict(event='zero_motion',source=provenance));return
        b=v/magnitude
        n=len(self.columns)
        G=np.zeros((n+1,n+1));G[:n,:n]=self.parameter_gram
        for j in range(n):G[j,n]=G[n,j]=float(torch.dot(self._read(j),b))
        G[n,n]=float(torch.dot(b,b));self.parameter_gram=G
        path=self.folder/f'motion{n:06d}.pt'
        torch.save(dict(direction=b.detach().cpu()),path)
        self.columns.append(dict(file=str(path),full_motion_norm=magnitude,source=provenance))
        for task in self.g.tasks:
            self.responses[task].append(self.g.tensor_to_numpy(self.g.jvp(task,b)))
        self.events.append(dict(event='full_motion_entered_control',index=n,
                                numerical_rank=int(np.linalg.matrix_rank(G)),source=provenance))

    def decode(self,z):
        if len(z)!=len(self.columns):raise ValueError('Control state and coefficients differ')
        result=self.g.numpy_to_tensor(np.zeros(self.g.dimension))
        for j,value in enumerate(z):
            if value:result+=float(value)*self._read(j)
        return result

    def solve_demand(self,target,y,protected_weights,regularization,*,parameter_metric_gram,
                     numerical_tolerance):
        """Control is tested before acquiring any new full-motion teacher for this demand."""
        n=len(self.columns);y=np.asarray(self.g.tensor_to_numpy(y))
        if not n:
            return dict(z=np.empty(0),response_residual=float(np.linalg.norm(y)),
                        n_controls=0,can_meet_numeric_response_tolerance=bool(np.linalg.norm(y)==0))
        A=np.column_stack(self.responses[target])
        H=regularization*np.asarray(parameter_metric_gram)
        for task,weight in protected_weights.items():
            Y=np.column_stack(self.responses[task]);H=H+weight*(Y.T@Y)
        U,s,Vt=np.linalg.svd(A,full_matrices=False)
        # Default machine-precision linear dependence, never an energy/selection cutoff.
        _,_,rank,_=np.linalg.lstsq(A,y,rcond=None)
        C=s[:rank,None]*Vt[:rank]
        rhs=np.r_[np.zeros(n),U[:,:rank].T@y]
        K=np.block([[H,C.T],[C,np.zeros((rank,rank))]])
        solution=np.linalg.lstsq(K,rhs,rcond=None)[0];z=solution[:n]
        residual=float(np.linalg.norm(A@z-y));ynorm=float(np.linalg.norm(y))
        return dict(z=z,n_controls=n,response_residual=residual,
             relative_response_residual=residual/ynorm if ynorm else None,
             can_meet_numeric_response_tolerance=bool(residual<=numerical_tolerance*ynorm),
             protected_energies={t:float(np.linalg.norm(np.column_stack(self.responses[t])@z)**2)
                                   for t in protected_weights},
             objective=float(.5*z@H@z),parameter_norm_squared=float(z@self.parameter_gram@z),
             parameter_gram_eigenvalues=np.linalg.eigvalsh(self.parameter_gram),
             role='Current learned controller, not a full-space optimality claim')

    def validate_unseen(self,*args,**kwargs):
        before=len(self.columns)
        result=self.solve_demand(*args,**kwargs)
        if len(self.columns)!=before:raise RuntimeError('Held-out demand changed the learned structure')
        return dict(result,heldout=True,structure_updated=False)
