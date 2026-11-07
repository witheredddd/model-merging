"""Independent real-model checks of reconstruction, signed CE and Fisher."""
import json,time
import numpy as np
import torch
from engine import Engine,OUT,TASKS,write

def main():
    E=Engine(micro=8);E.basis_load();zero=torch.zeros(E.reserve,device='cuda');checks=[]
    orth=float((E.u.gram(E.B[:,:E.r])-torch.eye(E.r,device='cuda')).abs().max())
    assert orth<2e-5,orth
    for k in [1,4]:
        x,y=E.batch(k,E.probe_ids[k][:2]);G=E.fisher(k,zero,x)
        E.place(zero);g,loss=E.gradient(k,x,y);signed=float(E.B[:,0]@g);del g
        rows=[]
        for eps in [.003,.001]:
            z=zero.clone();z[0]=eps
            with torch.no_grad():
                E.place(zero);a=E.models[k](x.cuda()).double();E.place(z);b=E.models[k](x.cuda()).double()
                pa=a.softmax(-1);kl=float((pa*(a.log_softmax(-1)-b.log_softmax(-1))).sum(-1).mean())
            plus=E.ce(k,z,x,y);z[0]=-eps;minus=E.ce(k,z,x,y)
            fd=(plus-minus)/(2*eps);expected=.5*eps**2*G[0,0]
            rows.append(dict(epsilon=eps,predictive_kl=kl,Fisher_second_order_prediction=float(expected),
                KL_relative_error=abs(kl-expected)/max(abs(expected),1e-12),signed_CE_derivative=signed,
                finite_difference_CE_derivative=fd,CE_derivative_relative_error=abs(fd-signed)/max(abs(signed),1e-4)))
        assert min(row['KL_relative_error'] for row in rows)<.1,rows
        assert min(row['CE_derivative_relative_error'] for row in rows)<.1,rows
        checks.append(dict(task=TASKS[k],checks=rows,min_fisher_eigenvalue=float(np.linalg.eigvalsh(G)[0])))
    result=dict(time=time.time(),passed=True,orthogonality_max_error=orth,real_CLIP_checks=checks)
    write(OUT/'implementation_checks.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
