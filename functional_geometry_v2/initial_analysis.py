"""Complete sample-bootstrap audit of the retained initial space during training."""
import json,torch
from engine import Engine,OUT
from analyze import analyze

if __name__=='__main__':
    E=Engine(micro=8);E.basis_load()
    E.r=json.loads((OUT/'cold_start.json').read_text())['rank']
    E.B[:,E.r:].zero_()
    p=E.matrices('initial_pool_theta0_bootstrap',torch.zeros(E.reserve,device='cuda'))
    r,_,_=analyze(p);print(json.dumps(r,indent=2))
