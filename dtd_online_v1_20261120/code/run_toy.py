"""A fully runnable synthetic end-to-end demonstration (CPU), not DTD evidence."""
from __future__ import annotations
import argparse,json
import torch
import torch.nn.functional as F
from controller import OnlineController,AdamWProposal,Config,finite_cap
from core import norm


class ToyGeometry:
    def __init__(self,point,X):
        self.point=point.detach().clone()
        self.X=X
    def phi(self,v):
        z=self.X@v.reshape(self.X.shape[1],-1)
        return (2*z.softmax(-1).sqrt()/(len(self.X)**.5)).reshape(-1)
    def jvp(self,task,v):
        return torch.func.jvp(self.phi,(self.point,),(v,))[1].detach()
    def vjp(self,task,y):
        _,pull=torch.func.vjp(self.phi,self.point)
        return pull(y)[0].detach()


def ce(x,y,point):
    return F.cross_entropy(x@point.reshape(x.shape[1],-1),y)


def run(arm='online',steps=6,seed=29):
    torch.manual_seed(seed)
    features,classes=5,3
    X=torch.randn(100,features,dtype=torch.float64)
    truth=torch.randn(features,classes,dtype=torch.float64)
    logits=X@truth
    labels=logits.argmax(-1)
    p=torch.zeros(features*classes,dtype=torch.float64)
    active=torch.ones_like(p,dtype=torch.bool)
    source=AdamWProposal(p,active,lr=.15)
    ctrl=OnlineController(active,Config(radius=.4,inverse_ridge=.05,
                    inverse_tol=1e-8,inverse_iter=80,max_modes=4,
                    backtracks=4,minimum_error_drop=.01,
                    frozen_after=1 if arm=='frozen' else None))
    logs=[]
    for t in range(steps):
        i=(t*13)%60
        xb,yb=X[i:i+12],labels[i:i+12]
        leaf=p.detach().requires_grad_(True)
        g=torch.autograd.grad(ce(xb,yb,leaf),leaf)[0].detach()
        u=source.propose(p,g)
        base_p=p.clone()
        def guard(v):return dict(ce=float(ce(X[60:80],labels[60:80],base_p+v)))
        if arm=='reference':
            v=finite_cap(u,.4)
            if guard(v)['ce'] >=guard(torch.zeros_like(v))['ce']:v=torch.zeros_like(v)
            report=dict(rank=None,actual_outside_mode_span=None,birth=None)
        else:
            geom=ToyGeometry(p,X[:60])
            d=ctrl.propose(geom,u,guard,step_id=t)
            v=d.delta;report=d.record
        p+=v
        logs.append(dict(step=t,birth=bool(report.get('birth',{}).get('admitted',False))
                         if report.get('birth') else False,
                         rank=report['rank'],write_norm=norm(v),
                         development_ce=guard(torch.zeros_like(v))['ce'],
                         projection_error=report.get('actual_outside_mode_span')))
    result=dict(arm=arm,initial_test_ce=float(ce(X[80:],labels[80:],torch.zeros_like(p))),
                final_test_ce=float(ce(X[80:],labels[80:],p)),
                rank=len(ctrl.modes) if arm!='reference' else None,steps=logs)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['online','frozen','reference'],default='online')
    ap.add_argument('--steps',type=int,default=6);a=ap.parse_args()
    print(json.dumps(run(a.arm,a.steps),indent=2))
