import unittest
import numpy as np
import torch
from precision import Float64Ops


class ToyContext:
    def __init__(self):
        self.batches={('T',0):[(torch.tensor([[1.,2.],[.3,-.8]],dtype=torch.float64),torch.tensor([0,1]))]}
    def predict(self,p,x,task):return x@p.reshape(2,3)


class TestEmbeddingOperators(unittest.TestCase):
    def test_analytic_embedding_derivative_matches_direct_autodiff(self):
        c=ToyContext();p=torch.arange(6,dtype=torch.float64)/5
        v=torch.tensor([.1,.3,-.2,.7,-.4,.2],dtype=torch.float64)
        ops=Float64Ops(c,p)
        x=c.batches['T',0][0][0]
        phi=lambda q:(2*c.predict(q,x,'T').softmax(-1).sqrt()/np.sqrt(2)).ravel()
        _,expected=torch.func.jvp(phi,(p,),(v,))
        actual=ops.j(v,'T',0)
        torch.testing.assert_close(actual,expected,rtol=1e-13,atol=1e-13)
        w=torch.arange(6,dtype=torch.float64)/7
        leaf=p.clone().requires_grad_(True)
        direct=torch.autograd.grad(phi(leaf),leaf,grad_outputs=w)[0]
        torch.testing.assert_close(ops.jt(w,'T',0),direct,rtol=1e-13,atol=1e-13)


if __name__=='__main__':unittest.main()
