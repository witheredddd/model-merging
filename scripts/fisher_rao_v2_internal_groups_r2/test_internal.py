import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from fr_math import embedding,sphere_log,sphere_exp,decoder,qf
from structure import orth_dictionary,cosine,choose_structure,overlaps,transport,reachable
from learn import combine,decode,derivatives,penalty,merge_pair,train,select
from ec_stage import atoms_from_tasks,pareto_indices


class Tests(unittest.TestCase):
    def test_full_warm_offset_is_kept(self):
        theta0=torch.zeros(6);warm=torch.arange(6.)
        S=qf(torch.randn(6,3));C=torch.randn(3,8,dtype=torch.double)
        U=qf(torch.randn(8,2,dtype=torch.double));z=torch.zeros(2,dtype=torch.double)
        torch.testing.assert_close(decode(warm,S,C,[U],[z]),warm)
        self.assertFalse(torch.equal(decode(warm,S,C,[U],[z]),theta0))

    def test_joint_derivatives_match_autograd_with_weak_constraints(self):
        torch.manual_seed(1)
        S=qf(torch.randn(7,4));A=torch.randn(10,4,dtype=torch.double);L,_,_=reachable(A);C,_=decoder(A,1e-3)
        groups=[qf(L@torch.randn(4,2,dtype=torch.double)).requires_grad_() for _ in range(2)]
        initial=[qf(L@torch.randn(4,2,dtype=torch.double)) for _ in range(2)]
        z=[torch.randn(2,dtype=torch.double,requires_grad=True) for _ in groups]
        anchor=torch.randn(7);theta=decode(anchor,S,C,groups,z);target=torch.randn(7)
        loss=(theta-target).square().sum()+penalty(groups,initial,.02,.03)
        raw=torch.autograd.grad(loss,groups+z)
        g=2*(theta.detach()-target)
        dz,dU=derivatives(C,S,g,groups,z,L,initial,.02,.03)
        for i,U in enumerate(groups):
            G=L@(L.T@raw[i]);P=G-U@((U.T@G+G.T@U)/2)
            torch.testing.assert_close(dU[i],P,atol=1e-6,rtol=1e-6)
            torch.testing.assert_close(dz[i],raw[len(groups)+i],atol=1e-6,rtol=1e-6)

    def test_zero_coordinate_task_direction_gradient_is_zero(self):
        L=qf(torch.randn(8,4,dtype=torch.double));U=[L[:,:2],L[:,2:]]
        dz,dU=derivatives(torch.randn(3,8,dtype=torch.double),torch.randn(5,3),torch.randn(5),U,
                          [torch.zeros(2,dtype=torch.double) for _ in U],L,U)
        self.assertTrue(all(float(G.norm())==0 for G in dU))
        self.assertTrue(any(float(g.norm())>0 for g in dz))

    def test_dictionary_and_response_linearity(self):
        torch.manual_seed(2);columns=[torch.randn(30) for _ in range(6)]
        S,coeff=orth_dictionary(columns,4);V=torch.stack(columns,1)
        torch.testing.assert_close(S,V.double()@coeff,atol=1e-6,rtol=1e-6,check_dtype=False)
        J=torch.randn(12,30,dtype=torch.double)
        torch.testing.assert_close(J@S.double(),(J@V.double())@coeff,atol=1e-6,rtol=1e-6)

    def test_antipodal_candidate_axes_and_single_task_groups(self):
        a=torch.tensor([1.,-1.],dtype=torch.double).repeat(16)
        b=(torch.tensor([1.,-1.],dtype=torch.double).repeat(16).view(16,2)*torch.tensor([1.,-1.]).repeat(8)[:,None]).flatten()
        R=torch.stack([a,-a,a,-a,b,-b,b,-b],1)
        self.assertAlmostEqual(float(cosine(R)[0,1]),-1.)
        result=choose_structure(R,R,torch.eye(32,dtype=torch.double),[dict(kind='minibatch')]*8,2,bootstrap=20)
        self.assertTrue(next(r['qualified'] for r in result['candidates'] if r['M']==2))

    def test_equivalent_groups_are_detected(self):
        U=qf(torch.randn(12,3,dtype=torch.double));Q=qf(torch.randn(3,3,dtype=torch.double))
        report=overlaps([U,U@Q])[0]
        self.assertGreater(report['projection_i_into_j'],.999999)
        self.assertGreater(report['projection_j_into_i'],.999999)

    def test_unreachable_candidate_groups_are_not_forced_into_a_decoder(self):
        R=torch.zeros(8,4,dtype=torch.double);R[0,:2]=1;R[1,2:]=1
        L=torch.eye(8,dtype=torch.double)[:,:1]
        result=choose_structure(R,R,L,[dict(kind='minibatch')]*4,2,bootstrap=5)
        self.assertFalse(any(row['qualified'] for row in result['candidates']))

    def test_merge_preserves_prediction_direction_and_rank(self):
        L=qf(torch.randn(12,6,dtype=torch.double));groups=[L[:,:2],qf(L@torch.randn(6,2,dtype=torch.double)),L[:,4:]]
        z=[torch.randn(2,dtype=torch.double) for _ in groups]
        Unew,znew=merge_pair(groups,z,0,1,L)
        torch.testing.assert_close(combine(groups,z),combine(Unew,znew),atol=1e-9,rtol=1e-9)
        self.assertEqual(sum(U.shape[1] for U in Unew),6)

    def test_transport_preserves_tangency(self):
        q=embedding(torch.randn(5,3,dtype=torch.double));p=embedding(torch.randn(5,3,dtype=torch.double))
        v=torch.randn(5,3,2,dtype=torch.double);v-=q[:,:,None]*(q[:,:,None]*v).sum(1,keepdim=True)/q.square().sum(1)[:,None,None]
        moved=transport(q.flatten(),p.flatten(),v.reshape(15,2),3).view(5,3,2)
        self.assertLess(float((p[:,:,None]*moved).sum(1).norm()),1e-10)

    def test_ec_refuses_single_task_and_uses_pareto_objectives(self):
        with self.assertRaises(ValueError):atoms_from_tasks({'DTD':{}})
        self.assertEqual(set(pareto_indices([[1,3],[2,2],[3,1],[4,4]])),{0,1,2})

    def test_training_generates_from_warm_anchor(self):
        class Toy:
            def __init__(self):
                torch.manual_seed(3);self.theta0=torch.zeros(5);self.theta_w=torch.ones(5)*.1;self.calls={'training_images':0}
                self.X=torch.randn(3,5);self.V=torch.randn(32,5)
            def predict(self,t,b):return torch.stack([b@t,-b@t],1)
            def scores(self,t,split='Vcal',retain=False):
                z=self.predict(t,self.V);y=(self.V[:,0]<0).long()
                return dict(ce=F.cross_entropy(z,y,reduction='none').detach(),acc=(z.argmax(1)==y).double().detach())
            def train_batch(self,seed,step):return self.V,(self.V[:,0]<0).long(),list(range(32))
            def grad(self,t,b,y):
                t=t.detach().requires_grad_();loss=F.cross_entropy(self.predict(t,b),y)
                return torch.autograd.grad(loss,t)[0],float(loss.detach())
            def exp(self,q,u):return sphere_exp(q.view(3,2),u.view(3,2)).flatten()
            def status(self,*a,**k):pass
        rt=Toy();S=qf(torch.randn(5,3));qw=embedding(rt.predict(rt.theta_w,rt.X)).flatten()
        A=torch.stack([torch.func.jvp(lambda t:embedding(rt.predict(t,rt.X)).flatten(),(rt.theta_w,),(v,))[1] for v in S.T],1)
        L,_,_=reachable(A);C,_=decoder(A,1e-3)
        with tempfile.TemporaryDirectory() as out:
            result=train(rt,S,A,C,L,qw,[L[:,:1],L[:,1:2]],11,Path(out)/'B.pt',max_steps=3,budget=1000)
        state=select(result,1000)
        self.assertTrue(result['weights_generated_every_forward'])
        self.assertEqual(result['anchor'],'theta_w')
        self.assertTrue(all(torch.equal(z,torch.zeros_like(z)) for z in result['states'][0]['z']))
        for U in state['groups']:torch.testing.assert_close(U.T@U,torch.eye(U.shape[1],dtype=U.dtype),atol=1e-9,rtol=1e-9)


if __name__=='__main__':unittest.main()
