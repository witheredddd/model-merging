import unittest
import numpy as np
from math_core import class_weights,gram_modes,metric_stats,dense_restricted

class Tests(unittest.TestCase):
    def test_all_resolved_ranks_and_unseen_demands(self):
        rng=np.random.default_rng(3);G=rng.normal(size=(13,5));values,C,info=gram_modes(G.T@G)
        np.testing.assert_allclose((G@C).T@(G@C),np.eye(5),atol=1e-13)
        self.assertEqual(len(info['proxy_tail_losses']),6)
        for r in range(6):
            Q=G@C[:,:r];res=G-Q@(Q.T@G)
            self.assertAlmostEqual(.5*np.linalg.norm(res)**2,info['proxy_tail_losses'][r],places=11)
        self.assertFalse(np.array_equal(class_weights(47,'sin_4'),class_weights(47,'sin_1')))

    def test_schur_deletion_and_true_complement(self):
        rng=np.random.default_rng(2);J=rng.normal(size=(4,7));H=np.eye(7)+J.T@J;g=rng.normal(size=7)
        v,q=dense_restricted(H,g,np.eye(7));P=np.eye(7);P[:2,:2]=0;w,q0=dense_restricted(H,g,P)
        S=H[:2,:2]-H[:2,2:]@np.linalg.solve(H[2:,2:],H[2:,:2])
        self.assertAlmostEqual(q0-q,.5*v[:2]@S@v[:2],places=11)
        self.assertLess(q0,0);np.testing.assert_allclose(w[:2],0,atol=1e-13)

    def test_output_woodbury_and_error_certificate(self):
        rng=np.random.default_rng(4);J=rng.normal(size=(4,9));g=rng.normal(size=9);P=np.eye(9);P[0,0]=0;beta=2
        u=np.linalg.solve(np.eye(4)+beta*J@P@J.T,J@P@g)
        v=-P@g+beta*P@J.T@u
        expected,_=dense_restricted(np.eye(9)+beta*J.T@J,g,P)
        np.testing.assert_allclose(v,expected,atol=1e-12)
        perturbed=v+.01*P@g;res=P@(g+perturbed+beta*J.T@J@perturbed)
        gap=.5*(perturbed-v)@(np.eye(9)+beta*J.T@J)@(perturbed-v)
        self.assertLessEqual(gap,.5*res@res+1e-13)

    def test_metrics_report_harm_and_discrete_resolution(self):
        labels=np.array([0,1]);base=np.array([[2.,0.],[0.,2.]])
        metrics=metric_stats(base[:,::-1],labels,np.ones(2),base)
        self.assertEqual(metrics['delta_Top1'],-1);self.assertGreater(metrics['delta_CE'],0)
        self.assertEqual(metrics['prediction_flip_rate'],1)

    def test_shared_objective_scaling_respects_parameter_budget(self):
        rng=np.random.default_rng(9);J=rng.normal(size=(3,8));g=rng.normal(size=8);H=np.eye(8)+J.T@J
        for P in (np.eye(8),np.diag([0,0,1,1,1,1,1,1])):
            v,q=dense_restricted(H,g,P);rho=.001;tau=rho/np.linalg.norm(g);delta=tau*v
            self.assertLessEqual(np.linalg.norm(delta),rho+1e-15)
            self.assertAlmostEqual(float(g@delta+.5*delta@(H/tau)@delta),tau*q,places=14)

    def test_freezing_training_span_is_trivial_but_unseen_need_not_be(self):
        rng=np.random.default_rng(23);G=rng.normal(size=(19,6));Q,_=np.linalg.qr(G)
        P=np.eye(19)-Q@Q.T
        self.assertLess(np.linalg.norm(P@G[:,0]),1e-12)
        self.assertGreater(np.linalg.norm(P@rng.normal(size=19)),1.)
        values,C,_=gram_modes(G.T@G/6)
        np.testing.assert_allclose((G@C/np.sqrt(6)).T@(G@C/np.sqrt(6)),np.eye(6),atol=1e-12)

if __name__=='__main__':unittest.main()
