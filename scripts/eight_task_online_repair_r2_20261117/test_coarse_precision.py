import unittest,tempfile
import numpy as np
from test_controls_math import G,optimum
from provisional import ProvisionalControls
from geometry import EuclideanMetric
from schur_session import SchurSession
from coarse import correct,accurate_products

class PrecisionTests(unittest.TestCase):
    def test_refinement_priority_does_not_wait_for_precise_case(self):
        from provisional import focused_take
        def case(name,error,iterations):
            return dict(id=name,status='provisional',iterations=iterations,visits=0,audit=dict(response_relative_residual=error))
        scheduler=dict(inflight=None,dispatches=0,cases={'precise':case('precise',.002,62),'uncertain':case('uncertain',.13,11)})
        self.assertEqual(focused_take(scheduler,{'precise','uncertain'})['case_id'],'uncertain')
    def test_large_unrelated_column_does_not_hide_resolved_gradient(self):
        D=np.array([[1.e20,0.],[0.,1.]])
        r=np.array([0.,1.]);h,b=accurate_products(D,r)
        old=np.finfo(float).eps*max(D.shape)*np.linalg.norm(D)*np.linalg.norm(r)
        self.assertLess(np.linalg.norm(h),old)
        self.assertGreater(h[1],b[1])
    def test_correction_invariant_under_control_coordinate_rescaling(self):
        g=G();weights={t:1. for t in g.tasks if t!='0'}
        with tempfile.TemporaryDirectory() as root:
            bank=ProvisionalControls(g,root,EuclideanMetric())
            for i,y in enumerate(np.eye(3)):
                v,q,_=optimum(g,y);bank.refresh(v,q,{},str(i))
            session=SchurSession(g,'0',weights,.7,g.numpy_to_tensor([.3,.4,.6]),np.ones(21),np.ones(3))
            _,_,cg=session.step(None,1.e-12,1,None)
            x,event=correct(session,cg,bank);self.assertTrue(event['applied'])
            base_duals=bank.duals;original_gram=bank.gram.copy();scale=np.array([1.e-12,1.e12,1.])
            bank.duals=lambda j:{t:v*scale[j] for t,v in base_duals(j).items()}
            bank.state['gram']=(original_gram*scale[:,None]*scale[None,:]).tolist()
            z,event2=correct(session,cg,bank);self.assertTrue(event2['applied'])
            np.testing.assert_allclose(x['x'],z['x'],atol=1.e-10,rtol=1.e-10)
            np.testing.assert_allclose(z['r'],session.b-session.apply(z['x']),atol=1.e-12)
            self.assertGreater(event2['predicted_schur_energy_decrease'],0)

if __name__=='__main__':unittest.main()
