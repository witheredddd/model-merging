import tempfile,unittest
import numpy as np
from bootstrap import CORE
from test_controls_math import G,optimum
from provisional import ProvisionalControls,focused_take
from geometry import EuclideanMetric
from schur_session import SchurSession
from coarse import correct


class TestOnlineProvisional(unittest.TestCase):
    def test_provisional_representation_recovers_without_derived_cache(self):
        import json
        from pathlib import Path
        g=G()
        with tempfile.TemporaryDirectory() as folder:
            bank=ProvisionalControls(g,folder,EuclideanMetric())
            v,q,_=optimum(g,np.array([1.,.2,.3]))
            bank.refresh(v,q,dict(certified=False),'owner')
            original=bank.vector(0).clone();metadata=json.loads(json.dumps(bank.state))
            for path in (Path(folder)/'materialized').glob('*.pt'):path.unlink()
            recovered=ProvisionalControls(g,folder,EuclideanMetric(),metadata)
            np.testing.assert_allclose(recovered.vector(0).numpy(),original.numpy(),atol=1e-12)

    def test_uncertified_refresh_replaces_owner_and_changes_existing_CG(self):
        g=G();weights={t:1. for t in g.tasks if t!='0'}
        with tempfile.TemporaryDirectory() as folder:
            bank=ProvisionalControls(g,folder,EuclideanMetric())
            v,q,_=optimum(g,np.array([1.,0.,0.]))
            bank.refresh(v,q,dict(response_relative_residual=.3,certified=False),'first')
            self.assertEqual(bank.size,1)
            v2,q2,_=optimum(g,np.array([0.,1.,0.]))
            bank.refresh(v2,q2,dict(response_relative_residual=.2,certified=False),'first')
            self.assertEqual(bank.size,1)
            bank.refresh(v,q,dict(certified=False),'second');self.assertEqual(bank.size,2)
            session=SchurSession(g,'0',weights,.7,g.numpy_to_tensor([.3,.4,.6]),np.ones(21),np.ones(3))
            _,_,cg=session.step(None,1e-12,1,None)
            old_x=cg['x'].copy();r0=session.b-session.A@old_x
            energy0=float(.5*old_x@(session.A@old_x)-session.b@old_x)
            updated,event=correct(session,cg,bank)
            self.assertTrue(event['applied'])
            energy1=float(.5*updated['x']@(session.A@updated['x'])-session.b@updated['x'])
            self.assertLess(energy1,energy0)
            np.testing.assert_allclose(updated['r'],session.b-session.A@updated['x'],atol=1e-10)
            old_v,_=session.materialize(cg);new_v,_=session.materialize(updated)
            np.testing.assert_allclose((new_v-old_v).numpy(),bank.decode(event['control_coefficients']).numpy(),atol=1e-10)
            self.assertEqual(updated['iterations'],cg['iterations'])


if __name__=='__main__':unittest.main()
