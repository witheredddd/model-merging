import sys,pathlib,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from controller import OnlineController,Mode,Config
from core import protect_graph_lift,norm

class ToyGeo:
    def __init__(self):
        torch.manual_seed(17)
        self.Jd=torch.randn(6,9,dtype=torch.float64)
        self.Jc=torch.randn(5,9,dtype=torch.float64)
    def jvp(self,task,v):return (self.Jd if task=='DTD' else self.Jc)@v
    def vjp(self,task,y):return (self.Jd if task=='DTD' else self.Jc).T@y

class ProtectedGraphTests(unittest.TestCase):
    def test_protection_refresh_preserves_raw_driver_graph(self):
        geo=ToyGeo()
        mask=torch.ones(9,dtype=torch.bool)
        controller=OnlineController(mask,Config(radius=.3,graph_ridge=.4,
                                     protection_weight=3.,graph_iter=80),protect='Cars')
        driver=torch.randn(9,dtype=torch.float64)
        lift=protect_graph_lift(geo,driver,controller.project,protect='Cars',
                                ridge=.4,protection_weight=3.,maxiter=80,tol=1e-9)
        self.assertTrue(lift.solver.converged)
        controller.modes.append(Mode(lift.mode,lift.driver,0,'Cars',0.))
        before=controller.modes[0].direction.clone()
        controller._refresh_protected(geo)
        self.assertLess(norm(controller.modes[0].direction-before),1e-2)
        self.assertLess(norm(controller.modes[0].direction-lift.mode),1e-3)
        self.assertLess(norm(geo.jvp('Cars',controller.modes[0].direction)),
                        norm(geo.jvp('Cars',controller.modes[0].driver)))
    def test_protect_task_rejects_missing_protect_weight(self):
        with self.assertRaises(ValueError):
            OnlineController(torch.ones(9,dtype=torch.bool),Config(protection_weight=0),protect='Cars')

if __name__=='__main__':unittest.main()
