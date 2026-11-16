import tempfile,unittest
import numpy as np
import torch
from controls import FunctionalControls


class Geometry:
    dimension=3;tasks=('T','P')
    J={'T':np.array([[1.,0.,0.],[0.,1.,0.]]),'P':np.array([[0.,0.,1.]])}
    def numpy_to_tensor(self,x):return torch.as_tensor(x,dtype=torch.float64)
    def tensor_to_numpy(self,x):return x.detach().numpy()
    def jvp(self,t,x):return self.numpy_to_tensor(self.J[t]@self.tensor_to_numpy(x))


class TestGrowingControls(unittest.TestCase):
    def test_new_demand_uses_learned_structure_without_PCA_or_test_birth(self):
        g=Geometry()
        with tempfile.TemporaryDirectory() as folder:
            bank=FunctionalControls(g,folder)
            bank.add_full_motion(torch.tensor([1.,0.,0.],dtype=torch.float64),{'demand':'first'})
            before=bank.validate_unseen('T',g.numpy_to_tensor([.3,.4]),{'P':1.},.1,
                    parameter_metric_gram=bank.parameter_gram,numerical_tolerance=1e-10)
            self.assertFalse(before['can_meet_numeric_response_tolerance'])
            bank.add_full_motion(torch.tensor([0.,1.,0.],dtype=torch.float64),{'demand':'second_train'})
            result=bank.validate_unseen('T',g.numpy_to_tensor([.7,-.2]),{'P':1.},.1,
                    parameter_metric_gram=bank.parameter_gram,numerical_tolerance=1e-10)
            self.assertTrue(result['can_meet_numeric_response_tolerance'])
            np.testing.assert_allclose(bank.decode(result['z']).numpy(),[.7,-.2,0.],atol=1e-12)
            self.assertEqual(len(bank.columns),2)
            self.assertFalse(result['structure_updated'])


if __name__=='__main__':unittest.main()
