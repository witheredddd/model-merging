import unittest
import numpy as np
from demands import product_sphere_log


class TestDemandCoordinates(unittest.TestCase):
    def test_family_rejects_anchor_and_probe_mismatch(self):
        import torch
        from types import SimpleNamespace
        from demands import DemandFamily,FunctionalDemand
        g=SimpleNamespace(point=torch.tensor([1.,2.]),probe=0,
                          output_dimensions={'Cars':4},output_coordinate_ids={'Cars':'test_order'},
                          tensor_to_numpy=lambda x:x.numpy())
        f=DemandFamily(g)
        good=FunctionalDemand('x','Cars',f.anchor,0,torch.ones(4),'test',True,
                              {'output_coordinates_id':'test_order'})
        f.append(good)
        view=f.analysis_view(good)
        np.testing.assert_array_equal(good.response.numpy(),np.ones(4))
        self.assertEqual(view['raw_response_norm'],2.)
        with self.assertRaises(ValueError):
            f.append(FunctionalDemand('old','Cars','old_anchor',0,torch.ones(4),'test',False))
        with self.assertRaises(ValueError):
            f.append(FunctionalDemand('probe','Cars',f.anchor,1,torch.ones(4),'test',False))
        with self.assertRaises(ValueError):
            f.append(FunctionalDemand('wrong_head','Cars',f.anchor,0,torch.ones(4),'test',False,
                                      {'output_coordinates_id':'other_head'}))
        g.point=torch.tensor([3.,4.])
        with self.assertRaises(ValueError):f.append(good)

    def test_samplewise_log_is_tangent_and_recovers_endpoint(self):
        p=np.array([[.8,.2],[.3,.7]])
        q=np.array([[.4,.6],[.6,.4]])
        r=2/np.sqrt(len(p));x=r*np.sqrt(p);z=r*np.sqrt(q)
        y=product_sphere_log(x,z)
        np.testing.assert_allclose(np.sum(x*y,axis=1),0,atol=1e-14)
        length=np.linalg.norm(y,axis=1,keepdims=True)
        reconstructed=np.cos(length/r)*x+r*np.sin(length/r)*y/length
        np.testing.assert_allclose(reconstructed,z,atol=1e-14)

    def test_zero_log_and_sample_axis_not_one_batch_sphere(self):
        x=np.array([[1.,0.],[0.,1.]])*np.sqrt(2)
        np.testing.assert_array_equal(product_sphere_log(x,x),np.zeros_like(x))
        q=x[::-1].copy()
        y=product_sphere_log(x,q)
        np.testing.assert_allclose(np.linalg.norm(y,axis=1),np.sqrt(2)*np.pi/2)

    def test_mean_metric_invariant_to_repeated_samples(self):
        p=np.array([[.8,.2],[.3,.7]]);q=np.array([[.4,.6],[.6,.4]])
        def embedded(a):return 2*np.sqrt(a)/np.sqrt(len(a))
        first=product_sphere_log(embedded(p),embedded(q))
        repeated=product_sphere_log(embedded(np.tile(p,(2,1))),embedded(np.tile(q,(2,1))))
        self.assertAlmostEqual(float(np.sum(first**2)),float(np.sum(repeated**2)),places=13)


if __name__=='__main__':unittest.main()
