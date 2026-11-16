import unittest
import numpy as np
from conditional import ConditionalMotion


class DenseReferenceGeometry:
    def __init__(self):
        self.dimension=5
        self.J={'target':np.array([[1.,2.,0.,1.,0.],[0.,1.,1.,0.,2.]]),
                'protected':np.array([[1.,0.,1.,2.,1.]])}
        self.output_dimensions={t:len(J) for t,J in self.J.items()}
    def jvp(self,t,v):return self.J[t]@v
    def vjp(self,t,y):return self.J[t].T@y
    def fisher(self,t,v):return self.vjp(t,self.jvp(t,v))
    def tensor_to_numpy(self,v):return np.asarray(v)
    def numpy_to_tensor(self,v):return np.asarray(v)


class TestFullConditional(unittest.TestCase):
    def test_CE_rowspace_slope_is_preserved_by_exact_response(self):
        from geometry import EuclideanMetric
        g=DenseReferenceGeometry()
        gradient=g.J['target'].T@np.array([.3,-.7])
        u=-gradient/np.linalg.norm(gradient)
        p=ConditionalMotion(g,'target',{'protected':1.},EuclideanMetric(),.4)
        v,_=p.solve_exact_response(g.J['target']@u,relative_tolerance=1e-12,max_iterations=100)
        self.assertAlmostEqual(float(gradient@v),-np.linalg.norm(gradient),places=10)

    def test_unreachable_response_not_certified_from_solver_stop(self):
        from geometry import EuclideanMetric
        g=DenseReferenceGeometry()
        g.J['target']=np.array([[1.,0.,0.,0.,0.],[1.,0.,0.,0.,0.]])
        p=ConditionalMotion(g,'target',{'protected':1.},EuclideanMetric(),.4)
        _,audit=p.solve_exact_response(np.array([1.,-1.]),relative_tolerance=1e-10,max_iterations=30)
        self.assertFalse(audit['target_constraint_numerically_satisfied'])
        self.assertGreater(audit['response_residual_norm'],1.)

    def test_output_schur_matches_full_kkt_without_parameter_restriction(self):
        from geometry import EuclideanMetric
        g=DenseReferenceGeometry();y=np.array([.3,-.4])
        problem=ConditionalMotion(g,'target',{'protected':2.},EuclideanMetric(),.2)
        full,_=problem.solve_exact_response(y,relative_tolerance=1e-12,max_iterations=100)
        schur,a=problem.solve_exact_response_schur_euclidean(y,
                    relative_tolerance=1e-12,max_iterations=100,
                    protected_diagonal_scale=3.,target_diagonal_scale=4.)
        np.testing.assert_allclose(schur,full,atol=1e-10)
        self.assertEqual(a['parameter_dimension'],5)
        self.assertEqual(a['numerical_system_dimension'],3)
        self.assertTrue(a['target_constraint_numerically_satisfied'])
        vector_scaled,_=problem.solve_exact_response_schur_euclidean(y,
            relative_tolerance=1e-12,max_iterations=100,
            protected_diagonal_scale=np.array([3.]),target_diagonal_scale=np.array([4.,7.]))
        np.testing.assert_allclose(vector_scaled,full,atol=1e-10)

    def test_full_coordinate_solution_matches_constrained_dense_optimum(self):
        g=DenseReferenceGeometry()
        metric=np.diag([1.,2.,3.,4.,5.])
        problem=ConditionalMotion(g,'target',{'protected':2.},lambda v:metric@v,.2)
        y=np.array([.3,-.4])
        v,audit=problem.solve_exact_response(y,relative_tolerance=1e-12,max_iterations=100)
        A=2*g.J['protected'].T@g.J['protected']+.2*metric
        W=np.linalg.solve(A,g.J['target'].T)
        reference=W@np.linalg.solve(g.J['target']@W,y)
        np.testing.assert_allclose(v,reference,atol=1e-10)
        np.testing.assert_allclose(g.J['target']@v,y,atol=1e-10)
        self.assertEqual(audit['parameter_dimension'],5)
        multiplied=ConditionalMotion(g,'target',{'protected':20.},lambda v:metric@v,2.)
        same,_=multiplied.solve_exact_response(y,relative_tolerance=1e-12,max_iterations=100)
        np.testing.assert_allclose(same,reference,atol=1e-10)
        scaled,_=problem.solve_exact_response(y,relative_tolerance=1e-12,max_iterations=100,
                                             parameter_scale=3.,response_scale=7.)
        np.testing.assert_allclose(scaled,reference,atol=1e-10)

    def test_least_squares_not_disguised_as_exact_response(self):
        g=DenseReferenceGeometry();problem=ConditionalMotion(g,'target',{'protected':1.},lambda v:v,.5)
        y=np.array([.3,-.4])
        v,a=problem.solve_response_least_squares(y,lambda z:2*z,
                         relative_tolerance=1e-12,max_iterations=100)
        A=g.J['protected'].T@g.J['protected']+.5*np.eye(5)+2*g.J['target'].T@g.J['target']
        np.testing.assert_allclose(v,np.linalg.solve(A,2*g.J['target'].T@y),atol=1e-10)
        self.assertFalse(a['exact_target_matching_claim'])
        self.assertGreater(a['response_residual_norm'],0.)
        np.linalg.cholesky(np.eye(5)) # Identity M is SPD, independent of solver convergence.
        self.assertTrue(np.all(np.linalg.eigvalsh(2*np.eye(2))>=0)) # Test W is PSD.


if __name__=='__main__':unittest.main()
