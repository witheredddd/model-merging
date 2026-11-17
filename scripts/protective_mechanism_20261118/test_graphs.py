import unittest
import numpy as np
import torch
from bootstrap import R3
from graphs import conditional_graph,mechanism_report,metric_split,ArbitrarySupportSchur,joint_mode_curve
from test_research import Geometry


class GraphTests(unittest.TestCase):
    def test_compensation_ratio_keeps_target_and_reduces_protected_response(self):
        Y={'T':np.array([[1.,0.]]),'P':np.array([[1.,1.]])};G=np.eye(2)
        r=mechanism_report(Y,G,'T',np.ones(1),{'P':1.},.01,[0])
        self.assertAlmostEqual(r['graph']['T'][0,0],-1/1.01)
        self.assertLess(r['fixed_driver_compensation'][0]['compensated_joint_cost'],r['fixed_driver_compensation'][0]['raw_joint_cost'])
        raw=r['fixed_target_comparisons']['uncompensated_driver'];joint=r['fixed_target_comparisons']['selective_modes']
        self.assertTrue(joint['reachable']);self.assertLess(joint['protected_energies']['P'],raw['protected_energies']['P']*.001)
        self.assertFalse(r['reoptimized_removals'][0]['reoptimized_remaining']['reachable'])
        self.assertGreater(r['reoptimized_removals'][1]['reoptimized_remaining']['objective'],joint['objective'])

    def test_multidimensional_graph_spectrum_has_no_single_direction_cap(self):
        rng=np.random.default_rng(7);Y={'T':rng.normal(size=(3,7)),'P':rng.normal(size=(4,7))}
        graph=conditional_graph(Y,np.eye(7),'T',{'P':1.},.2,[0,2,5])
        self.assertEqual(graph['joint_modes'].shape,(7,3))
        self.assertLess(graph['compensation_stationarity_norm'],1e-12)
        np.testing.assert_allclose(graph['joint_modes'].T@(Y['P'].T@Y['P']+.2*np.eye(7))@graph['joint_modes'],np.eye(3),atol=1e-12)

    def test_arbitrary_support_full_complement_matches_dense_solution(self):
        g=Geometry();M=np.array([0,3,8]);N=np.array([i for i in range(13) if i not in M]);w={t:1. for t in g.tasks if t!='0'}
        A=.9*np.eye(13)+sum(g.J[t].T@g.J[t] for t in w)
        v=torch.zeros(13,dtype=torch.float64);v[M]=torch.tensor([.2,-.4,.3],dtype=torch.float64)
        op=ArbitrarySupportSchur(g,M,w,.9);c,cert=op.apply(v,rtol=1e-12,max_iterations=200)
        np.testing.assert_allclose(c[N],-np.linalg.solve(A[np.ix_(N,N)],A[np.ix_(N,M)]@v[M].numpy()),atol=1e-10)
        self.assertEqual(cert['driver_parameter_count'],3)

    def test_functional_metric_split_reports_cross_terms_not_euclidean_claim(self):
        Y={'T':np.eye(2),'P':np.array([[1.,2.]])};G=np.eye(2);d=np.array([.4,-.1]);B=np.array([[1.],[.3]])
        split=metric_split(Y,G,'T',{t:a@d for t,a in Y.items()},d@d,d,B,{'P':1.},.2)
        self.assertLess(abs(split['protection_metric_cross_term']),1e-12)
        self.assertFalse(split['EC_executed'])

    def test_joint_curve_keeps_every_prefix_without_choosing_a_dimension(self):
        Y={'T':np.eye(3),'P':np.array([[1.,1.,0.]])};G=np.eye(3)
        curve=joint_mode_curve(Y,G,'T',np.ones(3),{'P':1.},.1,np.eye(3))
        self.assertEqual(len(curve['complete_prefix_curve']),3)
        self.assertIsNone(curve['automatically_selected_dimension'])
        obj=[r['objective'] for r in curve['complete_prefix_curve']]
        self.assertTrue(all(a>=b-1e-12 for a,b in zip(obj,obj[1:])))

if __name__=='__main__':unittest.main()
