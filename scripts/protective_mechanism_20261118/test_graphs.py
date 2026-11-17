import unittest
import numpy as np
import torch
import tempfile
from pathlib import Path
from bootstrap import R3
from graphs import conditional_graph,mechanism_report,metric_split,ArbitrarySupportSchur,joint_mode_curve,combined_reuse
from control_math import restricted_problem
from observe import input_fingerprint
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

    def test_full_spectrum_endpoint_equals_complete_WLS_and_invertible_basis_change(self):
        rng=np.random.default_rng(91);Y={'T':rng.normal(size=(8,4)),'P':rng.normal(size=(6,4))}
        G=np.eye(4);y=rng.normal(size=8);C=rng.normal(size=(4,4))+3*np.eye(4)
        ref=restricted_problem(Y,G,'T',y,{'P':1.},.2,problem='weighted_ls')
        a=joint_mode_curve(Y,G,'T',y,{'P':1.},.2,np.eye(4))
        b=joint_mode_curve(Y,G,'T',y,{'P':1.},.2,C)
        self.assertAlmostEqual(a['complete_prefix_curve'][-1]['objective'],ref['objective'])
        self.assertAlmostEqual(b['complete_prefix_curve'][-1]['objective'],ref['objective'])
        np.testing.assert_allclose(a['joint_modes']@a['requested_WLS_amplitudes'],ref['z'],atol=1e-12)
        np.testing.assert_allclose(b['joint_modes']@b['requested_WLS_amplitudes'],ref['z'],atol=1e-12)

    def test_fixed_response_prefix_is_distinct_from_soft_WLS(self):
        Y={'T':np.eye(2),'P':np.array([[1.,0.]])};y=np.ones(2)
        c=joint_mode_curve(Y,np.eye(2),'T',y,{'P':1.},10.,np.eye(2))
        self.assertFalse(c['fixed_response_prefix_curve'][0]['reachable'])
        self.assertTrue(c['fixed_response_prefix_curve'][-1]['reachable'])
        self.assertLess(c['fixed_response_prefix_curve'][-1]['target_relative_error'],1e-12)
        self.assertGreater(c['complete_prefix_curve'][-1]['target_relative_error'],.5)

    def test_cache_fingerprint_detects_replaced_data_without_index_change(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'region.npz';control=Path(d)/'control.bin';control.write_bytes(b'old')
            np.savez(f,value=np.ones(2));a=input_fingerprint([f],control,dict(index=7))
            np.savez(f,value=np.zeros(2));b=input_fingerprint([f],control,dict(index=7));self.assertNotEqual(a,b)
            control.write_bytes(b'new');self.assertNotEqual(b,input_fingerprint([f],control,dict(index=7)))

    def test_reuse_requires_same_physical_coordinates_and_recomputes_compensation(self):
        Y={'T':np.eye(2),'P':np.array([[1.,1.]])};G=np.eye(2);w={'P':1.}
        graphs=[conditional_graph(Y,G,'T',w,.1,[i]) for i in range(2)]
        family=np.column_stack([g['joint_modes'] for g in graphs]);old=joint_mode_curve(Y,G,'T',np.ones(2),w,.1,family)
        new=dict(Y,P=np.array([[1.,2.]]))
        result=combined_reuse(new,G,'T',np.ones(2),w,.1,graphs,old['joint_modes'],'same','same')
        self.assertTrue(result['same_new_demand_for_all_cases'])
        self.assertEqual(len(result['frozen_compensation_and_modes']),2)
        with self.assertRaises(ValueError):combined_reuse(new,G,'T',np.ones(2),w,.1,graphs,old['joint_modes'],'old','different')

if __name__=='__main__':unittest.main()
