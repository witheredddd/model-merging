import unittest
from dataclasses import replace
import torch
from state import Group
from geometry import refresh,fr_preconditioned_direction
from work import MeasuredOracle,Ledger
from test_online_v5 import ToyOracle,basic_state


class CoordinateQuotientTests(unittest.TestCase):
    def test_redundant_overlapping_groups_preserve_parameter_step_without_fallback(self):
        raw=ToyOracle();state=basic_state(raw,2)
        geo=refresh(state,MeasuredOracle(raw,Ledger()))
        G=state.G;gx=torch.tensor([.2,-.3,.1],dtype=torch.double)
        reference,_=fr_preconditioned_direction(state,geo,gx)
        for mixing in [torch.tensor([[1.,0.,1.,2.],[0.,1.,0.,-3.]],dtype=torch.double),
                       torch.tensor([[100.,.1,2.],[0.,.01,-1.]],dtype=torch.double)]:
            Z=G@mixing
            redundant=replace(state,groups=tuple(Group(str(j),Z[:,j:j+1],Z.new_zeros(1)) for j in range(Z.shape[1])))
            delta,info=fr_preconditioned_direction(redundant,geo,gx)
            self.assertIsNone(info['fallback'])
            self.assertEqual(info['effective_system_dimension'],2)
            self.assertEqual(info['coordinate_nullity'],Z.shape[1]-2)
            torch.testing.assert_close(Z@delta,G@reference,atol=1e-10,rtol=1e-10)

    def test_zero_represented_space_returns_no_step_without_refresh_fallback(self):
        raw=ToyOracle();state=basic_state(raw)
        geo=refresh(state,MeasuredOracle(raw,Ledger()))
        empty=replace(state,groups=(Group('zero',torch.zeros_like(state.G),state.a),))
        delta,info=fr_preconditioned_direction(empty,geo,torch.ones(3,dtype=torch.double))
        torch.testing.assert_close(delta,torch.zeros_like(delta))
        self.assertIsNone(info['fallback']);self.assertEqual(info['effective_system_dimension'],0)

    def test_genuine_reduced_system_condition_failure_still_uses_fallback(self):
        raw=ToyOracle();state=basic_state(raw,2)
        geo=refresh(state,MeasuredOracle(raw,Ledger()))
        delta,info=fr_preconditioned_direction(state,geo,torch.ones(3,dtype=torch.double),condition_limit=1.)
        self.assertEqual(info['fallback'],'ill_conditioned_FR_system')
        self.assertTrue(torch.isfinite(delta).all())


if __name__=='__main__':unittest.main()
