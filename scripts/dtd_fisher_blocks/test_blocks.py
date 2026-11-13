import unittest
import numpy as np
from blocks import discover,leakage

class BlocksTests(unittest.TestCase):
    def test_recovers_common_blocks_after_orthogonal_mixing(self):
        rng=np.random.default_rng(5);Q=np.linalg.qr(rng.normal(size=(5,5)))[0];family=[]
        for _ in range(6):
            A=np.zeros((5,5))
            for start,end in [(0,2),(2,5)]:
                B=rng.normal(size=(end-start,end-start));A[start:end,start:end]=B@B.T+np.eye(end-start)
            family.append(Q@A@Q.T)
        family=np.array(family);r=discover(family,[0,1,2,3],1e-7)
        self.assertEqual(len(set(r['labels'])),2)
        self.assertLess(max(leakage(np.array([r['Q'].T@A@r['Q'] for A in family]),r['labels'])),1e-7)
    def test_generic_family_is_not_forced_into_multiple_blocks(self):
        rng=np.random.default_rng(13);family=[]
        for _ in range(5):
            B=rng.normal(size=(5,5));family.append(B@B.T)
        r=discover(np.array(family),[0,1,2,3],1e-8)
        self.assertEqual(len(set(r['labels'])),1)

if __name__=='__main__':unittest.main()
