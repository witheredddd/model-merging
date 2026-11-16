import unittest
import numpy as np
from scipy.sparse.linalg import LinearOperator
from iterative import continuing_cg


class TestContinuation(unittest.TestCase):
    def test_chunks_preserve_the_uninterrupted_cg_iteration(self):
        rng=np.random.default_rng(4);Q=rng.normal(size=(35,35));M=Q.T@Q+.2*np.eye(35)
        A=LinearOperator(M.shape,matvec=lambda x:M@x,dtype=np.float64);b=rng.normal(size=35)
        uninterrupted,_,whole=continuing_cg(A,b,rtol=1e-15,max_iterations=16)
        _,_,first=continuing_cg(A,b,rtol=1e-15,max_iterations=7)
        resumed,_,last=continuing_cg(A,b,rtol=1e-15,max_iterations=9,state=first)
        np.testing.assert_array_equal(resumed,uninterrupted)
        np.testing.assert_array_equal(last['p'],whole['p'])
        solved,info,_=continuing_cg(A,b,rtol=1e-11,max_iterations=100,state=last)
        self.assertEqual(info,0)
        np.testing.assert_allclose(M@solved,b,atol=1e-10)


if __name__=='__main__':unittest.main()
