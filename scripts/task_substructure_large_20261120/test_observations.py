import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from observations import functional_observation
from numerics import sphere_log

class ObservationTest(unittest.TestCase):
    def test_shared_full_precision_arrays_reproduce_functional_response(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'evaluations').mkdir();(root/'linear_responses').mkdir()
            base=np.array([[2.,1.,0.],[0.,1.,2.]]);next_=base+np.array([[.01,-.02,0.],[0.,.02,-.01]])
            y=np.array([0,2]);j=np.arange(6,dtype=float)/100
            np.savez(root/'linear_responses/base.npz',DTD_p3_baseline=base,DTD_p3_labels=y)
            np.savez(root/'linear_responses/unit.npz',JVP=j)
            np.savez(root/'evaluations/one.npz',DTD_p3_logits=next_)
            (root/'evaluations/one.json').write_text(json.dumps(dict(arrays='evaluations/one.npz',scale=.01,
                shared_baseline_file='linear_responses/base.npz',shared_linear_response_files={'DTD_p3':'linear_responses/unit.npz'})))
            result=functional_observation(root,'one');np.testing.assert_array_equal(result['JVP'],.01*j)
            a=2*torch.tensor(base).softmax(-1).sqrt().numpy()/np.sqrt(2)
            b=2*torch.tensor(next_).softmax(-1).sqrt().numpy()/np.sqrt(2)
            np.testing.assert_array_equal(result['actual_Log'],sphere_log(a,b).reshape(-1))

if __name__=='__main__':unittest.main()
