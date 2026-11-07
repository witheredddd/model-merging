"""Exercise unattended finalization with TINY SYNTHETIC fixtures in /tmp only.

This is an integration test, never a research result or task performance claim.
"""
import argparse,json,os,sys,tempfile,types
from pathlib import Path
import numpy as np
import torch
import engine,train,finalize

def run():
    actual_out=engine.OUT
    fixture=Path(tempfile.mkdtemp(prefix='functional_geometry_synthetic_test_'))
    rng=np.random.default_rng(5);zs=torch.zeros(8,48);zs[:,:2]=torch.tensor(rng.normal(size=(8,2)),dtype=torch.float32)
    torch.save(dict(z=zs,best_validation=[dict(ce=.5,accuracy=.81,n=2) for _ in range(8)]),fixture/'selected_experts.pt')
    engine.OUT=fixture;train.OUT=fixture;finalize.OUT=fixture
    class FakeEngine:
        def __init__(self,**kw):
            self.r=2;self.D=10;self.reserve=48;self.models=[None]*8;self.frozen=['frozen']*8
            self.encoder=types.SimpleNamespace(val_preprocess=None)
            ds=torch.utils.data.TensorDataset(torch.zeros(2,1),torch.zeros(2,dtype=torch.long))
            self.validation=[torch.utils.data.DataLoader(ds,batch_size=2) for _ in range(8)]
            self.u=types.SimpleNamespace(loader=lambda ds:ds,frozen_hash=lambda m:'frozen')
        def basis_load(self):pass
        def place(self,z):pass
        def evaluate(self,k,z,loader=None):return dict(ce=float((z-zs[k].cuda()).square().sum())+.5,accuracy=.5,n=2)
        def matrices(self,label,z):
            path=fixture/'functional'/f'{label}.npz';path.parent.mkdir(exist_ok=True)
            mats=[]
            for _ in range(8):
                a=rng.normal(size=(2,2));mats.append(a.T@a+np.eye(2))
            mats=np.array(mats);np.savez(path,train=mats,holdout=mats,audit=mats,h=np.ones((8,2)))
            return path
    finalize.Engine=FakeEngine
    fake_registry=types.ModuleType('src.datasets.registry');fake_registry.get_dataset=lambda *a,**kw:types.SimpleNamespace(test_dataset=[0,1])
    sys.modules['src.datasets.registry']=fake_registry
    finalize.finish(argparse.Namespace(micro=2))
    assert (fixture/'final_complete.json').exists()
    assert (fixture/'final_evidence.zip').exists()
    first={p.name:p.read_bytes() for p in (fixture/'locked_tests').glob('*.json')}
    # Resuming after tests must reuse locked coefficients and cached test results.
    finalize.finish(argparse.Namespace(micro=2))
    assert first=={p.name:p.read_bytes() for p in (fixture/'locked_tests').glob('*.json')}
    row=dict(passed=True,test_kind='synthetic small-tensor finalization integration test; NOT experiment results',fixture=str(fixture),
        verifies=['geometry gates','EC pilot execution','one-hot reconstruction','selection lock','cached-test resume','raw evidence archive','curves and report'])
    engine.write(actual_out/'pipeline_integration_check.json',row);print(json.dumps(row,indent=2))

if __name__=='__main__':run()
