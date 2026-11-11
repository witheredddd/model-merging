"""Controlled nonlinear-prediction, actual-CE two-window smoke on CPU only.

This is synthetic evidence of the update mechanism, not DTD/CLIP performance.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import torch
from torch.nn import functional as F
from fr_math import embedding
from trainer import OnlineTrainer,Settings
from steps import StepSettings
from lookahead import LookaheadSettings
from runner import write


class ClassificationOracle:
    def __init__(self):
        self.theta0=torch.zeros(4,dtype=torch.double);self.n=6;self.phase=0
        self.calls=dict(images=0,gradient=0,jvp=0)
        self.features=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],
            [0.,0.,0.,1.],[.7,.7,0.,0.],[0.,.7,0.,.7]],dtype=torch.double)

    def sample(self,seed,step,exclude=()):
        ids=[seed*10000000+step*4,seed*10000000+step*4+1]
        while set(ids).intersection(exclude):ids=[i+1000000000 for i in ids]
        feature=torch.tensor([1.,0.,.001*(step%2),0.],dtype=torch.double)
        if self.phase:feature=torch.tensor([0.,1.,0.,.001*(step%2)],dtype=torch.double)
        return dict(ids=ids,x=feature.repeat(2,1),y=torch.ones(2,dtype=torch.long))

    def logits(self,theta,x):
        score=x@theta
        return torch.stack([torch.zeros_like(score),score],1)

    def _phi(self,theta):
        return embedding(self.logits(theta,self.features))

    def phi(self,theta,positions=None):
        self.calls['images']+=self.n if positions is None else len(positions)
        q=self._phi(theta)
        return q if positions is None else q[positions]

    def jvp(self,theta,v):
        self.calls['images']+=self.n;self.calls['jvp']+=1
        return torch.autograd.functional.jvp(self._phi,theta,v)[1]

    def gradient(self,theta,sample):
        self.calls['images']+=len(sample['y']);self.calls['gradient']+=1
        point=theta.detach().requires_grad_()
        loss=F.cross_entropy(self.logits(point,sample['x']),sample['y'])
        return float(loss.detach()),torch.autograd.grad(loss,point)[0].detach()

    def loss(self,theta,sample):
        self.calls['images']+=len(sample['y'])
        return float(F.cross_entropy(self.logits(theta,sample['x']),sample['y']))


def smoke():
    torch.set_num_threads(1)
    raw=ClassificationOracle()
    settings=Settings(cold_batches=2,persistence_windows=2,geometry_every=100,
        step=StepSettings(monitor_count=4),maximum_proposals=3,
        lookahead=LookaheadSettings(rank_penalty=0.,group_penalty=0.,continuity_penalty=0.))
    trainer=OnlineTrainer(raw,settings)
    assert torch.equal(trainer.state.parameters(),raw.theta0)
    trainer.step(raw.sample(11,20));base_state=trainer.state.clone()
    raw.phase=1
    first=trainer.geometry_window();second=trainer.geometry_window()
    dictionary=[p for p in second['proposals'] if p.get('kind')=='dictionary_replace' and p.get('accepted')]
    assert dictionary,'No accepted online dictionary replacement in controlled CE scenario'
    assert trainer.state.dictionary_version>base_state.dictionary_version
    proposal=dictionary[0]
    assert proposal['future_CE_new']<proposal['future_CE_old']
    assert proposal['old_branch_cost']==proposal['new_branch_cost']
    before=trainer.state.parameters().clone()
    trainer.step(raw.sample(11,30))
    assert float((trainer.state.parameters()-before).norm())>0
    assert trainer.ledger.used==raw.calls['images']
    trainer.state.validate()
    torch.testing.assert_close(trainer.state.parameters(),raw.theta0+trainer.state.S@(trainer.state.G@trainer.state.a))
    return dict(status='passed',device='CPU',task='synthetic_CE_controlled_batch_family_shift',
        training_probe_function_fixed=True,cold_high_dimensional_optimizer_steps=0,
        windows=2,dictionary_version=trainer.state.dictionary_version,groups=len(trainer.state.groups),
        independent_future_CE_old=proposal['future_CE_old'],independent_future_CE_new=proposal['future_CE_new'],
        old_branch_image_cost=proposal['old_branch_cost'],new_branch_image_cost=proposal['new_branch_cost'],
        pure_dictionary_parameter_error=proposal['metadata']['parameter_error'],
        subsequent_coordinate_parameter_change=float((trainer.state.parameters()-before).norm()),
        global_invariant=True,online_images=trainer.ledger.used,operations=trainer.ledger.operations,
        A100_executed=False,CLIP_executed=False,DTD_accuracy=None,
        limitations='Synthetic mechanism test; does not validate DTD performance or independent functional groups')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=smoke();args.output.parent.mkdir(parents=True,exist_ok=True);write(args.output,result)
    print(result)
