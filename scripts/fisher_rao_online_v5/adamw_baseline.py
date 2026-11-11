"""Independent full-dimensional AdamW control. Never imported by OnlineTrainer."""
import torch


class AdamWControl:
    def __init__(self,raw,learning_rate=1e-5,weight_decay=.01):
        self.raw=raw
        self.theta=raw.theta0.detach().clone().requires_grad_()
        self.optimizer=torch.optim.AdamW([self.theta],lr=learning_rate,weight_decay=weight_decay)
        self.images=0;self.backward=0

    def step(self,sample):
        loss,g=self.raw.gradient(self.theta.detach(),sample)
        self.images+=len(sample['y']);self.backward+=1
        if not torch.isfinite(g).all():return dict(accepted=False,reason='nonfinite_baseline_gradient')
        self.optimizer.zero_grad(set_to_none=True);self.theta.grad=g.detach().clone()
        self.optimizer.step()
        return dict(accepted=True,loss=loss,method='independent_AdamW_control')

    def parameters(self):
        return self.theta.detach()
