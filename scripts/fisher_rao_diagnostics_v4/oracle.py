"""Task CE and real FR diagnostics for a generated dictionary-coordinate model."""
import torch
from torch.nn import functional as F


class ClipOracle:
    def __init__(self,runtime,lift,case_name):
        self.rt,self.lift,self.case_name=runtime,lift,case_name
        self.geometry_trace=[];self._last_monitor=None
        self.radius=1.5*float((runtime.theta_w-runtime.theta0).norm())
        self.radius=max(self.radius,1e-6)

    def theta(self,x):return self.rt.theta_w+self.lift.S@x.to(self.lift.S.dtype)

    def sample(self,seed,step,batch,exclude):
        excluded=set(exclude);pool=[i for i in self.rt.split['train'] if i not in excluded]
        if len(pool)<batch:raise ValueError('Not enough independent training IDs for the geometry acceptance batch')
        generator=torch.Generator().manual_seed(seed*1000003+step)
        chosen=[pool[i] for i in torch.randperm(len(pool),generator=generator)[:batch].tolist()]
        x,y,ids=self.rt.train_batch(seed,step,batch,chosen)
        return dict(x=x,y=y,ids=ids,seed=seed,step=step)

    def value_grad(self,x,sample):
        self.rt.calls['training_images']+=len(sample['y'])
        xx=x.detach().requires_grad_()
        loss=F.cross_entropy(self.rt.predict(self.theta(xx),sample['x']),sample['y'])
        gradient=torch.autograd.grad(loss,xx)[0]
        return float(loss.detach()),gradient.detach()

    def replay_sample(self,recipe):
        x,y,ids=self.rt.train_batch(recipe['seed'],recipe['step'],len(recipe['ids']),recipe['ids'])
        return dict(x=x,y=y,ids=ids,seed=recipe['seed'],step=recipe['step'])

    def value(self,x,sample):
        self.rt.calls['training_images']+=len(sample['y'])
        with torch.no_grad():return float(F.cross_entropy(self.rt.predict(self.theta(x),sample['x']),sample['y']))

    def positive(self,u):
        with torch.no_grad():
            point=self.lift.qw.reshape(-1,self.rt.classes)
            angle=u.reshape_as(point).norm(dim=1)/point.norm(dim=1)
            return bool((angle<torch.pi/2).all() and (self.rt.exp(self.lift.qw,u)>0).all())

    def monitor(self,x,u):
        with torch.no_grad():
            actual=self.rt.phi(self.theta(x));target=self.rt.exp(self.lift.qw,u)
            error=float(self.rt.dist(actual,target));amplitude=float(self.rt.dist(self.lift.qw,target))
            function=self.rt.log(self.lift.qw,actual)
            normal=function-self.lift.L@(self.lift.L.T@function)
        self._last_monitor=dict(x=x.detach().cpu(),function_log=function.detach().cpu())
        return dict(FR_tracking_error=error,target_FR=amplitude,
                    relative_tracking_error=error/max(amplitude,1e-8),prediction_map='fixed_single_task_X',
                    real_log_chart_error=float((function-u).norm()),
                    outside_anchor_first_order_image=float(normal.norm()))

    def record_current(self,x,u,window):
        if self._last_monitor is None or not torch.equal(self._last_monitor['x'],x.detach().cpu()):
            self.monitor(x,u)
        self.geometry_trace.append(dict(window=window,x=x.detach().cpu(),
             function_log=self._last_monitor['function_log'],functional_target=u.detach().cpu(),
             actual_accepted_model=True,anchor='fixed_theta_w'))

    def scores(self,x,split='Vcal'):
        data=self.rt.scores(self.theta(x),split)
        return dict(CE=float(data['ce'].mean()),accuracy=float(data['acc'].mean()),n=len(data['ce']))

    def status(self,**fields):self.rt.status('group_conditioned_training',case=self.case_name,**fields)


def independent_reachability(runtime,lift,seed,excluded_ids,count=4):
    """Fresh directions not used to build S; no tautological self-span claim."""
    oracle=ClipOracle(runtime,lift,'independent_reachability');rows=[]
    used=set(excluded_ids)
    for j in range(count):
        sample=oracle.sample(seed,200000+j,32,exclude=used);used.update(sample['ids'])
        gradient,_=runtime.grad(runtime.theta_w,sample['x'],sample['y'])
        direction=-gradient/gradient.norm().clamp_min(1e-30)
        response=runtime.jvp(runtime.theta_w,direction)
        retained=float((lift.L.T@response).square().sum()/response.square().sum().clamp_min(1e-30))
        rows.append(dict(training_sample_ids=sample['ids'],functional_retention=retained,
              parameter_retention=float((lift.S.T@direction).square().sum()),
              definition='fresh gradient; excluded from initialization dictionary'))
    return rows
