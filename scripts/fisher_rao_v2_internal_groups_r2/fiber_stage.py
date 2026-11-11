"""Optional post-horizontal fiber study in the SAME task quotient; no group discovery."""
import importlib.util
import sys
from pathlib import Path
import torch
from runtime_task import write,load


def run_after_validation(rt,S,horizontal,seed,output,validation_path):
    report=__import__('json').loads(Path(validation_path).read_text(encoding='utf-8'))
    if not report['multi_group_hypothesis_supported']:
        write(Path(output)/'fiber_deferred.json',dict(reason='within_task_multigroup_validation_not_passed',
            functional_groups_are_not_fibers=True));return None
    # Reuse the frozen, numerically verified correction implementation, with a
    # strict single-task facade. Its base is the FULL warm anchor, not theta0.
    previous=Path(__file__).resolve().parent.parent/'fisher_rao_v2'
    spec=importlib.util.spec_from_file_location('verified_fiber_mechanism',previous/'fiber.py')
    sys.path.insert(0,str(previous));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.path.pop(0)
    class SingleTaskFiber:
        def __init__(self):
            self.base=rt.theta_w;self.out=Path(output);self.tasks=[rt.task];self.calls=rt.calls;self.n=rt.n
        def view(self,split='X',only=None,n=None):
            assert only is None or only==rt.task
            count=(rt.n if n is None else n) if split in ['X','Xprime'] else len(rt.split[split])
            return [(rt.task,split,count)]
        def chunks(self,task,split,n=None,batch=16):assert task==rt.task;return rt.batches(split,batch,n)
        def predict(self,theta,b,task):assert task==rt.task;return rt.predict(theta,b)
        def phi(self,theta,view):assert all(t==rt.task for t,_,_ in view);return rt.phi(theta,view[0][1])
        def phi_co(self,S,co,view,grad=False):return rt.phi_co(S,co,view[0][1],grad)
        def dist(self,q,p,view):return rt.dist(q,p)
        def log(self,q,p,view):return rt.log(q,p)
        def jvp(self,theta,v,view):return rt.jvp(theta,v,view[0][1])
        def vjp(self,theta,a,view):return rt.vjp(theta,a,view[0][1])
        def grad(self,theta,task,b,y):assert task==rt.task;return rt.grad(theta,b,y)
        def response(self,theta,S,view,label='fiber'):
            return torch.stack([self.jvp(theta,v,view) for v in S.T],1)
        def scores(self,theta,task,split='Vcal',retain=False):assert task==rt.task;return rt.scores(theta,split,retain)
    exp=SingleTaskFiber();exp.out.mkdir(parents=True,exist_ok=True)
    result=module.run(exp,S,horizontal,rt.task,seed,exp.out,n=rt.n)
    if result.get('completed'):
        Sfull=torch.cat([S,result['extra_S'].to(S.device)],1);xh=result['xh'].to(S.device)
        theta_h=rt.theta_w+Sfull@xh.float();qp=rt.phi(theta_h,'Xprime').detach()
        baseline=rt.phi(rt.theta_w,'Xprime').detach();tolerance=.05*float(rt.dist(qp,baseline));checks={}
        yfit_baseline=float(rt.scores(theta_h,'Yfit')['ce'].mean());cal_baseline=float(rt.scores(theta_h)['ce'].mean())
        for name,arm in result['arms'].items():
            co=arm.get('matched_co')
            if co is None:continue
            theta=rt.theta_w+Sfull@co.to(S.device).float();drift=float(rt.dist(qp,rt.phi(theta,'Xprime')))
            yfit=float(rt.scores(theta,'Yfit')['ce'].mean());cal=float(rt.scores(theta)['ce'].mean())
            checks[name]=dict(Xprime_FR_drift=drift,Xprime_tolerance=tolerance,prediction_preserved=drift<=tolerance,
                Yfit_CE=yfit,Vcal_CE=cal,Yfit_gain=yfit_baseline-yfit,Vcal_gain=cal_baseline-cal)
        write(exp.out/'independent_preservation.json',dict(task=rt.task,anchor='theta_w',functional_groups_are_not_fibers=True,checks=checks))
    return result
