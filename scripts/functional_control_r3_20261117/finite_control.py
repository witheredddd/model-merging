"""A real Bz update selected using explicit nonlinear WLS and CE descent."""
import numpy as np
import torch
from torch.nn import functional as F


def probabilities(g,point,baseline=None):
    result={}
    for task in g.tasks:
        logits,labels=g.c.baseline(task,g.probe,point)
        p=logits.softmax(-1)
        result[task]=dict(phi=(2*p.sqrt()/np.sqrt(len(labels))).reshape(-1),
            CE=float(F.cross_entropy(logits,labels)),Top1=float((logits.argmax(-1)==labels).double().mean()))
    return result


def line_search(g,motion,y,target,weights,rho,target_weight,CE_slope,*,leakage_budget=None):
    before=probabilities(g,g.point);desired=g.tensor_to_numpy(y)
    baseline_merit=.5*target_weight*float(desired@desired)
    Jmotion=g.tensor_to_numpy(g.jvp(target,motion))
    derivative=-target_weight*float(desired@Jmotion)
    if CE_slope>=0 or derivative>=0:
        return None,dict(accepted=False,reason='Proposed actuation is not CE and WLS descent',CE_slope=CE_slope,merit_derivative=derivative)
    alpha=1.;attempts=[];norm_squared=float(motion.square().sum())
    while alpha>0:
        point=g.point+alpha*motion
        if torch.equal(point,g.point):break
        current=probabilities(g,point);responses={t:(current[t]['phi']-before[t]['phi']).numpy() for t in g.tasks}
        protected={t:float(responses[t]@responses[t]) for t in weights}
        residual=responses[target]-desired
        merit=.5*sum(weights[t]*protected[t] for t in weights)+.5*rho*alpha*alpha*norm_squared+.5*target_weight*float(residual@residual)
        target_ce=current[target]['CE']-before[target]['CE']
        safe=leakage_budget is None or sum(weights[t]*protected[t] for t in weights)<=leakage_budget
        accepted=merit<=baseline_merit+1e-4*alpha*derivative and target_ce<=1e-4*alpha*CE_slope and safe
        row=dict(alpha=alpha,merit=merit,baseline_merit=baseline_merit,target_CE_change=target_ce,
            actual_protected_response_energies=protected,
            CE_changes={t:current[t]['CE']-before[t]['CE'] for t in g.tasks},
            Top1_changes={t:current[t]['Top1']-before[t]['Top1'] for t in g.tasks},accepted=accepted)
        attempts.append(row)
        if accepted:return point,dict(accepted=True,alpha=alpha,attempts=attempts,
            explicit_hard_leakage_budget=leakage_budget,
            actual_update_is_mode_amplitudes=True,
            acceptance='Armijo on explicit nonlinear WLS merit and true target CE; regularization alone is not a hard leakage guarantee')
        alpha*=.5
    return None,dict(accepted=False,reason='No representable finite descent step',attempts=attempts)
