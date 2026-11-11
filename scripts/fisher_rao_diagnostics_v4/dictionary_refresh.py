"""Opt-in fixed-rank dictionary rotation; a failed reconstruction is transactional."""
import torch
from manifold import span,align_to


def rotate_dictionary(S,x,direction):
    d=S.shape[1]
    if d<2:raise ValueError('Need two dictionary columns to protect the current model and add a direction')
    novel=direction-S@(S.T@direction)
    if float(novel.norm())<1e-10:return None
    novel=novel/novel.norm()
    if float(x.norm())>1e-12:
        rotation=torch.linalg.qr((x/x.norm())[:,None].to(S.dtype),mode='complete').Q
        result=S@rotation
    else:result=S.clone()
    result=result.clone();result[:,-1]=novel
    # Current displacement lies in the protected first direction when nonzero.
    target=S@x.to(S.dtype);newx=result.T@target
    if float((result@newx-target).norm())>1e-5*max(float(target.norm()),1e-12):
        raise ValueError('Dictionary rotation would lose the current parameter model')
    return result,newx


def propose(old_lift,groups,coordinates,direction,evidence,rebuild,validate):
    if evidence.get('persistent_windows',0)<3 or evidence.get('independent_task_gain',0)<=0:
        return dict(accepted=False,reason='insufficient_persistent_independent_outside_span_evidence')
    oldx=old_lift.coefficients(groups,coordinates)
    proposal=rotate_dictionary(old_lift.S,oldx,direction)
    if proposal is None:return dict(accepted=False,reason='no_new_parameter_span')
    newS,target=proposal;new_lift=rebuild(newS)  # MUST recompute A,C,L; never reuse stale C.
    remapped=[]
    for q in groups:
        function=old_lift.L@q
        candidate=new_lift.L.T@function
        u,s,vt=torch.linalg.svd(candidate,full_matrices=False)
        if len(s)<q.shape[1] or float(s[-1])<1e-5:
            return dict(accepted=False,reason='functional_group_rank_lost_after_dictionary_rotation')
        remapped.append(u@vt)
    H=new_lift.T@torch.cat(remapped,1);prior=torch.cat(coordinates)
    fitted=prior+torch.linalg.pinv(H,rtol=1e-8)@(target.to(H.dtype)-H@prior)
    error=float((newS@(H@fitted).to(newS.dtype)-old_lift.S@oldx.to(old_lift.S.dtype)).norm())
    relative=error/max(float((old_lift.S@oldx.to(old_lift.S.dtype)).norm()),1e-12)
    if relative>1e-4:return dict(accepted=False,reason='current_model_not_reconstructed',relative_error=relative)
    newcoords=list(fitted.split([q.shape[1] for q in remapped]))
    if not validate(new_lift,remapped,newcoords):return dict(accepted=False,reason='independent_CE_or_FR_rejected')
    return dict(accepted=True,lift=new_lift,groups=remapped,coordinates=newcoords,
           relative_reconstruction_error=relative,dictionary_rank_unchanged=newS.shape[1]==old_lift.S.shape[1])
