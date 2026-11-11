"""Distinguish recurring residual subspaces from recurring but different novelty."""
import math
import torch
from manifold import span


def transport_columns(current,reference,matrix):
    """Product-sphere parallel transport to one fixed observational tangent.

    This reference is evidence bookkeeping, never a parameter/model anchor.
    """
    if current.shape!=reference.shape or matrix.shape[0]!=current.numel():
        raise ValueError('Functional observation shapes changed across windows')
    if not torch.allclose(current.norm(dim=1),reference.norm(dim=1),rtol=2e-5,atol=1e-7):
        raise ValueError('Functional-map normalization changed across windows')
    vectors=matrix.reshape(*current.shape,matrix.shape[1])
    radius2=current.square().sum(1,keepdim=True)
    radial=(current[:,:,None]*vectors).sum(1)
    tolerance=1e-8+1e-5*current.norm(dim=1,keepdim=True)*vectors.norm(dim=1)
    if not torch.isfinite(vectors).all() or bool((radial.abs()>tolerance).any()):
        raise ValueError('Residual columns are not finite current functional tangents')
    vectors=vectors-current[:,:,None]*(radial/radius2)[:,None,:]
    denominator=radius2+(current*reference).sum(1,keepdim=True)
    if bool((denominator<=1e-8*radius2).any()):raise ValueError('Antipodal evidence transport')
    coefficient=(reference[:,:,None]*vectors).sum(1)/denominator
    result=vectors-(current+reference)[:,:,None]*coefficient[:,None,:]
    return result.reshape(matrix.shape)


class StructuralPersistence:
    def __init__(self,required=3,outside_fraction=.2,overlap=.7):
        if required<1 or not 0<overlap<=1:raise ValueError('Invalid recurrence controls')
        self.required=required;self.threshold=outside_fraction;self.overlap=overlap
        self.windows=[];self.reference=None;self.last={}

    def snapshot(self):
        return list(self.windows),self.reference,dict(self.last)

    def restore(self,snapshot):
        self.windows,self.reference,self.last=snapshot

    def excluded_ids(self):
        previous=self.windows[-(self.required-1):] if self.required>1 else []
        return set().union(*(record['ids'] for record in previous)) if previous else set()

    def observe(self,evidence):
        ids=set(i for row in evidence['rows'] for i in row['ids'])|set(evidence['independent_ids'])
        previous=self.windows[-(self.required-1):] if self.required>1 else []
        independent=not any(ids.intersection(old['ids']) for old in previous)
        eligible=[i for i,row in enumerate(evidence['rows']) if row['outside_fraction']>self.threshold
                  and row['independent_candidate_gain']>0 and float(evidence['outside'][:,i].norm())>1e-10]
        report=dict(kind='insufficient_history',coherent=False,diverse=False,independent=independent,
                    candidate_indices=eligible,weights=None,selected_index=None,structural_identity_proven=False)
        if not independent:
            report['kind']='dependent_window';self.last=report;return report
        if not eligible:
            self.windows=[];self.reference=None
            report['kind']='no_beneficial_outside_residual';self.last=report;return report
        if self.reference is None:self.reference=evidence['q'].detach().clone()
        transported=transport_columns(evidence['q'],self.reference,evidence['outside'][:,eligible])
        unit=transported/transported.norm(dim=0,keepdim=True).clamp_min(1e-30)
        basis=span(unit,tol=1e-6)
        comparisons=[]
        for old in previous:
            cosines=torch.linalg.svdvals(old['basis'].T@basis).clamp(0,1)
            comparisons.append(dict(principal_cosines=cosines.tolist(),
                projection_overlap=float(cosines.square().sum())/max(old['basis'].shape[1],basis.shape[1])))
        self.windows=(self.windows+[dict(ids=ids,basis=basis.detach())])[-self.required:]
        report.update(independent_windows=len(self.windows),current_residual_rank=basis.shape[1],comparisons=comparisons)
        if len(self.windows)<self.required:
            self.last=report;return report
        # Mean-projector eigenvalues from a thin concatenated matrix, never p x p.
        pooled=torch.cat([record['basis'] for record in self.windows],1)/math.sqrt(len(self.windows))
        vectors,s,_=torch.linalg.svd(pooled,full_matrices=False)
        support=s.square().clamp(0,1)
        stable=vectors[:,support>=self.overlap]
        if stable.shape[1]:
            present=(basis.T@stable).square().sum(0)
            stable=stable[:,present>=self.overlap]
        report['mean_projector_support']=support.tolist()
        report['recurring_rank']=stable.shape[1]
        if stable.shape[1]:
            # Reconstruct a parameter candidate from the current response bank's
            # common residual component, rather than taking the largest one-off.
            gains=transported.new_tensor([evidence['rows'][i]['independent_candidate_gain'] for i in eligible])
            gains=gains/gains.mean().clamp_min(1e-12)
            task_supported=(stable.T@transported)*gains.sqrt()[None,:]
            axes,_,_=torch.linalg.svd(task_supported,full_matrices=False)
            target=stable@axes[:,0]
            coefficients=torch.linalg.pinv(transported,rtol=1e-6)@target
            weights=evidence['R'].new_zeros(evidence['R'].shape[1])
            weights[eligible]=coefficients
            projection=(stable.T@unit).square().sum(0)
            chosen=eligible[int(projection.argmax())]
            report.update(kind='coherent_recurrence',coherent=True,weights=weights.tolist(),selected_index=chosen)
        else:
            chosen=max(eligible,key=lambda i:evidence['rows'][i]['outside_fraction']*
                       evidence['rows'][i]['independent_candidate_gain'])
            report.update(kind='diverse_recurrence',diverse=True,selected_index=chosen)
        self.last=report;return report
