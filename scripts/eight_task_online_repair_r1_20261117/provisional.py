"""Revisable per-demand provisional modes; no KKT admission and no optimum claim."""
import copy
import numpy as np
import torch
from conditional_controls import ConditionalControls,plain
from checkpoint import atomic_json,digest


class ProvisionalControls(ConditionalControls):
    def vector(self,index):
        mode=self.state['modes'][index]
        if not mode.get('dual_only'):return super().vector(index)
        # Exact algebraic adjoint representation on this locked model/window.
        # The materialization cache is disposable, never the recovery authority.
        cache=self.root/'materialized';cache.mkdir(exist_ok=True)
        path=cache/(mode['duals_sha256']+'.pt')
        if path.exists():
            return self.g.numpy_to_tensor(torch.load(path,map_location='cpu',weights_only=True)['direction'].numpy())
        q=self.duals(index);v=self.g.numpy_to_tensor(np.zeros(self.g.dimension))
        for task in self.g.tasks:v+=self.g.vjp(task,self.g.numpy_to_tensor(q[task]))
        temp=path.with_suffix('.tmp');torch.save(dict(direction=v.detach().cpu()),temp);temp.replace(path)
        return v

    def refresh(self,v,duals,quality,owner):
        keep=[i for i,m in enumerate(self.state['modes']) if m['source']['case_id']!=owner]
        # Replace this demand's provisional mode; do not append every Krylov
        # iterate as another independent permanent control.
        old=self.state;self.state=copy.deepcopy(old)
        self.state['modes']=[self.state['modes'][i] for i in keep]
        G=np.asarray(old['gram']).reshape(len(old['modes']),len(old['modes']))
        self.state['gram']=G[np.ix_(keep,keep)].tolist()
        event=self.observe_provisional(v,duals,quality,dict(case_id=owner))
        if event['independent_at_machine_precision']:
            mode=self.state['modes'][-1];path=self.root/mode['vector']
            cache=self.root/'materialized';cache.mkdir(exist_ok=True)
            dest=cache/(mode['duals_sha256']+'.pt')
            path.replace(dest)
            mode.update(dual_only=True,vector=None,vector_sha256=None,
                representation='Full adjoint coefficients; FP64 lift reconstruction error recorded')
        atomic_json(self.root/'CONTROL_STATE.json',self.state)
        return event

    def prune_materialization_cache(self):
        cache=self.root/'materialized'
        active={m['duals_sha256']+'.pt' for m in self.state['modes'] if m.get('dual_only')}
        if cache.exists():
            for path in cache.glob('*.pt'):
                if path.name not in active:path.unlink() # Derived cache only; dual checkpoints are retained.


def focused_take(scheduler):
    if scheduler['inflight'] is not None:return scheduler['inflight']
    cases=scheduler['cases']
    resolved={'certified','numerical_breakdown','working_precision_exhausted','undefined_zero_source','zero_response'}
    started=[c for c in cases.values() if c['iterations']>0 and c['status'] not in resolved]
    candidates=started or [c for c in cases.values() if c['status'] not in resolved]
    if not candidates:return None
    # Existing progress first. No active-set size, rank, iteration or time cap.
    case=max(candidates,key=lambda c:(c['iterations'],-list(cases).index(c['id'])))
    scheduler['dispatches']+=1
    scheduler['inflight']=dict(case_id=case['id'],start_iteration=case['iterations'])
    return scheduler['inflight']
