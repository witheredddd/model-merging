"""CPU reconstruction of counterfactual controls from retained numerical inputs.

The algebra is retained, but CPU/GPU reduction orders can differ in last bits.
The recorded tensor hash identifies the original; it is not a cross-device promise.
"""
import argparse,json,re
from pathlib import Path
import torch

def reconstruct(root,case):
    root=Path(root);r=json.loads((root/'evaluations'/f'{case}.json').read_text())
    def js(name):return json.loads((root/name).read_text())
    def load(ref):return torch.load(root/ref['file'],map_location='cpu',mmap=True,weights_only=True)['vector'].double()
    anchor=r['anchor'];demand=r['demand'];cfg=js('CONFIG.json');first=cfg['anchors'][0]
    initial=js(f'bases/a{first}.json');family=r['family'];method=r['method']
    def gradient(task,probe,demand):return load(js(f'gradients/a{anchor}_{task}_p{probe}_{demand}.json'))
    def basis(kind):
        if kind=='frozen_gradient':return initial['basis']
        if kind=='existing_fisher':return js('bases/existing_fisher.json')['basis']
        return js(f'bases/a{anchor}.json')['basis']
    def combine(refs,coeff):
        v=torch.zeros(refs[0]['dimension'],dtype=torch.float64)
        for ref,c in zip(refs,coeff):v+=float(c)*load(ref)
        return v
    def project(v,refs):return combine(refs,[float(load(b)@v) for b in refs])
    def P(v,kind):
        if kind=='full':return v
        if kind=='freeze_learned_motion':return v-project(v,initial['basis'])
        support=js(f'regions/a{anchor}_selected.json')['supports'][kind]
        v=v.clone()
        if isinstance(support,dict):v[support['start']:support['stop']]=0
        else:
            regions=js('PARAMETER_REGIONS.json')
            for i in support:v[regions[i]['start']:regions[i]['stop']]=0
        return v
    if family=='sufficiency':
        g=gradient('DTD',2,demand)
        if method=='full_gradient':v=-g
        elif method=='mean_gradient_1d':
            mean=load(initial['mean_gradient']);mean/=mean.norm();v=-float(mean@g)*mean
        else:
            kind,rank=method.rsplit('_rank',1);refs=basis(kind)[:int(rank)]
            coeff=js(f'projections/a{anchor}_{demand}.json')['coefficients'][kind][:int(rank)]
            v=-combine(refs,coeff)
        return v/v.norm()
    if family=='motion_complement':
        kind,rank=method.removeprefix('outside_').rsplit('_rank',1);g=gradient('DTD',2,demand)
        v=-g+project(g,basis(kind)[:int(rank)]);return v/v.norm()
    if family=='euclidean_freeze':
        v=-P(gradient('DTD',2,demand),method);return v/v.norm()
    if family=='compatibility':
        gd=gradient('DTD',0,'mean');gc=gradient('Cars',0,'mean');w=r['mixture_weight']
        v=-(w*gd/gd.norm()+(1-w)*gc/gc.norm())
        if method.startswith('freeze_'):v=P(v,method.removeprefix('freeze_'))
        return v/v.norm()
    if family in ('fisher_freeze','fisher_common_objective'):
        name=r.get('solver_report') or ('solves/'+case.rsplit('_native_r',1)[0]+'.json')
        return load(js(name)['vector'])
    if family=='schur_conditional':
        full=load(r['full_solver_vector']);restricted=load(r['restricted_solver_vector']);t=r['driver_amplitude']
        if r['compensated']:return t*full+(1-t)*restricted
        return t*(full-P(full,method))+restricted
    if family in ('naive_freeze','naive_freeze_common_objective'):
        key=case.removesuffix('_naive_r1').removesuffix('_naive_native_r1');prefix=key[:-(len(method)+1)]
        return P(load(js(f'solves/{prefix}_full.json')['vector']),method)
    if family.startswith('expert_increment_'):
        trace=next(Path(k).parent for k in cfg['inputs'] if k.endswith('/trace.json'))
        rows=json.loads((trace/'trace.json').read_text())['rows'];row=next(x for x in rows if x['step']==anchor)
        visual=torch.load(trace/row['checkpoint'],map_location='cpu',mmap=True,weights_only=True)['visual']
        names=[x['name'] for x in js('PARAMETER_REGIONS.json')];point=torch.cat([visual[k].reshape(-1) for k in names]).double()
        inventory=next(Path(k) for k in cfg['inputs'] if k.endswith('INVENTORY.json'))
        tasks=json.loads(inventory.read_text())['tasks'];v=torch.zeros_like(point)
        for task,w in [('DTD',r['mixture_weight']),('Cars',1-r['mixture_weight'])]:
            row=next(t for t in tasks if t['task']==task)
            visual=torch.load(row['expert_path'],map_location='cpu',mmap=True,weights_only=False)['visual']
            v+=w*(torch.cat([visual[k].reshape(-1) for k in names]).double()-point)
        return v
    raise ValueError('Unknown control recipe: '+family)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('case');p.add_argument('destination',type=Path)
    a=p.parse_args()
    if a.destination.exists():raise FileExistsError(a.destination)
    torch.save(dict(vector=reconstruct(a.root,a.case)),a.destination)
