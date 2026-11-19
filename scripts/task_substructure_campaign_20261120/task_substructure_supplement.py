"""Run after the primary worker: full complements of every rank and expert baselines."""
import argparse,json,sys
from pathlib import Path
import torch
import numpy as np
from task_substructure_utils import intervention_scale,archive_completed_controls

p=argparse.ArgumentParser()
p.add_argument('--scripts',type=Path,required=True)
for key in ('output','inventory','probes','coordinates','trace','flow-source'):p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--anchors',type=int,nargs='+',default=[0,25]);p.add_argument('--rtol',type=float,default=.001)
p.add_argument('--batch-size',type=int,default=16)
a=p.parse_args();sys.path.insert(0,str(a.scripts))
from worker import Experiment,UNSEEN
from io_state import atomic_json

e=Experiment(a);initial=json.loads((a.output/f'bases/a{a.anchors[0]}.json').read_text())
fisher=json.loads((a.output/'bases/existing_fisher.json').read_text())
for step in a.anchors:
    e.set_anchor(step);rhos=json.loads((a.output/f'calibration_a{step}.json').read_text())['rhos']
    for demand in UNSEEN:
        g=e.load(e.grad('DTD',2,demand))
        for family,refs in [('frozen_gradient',initial['basis']),('existing_fisher',fisher['basis'])]:
            for r in range(1,len(refs)+1):
                P=e.projector(basis=refs[:r]);v=-P(g);norm=float(v.norm())
                if not norm:continue
                ref=e.vector_cached(f'controls/a{step}_{demand}_outside_{family}_rank{r}',lambda:v/norm)
                for i,rho in enumerate(rhos):
                    e.evaluate(f'a{step}_{demand}_outside_{family}_rank{r}_r{i}',ref,rho,demand,
                        dict(family='motion_complement',method=f'outside_{family}_rank{r}',radius_index=i,
                            whole_parameter_complement=True,euclidean_proxy_deletion=.5*float(e.project(g,refs[:r]).norm())**2,
                            conditional_fisher_not_claimed=True))
    # Expert increment baselines have identical parameter radius and candidate count.
    increments=[]
    inventory=json.loads(a.inventory.read_text())['tasks']
    for task in ('DTD','Cars'):
        row=next(x for x in inventory if x['task']==task)
        visual=torch.load(row['expert_path'],map_location='cpu',mmap=True,weights_only=False)['visual']
        full=torch.cat([visual[k].reshape(-1) for k in e.keys]).cuda().double()
        increments.append(full-e.point)
    for weight in np.linspace(0,1,9):
        v=weight*increments[0]+(1-weight)*increments[1];norm=float(v.norm())
        ref=e.vector_cached(f'expert_baselines/a{step}_w{weight:.3f}',lambda:v)
        for suffix,scale in [('matched',intervention_scale(norm,rhos[1])),('original_scale',1.)]:
            e.evaluate(f'expert_a{step}_w{weight:.3f}_{suffix}',ref,scale,'mean',
                dict(family='expert_increment_'+suffix,method='ordinary_expert_coefficients',mixture_weight=float(weight),
                    radius_index=1 if suffix=='matched' else None,equal_radius=suffix=='matched' and norm>0,
                    zero_control=norm==0.,radius_upper_bound=rhos[1] if suffix=='matched' else None,
                    original_scale_has_different_parameter_budget=suffix=='original_scale'))
    # Independent two-coefficient search, not only a convex expert-average line.
    for coefficient_DTD in (0.,.5,1.):
        for coefficient_Cars in (0.,.5,1.):
            v=coefficient_DTD*increments[0]+coefficient_Cars*increments[1];norm=float(v.norm())
            ref=e.vector_cached(f'expert_baselines/a{step}_grid_D{coefficient_DTD}_C{coefficient_Cars}',lambda:v)
            for suffix,scale in [('matched',intervention_scale(norm,rhos[1])),('original_scale',1.)]:
                e.evaluate(f'expert_grid_a{step}_D{coefficient_DTD}_C{coefficient_Cars}_{suffix}',ref,scale,'mean',
                    dict(family='expert_grid_'+suffix,method='independent_expert_coefficients',
                        coefficient_DTD=coefficient_DTD,coefficient_Cars=coefficient_Cars,mixture_weight=None,
                        radius_index=1 if suffix=='matched' else None,equal_radius_for_nonzero_direction=suffix=='matched' and norm>0,
                        radius_upper_bound=rhos[1] if suffix=='matched' else None,
                        zero_direction_not_normalized=not bool(norm),
                        normalized_grid_can_have_repeated_directions=True,
                        original_scale_has_different_parameter_budget=suffix=='original_scale'))
    del increments
    if hasattr(e,'archive_controls'):e.archive_controls(step)
    else:archive_completed_controls(e.out,step)
atomic_json(a.output/'SUPPLEMENT_COMPLETE.json',dict(anchors=a.anchors,full_complement_all_resolved_ranks=True,
    expert_increment_baselines=True,final_controlled_training=False,EC_search=False))
e.status('COMPLETE')
