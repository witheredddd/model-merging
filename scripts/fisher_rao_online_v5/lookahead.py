"""Geometry acceptance measures equal-cost subsequent optimization, not one CE."""
from dataclasses import dataclass
import math
import torch
from geometry import refresh,coefficient_step_radius
from steps import coordinate_trial
from work import Ledger,MeasuredOracle,BudgetEnd


@dataclass(frozen=True)
class LookaheadSettings:
    steps: int = 2
    validation_noise: float = 1e-4
    maximum_initial_rise: float = .05
    rank_penalty: float = 1e-4
    group_penalty: float = 1e-5
    dictionary_penalty: float = 1e-5
    continuity_penalty: float = 1e-5

    def __post_init__(self):
        if self.steps not in [2,3,4]:raise ValueError('Matched lookahead uses 2-4 task batches')
        values=[self.validation_noise,self.maximum_initial_rise,self.rank_penalty,self.group_penalty,
                self.dictionary_penalty,self.continuity_penalty]
        if not all(math.isfinite(v) and v>=0 for v in values):raise ValueError('Invalid lookahead controls')


def complexity(state,settings):
    return (settings.rank_penalty*state.total_rank+settings.group_penalty*len(state.groups)
            +settings.dictionary_penalty*state.S.shape[1])


def compare(old,new,raw,ledger,samples,validation,step_settings,settings,continuity=0.,gamma=1e-5):
    """All work in BOTH branches is charged, even when the proposal is rejected.

    Gradients, monitor predictions and validation use identical sample IDs.
    Group/dictionary changes can move the initial model; this is recorded and
    bounded before measuring the short-run value. No free branch progress.
    """
    if len(samples)!=settings.steps:raise ValueError('Wrong lookahead sample count')
    fitting=set(i for sample in samples for i in sample['ids'])
    if fitting.intersection(validation['ids']):raise ValueError('Lookahead validation leaks fitting IDs')
    width=max(old.S.shape[1],new.S.shape[1])
    n_monitor=min(step_settings.monitor_count,raw.n)
    image_cost=raw.n*(width+1)+2*len(validation['y'])+sum(2*len(s['y'])+n_monitor for s in samples)
    if ledger.used+2*image_cost>ledger.limit:raise BudgetEnd('Cannot fund both geometry branches')
    branches=[]
    for label,start in [('old',old),('new',new)]:
        branch_ledger=Ledger(image_cost);oracle=MeasuredOracle(raw,branch_ledger,'lookahead_'+label+'_')
        try:
            state=start.clone();geometry=refresh(state,oracle,gamma)
            # Extra JVP checks equalize the image cost of an explicit d-growth
            # proposal, without charging invented work or applying an update.
            if state.S.shape[1]<width:
                for v in new.S[:,state.S.shape[1]:].T:oracle.jvp(state,v)
            initial_loss=oracle.loss(state,validation)
            radius=coefficient_step_radius(geometry);events=[]
            for sample in samples:
                state,radius,event=coordinate_trial(state,geometry,oracle,sample,step_settings,radius)
                events.append(event)
            final_loss=oracle.loss(state,validation)
            if branch_ledger.used!=image_cost:raise RuntimeError('Lookahead branch cost mismatch')
        finally:
            ledger.merge_branch(branch_ledger)
        branches.append(dict(state=state,initial=initial_loss,final=final_loss,events=events,
                             cost=branch_ledger.used,operations=branch_ledger.operations,geometry=geometry))
    before,after=branches
    finite=all(math.isfinite(branch[k]) for branch in branches for k in ['initial','final'])
    ceiling=settings.maximum_initial_rise*max(1.,abs(before['initial']))
    initial_safe=finite and after['initial']<=before['initial']+ceiling
    noise=settings.validation_noise*max(1.,abs(before['final']))
    task_safe=finite and after['final']<=before['final']+noise
    gain=before['final']-after['final']
    score=gain-(complexity(new,settings)-complexity(old,settings))-settings.continuity_penalty*continuity
    accepted=initial_safe and task_safe and score>1e-10
    info=dict(accepted=accepted,initial_CE_old=before['initial'],initial_CE_new=after['initial'],
        future_CE_old=before['final'],future_CE_new=after['final'],future_CE_gain=gain,net_gain=score,
        old_branch_cost=before['cost'],new_branch_cost=after['cost'],old_operations=before['operations'],
        new_operations=after['operations'],commit='new_branch' if accepted else 'old_branch',
        old_accepted_steps=sum(e['accepted'] for e in before['events']),
        new_accepted_steps=sum(e['accepted'] for e in after['events']),
        future_parameter_difference=float((after['state'].parameters()-before['state'].parameters()).norm()),
        reason='accepted' if accepted else ('catastrophic_initial_move' if not initial_safe else
            ('future_task_loss' if not task_safe else 'capacity_or_continuity_cost')),
        starting_parameter_move=float((new.parameters()-old.parameters()).norm()))
    info['_selected_geometry']=after['geometry'] if accepted else before['geometry']
    return after['state'] if accepted else before['state'],info
