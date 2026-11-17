import numpy as np
from bootstrap import R3
from diagnostics import compare,measurement
from control_math import restricted_problem


def functional_signal(Y,G,target,y,weights,rho,new_responses,cross,norm2):
    """Measure the SAME WLS before/after a candidate; no gain percentage gate."""
    r=G.shape[0];aug=np.empty((r+1,r+1));aug[:r,:r]=G
    aug[:r,r]=cross;aug[r,:r]=cross;aug[r,r]=norm2
    extended={t:np.column_stack([Y[t],new_responses[t]]) for t in Y}
    old_z=(restricted_problem(Y,G,target,y,weights,rho,problem='weighted_ls')['z'] if r else np.empty(0))
    new=restricted_problem(extended,aug,target,y,weights,rho,problem='weighted_ls')
    old=measurement({t:Y[t]@old_z for t in Y},float(old_z@G@old_z),target,y,weights,rho)
    new_measure=measurement({t:extended[t]@new['z'] for t in Y},float(new['z']@aug@new['z']),target,y,weights,rho)
    improvement=old['joint_WLS_objective']-new_measure['joint_WLS_objective']
    # Small-matrix objective evaluation roundoff, NOT an empirical gain cutoff.
    scale=abs(old['joint_WLS_objective'])+abs(new_measure['joint_WLS_objective'])+float(y@y)
    roundoff=np.finfo(float).eps*(r+8)**2*scale
    return dict(before=old,after=new_measure,joint_objective_improvement=improvement,
        numerical_comparison_roundoff=roundoff,functionally_resolved=improvement>roundoff,
        target_error_change=new_measure['target_relative_error']-old['target_relative_error'],
        protection_energy_change=new_measure['weighted_protected_energy']-old['weighted_protected_energy'],
        parameter_regularization_change=new_measure['parameter_regularization_cost']-old['parameter_regularization_cost'],
        augmented_coefficients=new['z'])


def refinement_question(signal,certificate,innovation_norm):
    gap=certificate['objective_gap_upper_bound'];gain=signal['joint_objective_improvement']
    source_error=certificate['parameter_error_bound']
    return dict(missing_control_resolved=gain>gap+signal['numerical_comparison_roundoff'],
        source_innovation_resolved=innovation_norm>source_error,
        refinement_priority=gap/(max(0.,gain)+gap+np.finfo(float).tiny)+source_error/(innovation_norm+source_error+np.finfo(float).tiny),
        reference_objective_uncertainty=gap,
        no_unified_relative_error_admission=True)
