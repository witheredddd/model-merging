"""A common WLS objective, explicit finite-reference errors, and solver budgets."""
import numpy as np
from bootstrap import R3
from control_math import restricted_problem,quadratic_solve


def error_budget(cert,norm,function_scale,relative):
    e=cert['parameter_error_bound'];lower=max(0.,norm-e)
    return dict(ready=bool(e<=relative*lower and cert['joint_energy_error_bound']<=relative*function_scale),
        parameter_relative_error_upper=(e/lower if lower else None),
        functional_relative_error_upper=cert['joint_energy_error_bound']/max(function_scale,np.finfo(float).tiny),
        requested_relative_budget=relative,uses_fresh_normal_residual=True)


def measurement(responses,norm2,target,y,weights,rho):
    y=np.asarray(y);residual=responses[target]-y
    energy={t:float(responses[t]@responses[t]) for t in weights}
    leakage=sum(weights[t]*energy[t] for t in weights)
    regularization=rho*norm2
    return dict(target_relative_error=float(np.linalg.norm(residual)/max(np.linalg.norm(y),np.finfo(float).tiny)),
        protected_energies=energy,weighted_protected_energy=leakage,
        parameter_norm_squared=float(norm2),parameter_regularization_cost=float(regularization),
        joint_WLS_objective=float(.5*(residual@residual+leakage+regularization)))


def compare(Y,G,target,y,weights,rho,full,fullnorm,certificate,gain=1.):
    """Primary evaluation uses the undamped restricted optimum, never its gain."""
    if G.shape[0]:
        solution=restricted_problem(Y,G,target,y,weights,rho,problem='weighted_ls')
        z=solution['z']
    else:z=np.empty(0)
    response={t:Y[t]@z for t in Y};applied={t:a*gain for t,a in response.items()}
    restricted=measurement(response,float(z@G@z),target,y,weights,rho)
    executed=measurement(applied,float(z@G@z)*gain**2,target,y,weights,rho)
    reference=measurement(full,fullnorm,target,y,weights,rho)
    gap=restricted['joint_WLS_objective']-reference['joint_WLS_objective']
    bound=certificate['objective_gap_upper_bound']
    repr_error={t:float(np.linalg.norm(response[t]-full[t])/max(np.linalg.norm(full[t]),np.finfo(float).tiny)) for t in full}
    return dict(full_candidate=reference,undamped_restricted_optimum=restricted,damped_observation=executed,
        e_full=reference['target_relative_error'],e_repr=repr_error[target],
        e_demand=restricted['target_relative_error'],e_applied=executed['target_relative_error'],
        representation_error_definition='Response difference of the WLS-optimal restricted motion; not a separate response-only projection optimum',
        per_task_response_difference=repr_error,undamped_coefficients=z,
        objective_difference_from_finite_reference=gap,
        objective_excess_vs_full_optimum_interval=[max(0.,gap),max(0.,gap+bound)],
        reference_certificate=certificate,reference_is_not_asserted_exact=True,
        gain=gain,primary_space_metric_is_undamped=True)


def projection(G,rhs,norm2):
    z=quadratic_solve(G,rhs) if len(rhs) else np.empty(0)
    residual2=max(0.,norm2-2*z@rhs+z@G@z)
    return dict(coefficients=z,relative_error=float(np.sqrt(residual2/max(norm2,np.finfo(float).tiny))),
        explained_Euclidean_energy=float(1-residual2/max(norm2,np.finfo(float).tiny)))


def condition_grid(tasks,classes=None):
    conditions=[dict(id=f'{task}_{family}',target=task,family=family)
        for family in ('normal_response','expert_response') for task in tasks]
    # Two global demands per task are starting probes, not a rank ceiling.
    # Further class-conditional functional demands use the actual sample labels.
    if classes is not None:
        for family in ('normal_response','expert_response'):
            for task in tasks:
                for label in classes[task]:
                    conditions.append(dict(id=f'{task}_{family}_class{label}',target=task,
                        family=family,class_label=int(label)))
    return conditions
