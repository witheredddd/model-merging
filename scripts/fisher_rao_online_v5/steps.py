"""Fixed-cost coordinate trials used both online and in matched lookahead."""
from dataclasses import dataclass,replace
import math
import torch
from geometry import fr_preconditioned_direction,tracking_error,coefficient_step_radius


@dataclass(frozen=True)
class StepSettings:
    learning_rate: float = .05
    noise_allowance: float = 1e-4
    local_absolute: float = 2e-4
    local_relative: float = .15
    condition_limit: float = 1e8
    monitor_count: int = 32

    def __post_init__(self):
        if not all(math.isfinite(v) and v>=0 for v in [self.learning_rate,self.noise_allowance,self.local_absolute,self.local_relative,self.condition_limit]):
            raise ValueError('Coordinate controls must be finite and nonnegative')
        if self.learning_rate<=0 or self.condition_limit<1 or self.monitor_count<1:
            raise ValueError('Invalid coordinate learning rate, condition limit or monitor size')


def coordinate_trial(state,geometry,oracle,sample,settings,radius):
    """Always one gradient, one candidate CE and one monitor prediction.

    No full-dimensional optimizer write. A rejected trial preserves the whole
    model; rejection is an event, not a global stopping rule.
    """
    before,gradient=oracle.gradient(state,sample)
    finite=math.isfinite(before) and bool(torch.isfinite(gradient).all() and torch.isfinite(gradient.norm()))
    info=dict(fallback='nonfinite_gradient')
    proposed=state
    predicted=0.
    if finite:
        gx=(state.S.T@gradient.to(state.S.dtype)).double()
        delta,info=fr_preconditioned_direction(state,geometry,gx,condition_limit=settings.condition_limit)
        da=settings.learning_rate*delta
        dx=state.G@da
        norm=float(dx.norm())
        if norm>radius:da=da*(radius/norm)
        if not torch.isfinite(da).all():finite=False
        else:
            proposed=state.with_coordinates(state.a+da)
            predicted=-float(gx@(proposed.x-state.x))
            if not math.isfinite(predicted):finite=False;proposed=state
    after=oracle.loss(proposed,sample)
    positions=torch.arange(min(settings.monitor_count,len(geometry.q)),device=geometry.q.device)
    actual=oracle.phi(proposed,positions)
    check=tracking_error(geometry,proposed,actual,positions,settings.local_absolute,settings.local_relative)
    allowance=settings.noise_allowance*max(1.,abs(before)) if math.isfinite(before) else 0.
    safe_ce=finite and math.isfinite(after) and after<=before+allowance
    accepted=safe_ce and check['safe'] and predicted>=-1e-12
    gain=before-after
    ratio=gain/max(abs(predicted),1e-12) if finite and math.isfinite(after) else 0.
    if accepted:
        new_radius=radius*(1.25 if ratio>.75 else (.7 if gain<0 else 1.))
    else:new_radius=radius*.5
    minimum=coefficient_step_radius(geometry)*1e-6
    maximum=coefficient_step_radius(geometry)*4
    new_radius=max(minimum,min(maximum,new_radius))
    selected=proposed.validate() if accepted else state
    dx=selected.x-state.x;vertical=dx-geometry.V@(geometry.V.T@dx)
    return selected,new_radius,dict(accepted=accepted,loss_before=before,loss_after=after,
        predicted_gain=predicted,actual_gain=gain,ratio=ratio,tracking=check,
        reason='accepted' if accepted else ('local_model_error' if not check['safe'] else 'task_or_numerical_control'),
        new_increment_near_null_energy=float(vertical.square().sum()),solver=info)
