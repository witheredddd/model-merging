"""Anchor-tagged functional demands; sources never prescribe parameter supports."""
from dataclasses import dataclass,field
import hashlib
import numpy as np


def product_sphere_log(base,endpoint):
    """Per-image Log for the existing 2/sqrt(N) square-root probability embedding."""
    x=np.asarray(base,dtype=np.float64)
    z=np.asarray(endpoint,dtype=np.float64)
    if x.shape!=z.shape or x.ndim!=2:
        raise ValueError('Expected aligned sample-by-class arrays')
    r=2/np.sqrt(len(x))
    nx=np.linalg.norm(x,axis=1,keepdims=True)
    nz=np.linalg.norm(z,axis=1,keepdims=True)
    if np.any(nx==0) or np.any(nz==0):raise ValueError('Zero probability embedding')
    ux=x/nx;uz=z/nz
    cos=np.clip(np.sum(ux*uz,axis=1,keepdims=True),-1.,1.)
    tangent=uz-cos*ux
    sine=np.linalg.norm(tangent,axis=1,keepdims=True)
    angle=np.arctan2(sine,cos)
    if np.any((sine==0)&(cos<0)):
        raise ValueError('Antipodal Log is not uniquely defined')
    factor=np.zeros_like(sine)
    np.divide(angle,sine,out=factor,where=sine!=0)
    return r*factor*tangent


def anchor_key(g):
    point=g.tensor_to_numpy(g.point)
    return hashlib.sha256(memoryview(np.ascontiguousarray(point)).cast('B')).hexdigest()


@dataclass(frozen=True)
class FunctionalDemand:
    name:str
    task:str
    anchor:str
    probe:int
    response:object
    origin:str
    reachable_by_construction:bool
    metadata:dict=field(default_factory=dict)


def ce_demand(g,task,name):
    y,gradient,reference,info=g.ce_unit_target(task)
    if y is None:return None,info
    demand=FunctionalDemand(name,task,anchor_key(g),g.probe,y,
                    'CE unit-gradient functional response',True,
                    dict(info,source_is_one_demand_not_entire_family=True,
                         output_coordinates_id=g.output_coordinate_ids[task]))
    return demand,dict(gradient=gradient,reference=reference)


def motion_demand(g,task,full_motion,name,source_description):
    if len(full_motion)!=g.dimension:raise ValueError('Full parameter motion required')
    return FunctionalDemand(name,task,anchor_key(g),g.probe,g.jvp(task,full_motion),
                'Current-anchor linearized full motion',True,
                dict(source=source_description,expert_difference_is_not_a_search_basis=True,
                     output_coordinates_id=g.output_coordinate_ids[task]))


def expert_log_demand(g,task,expert_point,name):
    if expert_point.numel()!=g.dimension:raise ValueError('Complete expert parameter point required')
    expert_point=expert_point.to(device=g.device,dtype=g.point.dtype)
    current,labels=g.c.baseline(task,g.probe,g.point)
    expert,expert_labels=g.c.baseline(task,g.probe,expert_point)
    if not np.array_equal(np.asarray(labels),np.asarray(expert_labels)):
        raise ValueError('Sample order differs between current model and expert')
    N=len(labels)
    x=(2*current.double().softmax(-1).sqrt()/np.sqrt(N)).cpu().numpy()
    z=(2*expert.double().softmax(-1).sqrt()/np.sqrt(N)).cpu().numpy()
    y=product_sphere_log(x,z).ravel()
    return FunctionalDemand(name,task,anchor_key(g),g.probe,g.numpy_to_tensor(y),
                    'Expert functional Log at current model',False,
                    dict(log_base='Current model, same as Jacobian anchor',
                         image_tangent_not_necessarily_parameter_Jacobian_range=True,
                         output_coordinates_id=g.output_coordinate_ids[task],
                         preserved_CE_descent_claim=False))


class DemandFamily:
    def __init__(self,geometry):
        self.geometry=geometry
        self.anchor=anchor_key(geometry)
        self.demands=[]

    def analysis_view(self,demand):
        raw=self.geometry.tensor_to_numpy(demand.response)
        magnitude=float(np.linalg.norm(raw))
        return dict(raw_response_norm=magnitude,
                    unit_response_analysis_only=raw.copy()/magnitude if magnitude else None,
                    raw_demand_preserved=True,aggregation_policy=None,
                    reachability_source_status=('reachable_by_construction' if
                         demand.reachable_by_construction else 'reachability_unverified'))

    def append(self,demand):
        if demand is None:raise ValueError('Undefined target must be recorded, not substituted')
        if (demand.anchor!=self.anchor or self.anchor!=anchor_key(self.geometry)
                or demand.probe!=self.geometry.probe):
            raise ValueError('Demand and Jacobian refer to different anchors/probes')
        if demand.response.numel()!=self.geometry.output_dimensions[demand.task]:
            raise ValueError('Demand output dimensions differ')
        if demand.metadata.get('output_coordinates_id')!=self.geometry.output_coordinate_ids[demand.task]:
            raise ValueError('Sample/head/embedding coordinate identity differs')
        self.demands.append(demand)

    def solve(self,condition,demand,*,formulation,relative_tolerance,
              max_iterations,output_weight=None,**numerical_options):
        if (demand.anchor!=self.anchor or demand.anchor!=anchor_key(self.geometry)
                or demand.probe!=self.geometry.probe or demand.task!=condition.target):
            raise ValueError('Condition, demand and current Jacobian do not align')
        if condition.geometry is not self.geometry:
            raise ValueError('Conditional solver uses a different geometry instance')
        if demand.metadata.get('output_coordinates_id')!=self.geometry.output_coordinate_ids[demand.task]:
            raise ValueError('Demand output coordinates do not match current Jacobian')
        if formulation=='exact_response':
            v,audit=condition.solve_exact_response(demand.response,
                        relative_tolerance=relative_tolerance,max_iterations=max_iterations,
                        **numerical_options)
        elif formulation=='weighted_fit':
            if output_weight is None:raise ValueError('Output metric W must be explicit')
            v,audit=condition.solve_response_least_squares(demand.response,output_weight,
                        relative_tolerance=relative_tolerance,max_iterations=max_iterations,
                        **numerical_options)
        else:raise ValueError('Formulation must be explicit; no automatic fallback')
        audit.update(demand_name=demand.name,demand_origin=demand.origin,
                     reachable_by_construction=demand.reachable_by_construction,
                     numerical_target_feasibility=bool(audit.get('target_constraint_numerically_satisfied',False)),
                     exact_feasible_is_only_a_numerical_tolerance_statement=True,
                     unverified_reachability_is_not_certified_by_solver_stop=True)
        return v,audit
