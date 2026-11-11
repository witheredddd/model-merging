"""Online geometry changes precede and alter subsequent SZa coordinate writes."""
from dataclasses import dataclass,field,replace,asdict
import torch
from manifold import span,weighted_target
from state import Group
from geometry import refresh,coefficient_step_radius,tracking_error
from initialization import initialize
from work import Ledger,MeasuredOracle,BudgetEnd
from steps import StepSettings,coordinate_trial
from evidence import Persistence,actual_responses,responsibilities
from transforms import replace_dictionary,replace_group,add_group,merge_groups,drop_group
from lookahead import LookaheadSettings,compare


@dataclass(frozen=True)
class Settings:
    seed: int = 11
    cold_batches: int = 3
    initial_dictionary_budget: int = 8
    maximum_dictionary_rank: int = 8
    maximum_total_rank: int = 8
    maximum_groups: int = 4
    geometry_every: int = 8
    candidate_batches: int = 3
    persistence_windows: int = 3
    maximum_proposals: int = 4
    gamma: float = 1e-5
    response_weight: float = .001
    step: StepSettings = field(default_factory=StepSettings)
    lookahead: LookaheadSettings = field(default_factory=LookaheadSettings)
    online_dictionary: bool = True
    dynamic_groups: bool = True
    allow_dictionary_growth: bool = False

    def __post_init__(self):
        if self.cold_batches not in [2,3,4] or self.candidate_batches not in [2,3,4]:
            raise ValueError('Cold/online task candidates must use 2-4 independent batches')
        if min(self.initial_dictionary_budget,self.maximum_dictionary_rank,self.maximum_total_rank,self.maximum_groups,self.geometry_every,self.maximum_proposals)<1:
            raise ValueError('Capacity and interval budgets must be positive')
        if self.gamma<=0 or self.response_weight<0:raise ValueError('Invalid damping/response weight')


def lifted_axes(geometry,target):
    Z=geometry.C@target
    U=geometry.A@Z
    _,s,Vt=torch.linalg.svd(U,full_matrices=False)
    if not len(s) or float(s[-1])<1e-8:return None
    return Z@Vt.T@torch.diag(1/s)


class OnlineTrainer:
    def __init__(self,raw,settings=None,budget=1000000):
        self.raw=raw;self.settings=settings or Settings()
        if not 2<=self.settings.candidate_batches<=4 or self.settings.geometry_every<1:
            raise ValueError('Use 2-4 candidate batches and a positive geometry interval')
        if self.settings.lookahead.steps not in [2,3,4]:raise ValueError('Lookahead must use 2-4 batches')
        self.ledger=Ledger(budget);self.oracle=MeasuredOracle(raw,self.ledger)
        self.state,self.geometry,self.initialization=initialize(raw,self.oracle,self.settings.seed,
            self.settings.cold_batches,min(self.settings.initial_dictionary_budget,self.settings.maximum_dictionary_rank),
            self.settings.gamma,maximum_total_rank=self.settings.maximum_total_rank)
        self.radius=coefficient_step_radius(self.geometry)
        self.persistence=Persistence(self.settings.persistence_windows)
        self.iteration=0;self.window=0;self.events=[];self.geometry_events=[]
        self.snapshots=[]
        self.dictionary_replacement_failures=0

    def step(self,sample=None):
        sample=sample or self.raw.sample(self.settings.seed,1000+self.iteration,())
        before=self.state
        self.state,self.radius,event=coordinate_trial(self.state,self.geometry,self.oracle,sample,self.settings.step,self.radius)
        self.iteration+=1
        event.update(iteration=self.iteration,dictionary_version=self.state.dictionary_version,
            changed_parameters=bool(not torch.equal(before.parameters(),self.state.parameters())),
            total_rank=self.state.total_rank,groups=len(self.state.groups),online_images=self.ledger.used)
        self.state.validate();self.events.append(event)
        force=not event['tracking']['safe'] or event['solver'].get('fallback') is not None
        if self.iteration%self.settings.geometry_every==0 or force:
            self.geometry_window()
        return event

    def _samples(self,offset,count,excluded):
        result=[];used=set(excluded)
        for j in range(count):
            sample=self.raw.sample(self.settings.seed,offset+j,used)
            if used.intersection(sample['ids']):raise ValueError('Geometry samples are not independent')
            used.update(sample['ids']);result.append(sample)
        return result,used

    def _rotation(self,state,geometry,evidence,rho,index):
        group=state.groups[index]
        U=span(geometry.actual_U(group))
        rank=min(group.Z.shape[1],geometry.L.shape[1])
        if rank<1 or rank!=group.Z.shape[1] or U.shape[1]<rank:return None
        D=geometry.L.T@evidence['R']
        D=D/D.square().sum(0).mean().sqrt().clamp_min(1e-12)
        old=geometry.L.T@U[:,:rank]
        target,_=weighted_target(D,rho[:,index],rank,old,drift_weight=.001,
                                 response_weight=max(self.settings.response_weight,1e-12))
        Z=lifted_axes(geometry,geometry.L@target)
        if Z is None:return None
        # Soft temporal continuation, not a universal reconstruction-% gate.
        proposal,_=replace_group(state,index,.85*group.Z+.15*Z,geometry.A)
        return proposal

    def _continuity(self,proposed,metric,merge=False):
        if merge:
            first=span(self.geometry.A@self.state.G);second=span(metric@proposed.G)
            return max(0.,first.shape[1]+second.shape[1]-2*float((first.T@second).square().sum()))
        costs=[]
        for old in self.state.groups:
            found=next((g for g in proposed.groups if g.name==old.name),None)
            if found is None:continue
            first=span(self.geometry.actual_U(old));second=span(metric@found.Z)
            costs.append(max(0.,first.shape[1]+second.shape[1]-2*float((first.T@second).square().sum())))
        return sum(costs)

    def _birth(self,state,geometry,R,name):
        existing=span(geometry.A@state.G)
        residual=R-existing@(existing.T@R)
        # Only the currently reachable part is lifted; outside evidence must
        # first change S. A zero/near-kernel matrix is a normal no-proposal case.
        residual=geometry.L@(geometry.L.T@residual)
        u,s,_=torch.linalg.svd(residual,full_matrices=False)
        if not len(s) or float(s[0])<1e-6*max(float(R.norm()),1e-12):return None
        Z=lifted_axes(geometry,u[:,:1])
        if Z is None:return None
        return add_group(state,Z,name,self.settings.maximum_total_rank,self.settings.maximum_groups)

    def geometry_window(self):
        original,geometry,radius=self.state,self.geometry,self.radius
        persistence=list(self.persistence.windows)
        try:
            return self._geometry_window()
        except (ValueError,RuntimeError,BudgetEnd) as error:
            self.state,self.geometry=original,geometry
            self.radius=max(coefficient_step_radius(geometry)*1e-6,radius*.5)
            self.persistence.windows=persistence
            event=dict(window=self.window,reason='geometry_transaction_rolled_back',error=str(error),
                state_preserved=True,online_images=self.ledger.used,proposals=[])
            self.geometry_events.append(event)
            if isinstance(error,BudgetEnd):raise
            return event

    def _geometry_window(self):
        self.window+=1;settings=self.settings
        refresh_cost=self.raw.n*(self.state.S.shape[1]+1)
        if self.ledger.used+refresh_cost>self.ledger.limit:
            event=dict(window=self.window,reason='insufficient_refresh_budget',proposals=[])
            self.geometry_events.append(event);return event
        # Reanchoring updates local q/J/reference_x only. theta0 is unchanged.
        self.geometry=refresh(self.state,self.oracle,settings.gamma)
        self.radius=min(self.radius,coefficient_step_radius(self.geometry)*4)
        excluded=self.persistence.excluded_ids()
        samples,used=self._samples(100000+self.window*100,settings.candidate_batches,excluded)
        checks,used=self._samples(100050+self.window*100,1,used)
        evidence=actual_responses(self.state,self.geometry,self.oracle,samples,checks[0],self.radius)
        event=dict(window=self.window,iteration=self.iteration,proposals=[],local_refresh=True,
            dictionary_version_before=self.state.dictionary_version,groups_before=[g.name for g in self.state.groups])
        if evidence is None:
            event['reason']='no_finite_nonzero_candidates';self.geometry_events.append(event);return event
        rho,responsibility=responsibilities(self.state,self.geometry,evidence)
        persistent=self.persistence.observe(evidence)
        event.update(responsibility=responsibility,evidence=evidence['rows'],persistent_outside=persistent)
        candidates=[]
        if persistent and settings.online_dictionary:
            index=max(range(len(evidence['rows'])),key=lambda j:evidence['rows'][j]['outside_fraction']*
                max(evidence['rows'][j]['independent_candidate_gain'],0.))
            lifted=self.geometry.C@evidence['R']
            importance=lifted@lifted.T
            importance=importance/importance.trace().clamp_min(1e-12)
            contributions=torch.stack([g.Z@g.a for g in self.state.groups],1)
            group_importance=contributions@contributions.T
            importance=importance+.25*group_importance/group_importance.trace().clamp_min(1e-12)
            proposed,info=replace_dictionary(self.state,evidence['directions'][index],settings.maximum_dictionary_rank,importance=importance)
            unrepresentable=proposed is None
            if proposed is None and settings.allow_dictionary_growth:
                proposed,info=replace_dictionary(self.state,evidence['directions'][index],settings.maximum_dictionary_rank,grow=True)
            if proposed is not None:
                new_geometry=refresh(proposed,self.oracle,settings.gamma)
                R=torch.stack([self.oracle.jvp(proposed,v).flatten() for v in evidence['directions']],1).double()
                newborn=self._birth(proposed,new_geometry,R,f'g{self.window}_dictionary') if settings.dynamic_groups else None
                if newborn is not None:proposed=newborn
                else:
                    changed=self._rotation(proposed,new_geometry,dict(R=R),rho,0)
                    if changed is not None:proposed=changed
                info['continuity']=self._continuity(proposed,new_geometry.A)
                kind='dictionary_grow_after_unrepresentable_replacement' if unrepresentable else 'dictionary_replace'
                candidates.append((kind,proposed,info))
            # Growth is a separate, explicitly enabled candidate only after a
            # prior fixed-rank replacement failed despite persistent task evidence.
            if settings.allow_dictionary_growth and self.dictionary_replacement_failures>0:
                grown,info=replace_dictionary(self.state,evidence['directions'][index],settings.maximum_dictionary_rank,grow=True)
                if grown is not None:
                    grown_geometry=refresh(grown,self.oracle,settings.gamma)
                    R=torch.stack([self.oracle.jvp(grown,v).flatten() for v in evidence['directions']],1).double()
                    born=self._birth(grown,grown_geometry,R,f'g{self.window}_growth') if settings.dynamic_groups else None
                    if born is not None:grown=born
                    else:
                        rotated=self._rotation(grown,grown_geometry,dict(R=R),rho,0)
                        if rotated is not None:grown=rotated
                    info['continuity']=self._continuity(grown,grown_geometry.A)
                    candidates.append(('dictionary_grow_after_failed_replacement',grown,info))
        if settings.response_weight>0:
            index=int(rho.sum(0).argmax())
            proposed=self._rotation(self.state,self.geometry,evidence,rho,index)
            if proposed is not None:candidates.append(('rotate_group',proposed,dict(continuity=self._continuity(proposed,self.geometry.A))))
        if settings.dynamic_groups:
            born=self._birth(self.state,self.geometry,evidence['R'],f'g{self.window}_birth')
            if born is not None:candidates.append(('birth_group',born,dict(continuity=self._continuity(born,self.geometry.A))))
            merged=merge_groups(self.state)
            if merged is not None:candidates.append(('merge_groups',merged,dict(continuity=self._continuity(merged,self.geometry.A,merge=True))))
            if len(self.state.groups)>1:
                index=min(range(len(self.state.groups)),key=lambda m:float((self.state.groups[m].Z@self.state.groups[m].a).norm()))
                candidates.append(('retire_group',drop_group(self.state,index),{}))
        fits,used=self._samples(200000+self.window*100,settings.lookahead.steps,used)
        validations,used=self._samples(200050+self.window*100,1,used)
        original=self.state;best=None;best_score=-float('inf');baseline_end=None
        best_geometry=None;baseline_geometry=None
        for kind,proposed,metadata in candidates[:settings.maximum_proposals]:
            if proposed is None:continue
            displacement=float((proposed.parameters()-original.parameters()).norm())
            if displacement>self.radius:
                event['proposals'].append(dict(kind=kind,accepted=False,reason='parameter_step_trust_region',move=displacement));continue
            try:
                selected,comparison=compare(original,proposed,self.raw,self.ledger,fits,validations[0],
                    settings.step,settings.lookahead,continuity=metadata.get('continuity',0.), gamma=settings.gamma)
                selected_geometry=comparison.pop('_selected_geometry')
                comparison.update(kind=kind,metadata=metadata)
                event['proposals'].append(comparison)
                if comparison['accepted'] and comparison['net_gain']>best_score:
                    best,best_score,best_geometry=selected,comparison['net_gain'],selected_geometry
                    event['selected_operation']=kind
                elif not comparison['accepted'] and baseline_end is None:
                    baseline_end,baseline_geometry=selected,selected_geometry
            except BudgetEnd:
                event['proposals'].append(dict(kind=kind,accepted=False,reason='insufficient_matched_lookahead_budget'))
                break
            except (ValueError,RuntimeError) as error:
                event['proposals'].append(dict(kind=kind,accepted=False,reason='transaction_rolled_back',error=str(error)))
        selected_state=(best if best is not None else baseline_end) or original
        selected_state.validate()
        selected_geometry=(best_geometry if best is not None else baseline_geometry) or self.geometry
        if self.ledger.used+self.raw.n*(selected_state.S.shape[1]+1)<=self.ledger.limit:
            renewed=refresh(selected_state,self.oracle,settings.gamma)
            check=tracking_error(selected_geometry,selected_state,renewed.q,absolute=settings.step.local_absolute,relative=settings.step.local_relative)
            event['full_local_tracking']=check
            if not check['safe']:self.radius*=.5
            selected_geometry=renewed
        else:event['refresh_deferred_budget']=True
        # Commit only after every required validation has completed. Reporters
        # and other observers never see an unvalidated transactional model.
        self.state,self.geometry=selected_state,selected_geometry
        self.radius=min(self.radius,coefficient_step_radius(self.geometry)*4)
        event.update(dictionary_version_after=self.state.dictionary_version,groups_after=[g.name for g in self.state.groups],
            total_rank=self.state.total_rank,online_images=self.ledger.used,
            actual_joint_parameter_rank=int(torch.linalg.matrix_rank(self.state.G,rtol=1e-8)),
            actual_joint_functional_rank=int(torch.linalg.matrix_rank(self.geometry.A@self.state.G,rtol=1e-6)),
            global_base_unchanged=bool(torch.equal(self.state.theta0,self.raw.theta0)))
        self.geometry_events.append(event)
        replacements=[p for p in event['proposals'] if p.get('kind')=='dictionary_replace']
        if replacements:
            self.dictionary_replacement_failures=0 if any(p.get('accepted') for p in replacements) else self.dictionary_replacement_failures+1
        return event
