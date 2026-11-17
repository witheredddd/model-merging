"""Persistent fair task/source dispatch. Numerical certification never admits a source."""
from collections import deque


FAMILIES=('CE','expert_motion_JVP','class_CE')


def build_catalog(tasks,classes,gammas):
    cases={};strata=[]
    for task in tasks:
        for family in FAMILIES:
            specs=([dict(task=task,source=family,label=int(c)) for c in classes[task]]
                   if family=='class_CE' else [dict(task=task,source=family)])
            ids=[]
            for spec in specs:
                name=f"{task}_{family}"+(f"_{spec['label']}" if 'label' in spec else '')
                for gamma in gammas:
                    case_id=f'{name}_gamma{gamma:g}'
                    cases[case_id]=dict(id=case_id,spec=dict(spec,name=name),gamma=gamma,
                        status='candidate',iterations=0,cg_checkpoint=None,inner_rtol=None,
                        visits=0,certified=False)
                    ids.append(case_id)
            strata.append(dict(task=task,family=family,new=ids,retry=[],next='new'))
    return dict(cursor=0,strata=strata,cases=cases,inflight=None,dispatches=0)


def take(state):
    if state['inflight'] is not None:
        # A process crash before a committed CG iteration replays this dispatch.
        return state['inflight']
    strata=state['strata']
    for _ in strata:
        index=state['cursor'];state['cursor']=(index+1)%len(strata)
        group=strata[index]
        first=group['next'];second='retry' if first=='new' else 'new'
        lane=first if group[first] else second
        if not group[lane]:continue
        case_id=group[lane].pop(0)
        group['next']='retry' if lane=='new' else 'new'
        state['inflight']=dict(case_id=case_id,stratum=index)
        state['dispatches']+=1
        return state['inflight']
    return None


def finish(state,*,resolved):
    item=state['inflight']
    if item is None:raise ValueError('No active dispatch')
    if not resolved:state['strata'][item['stratum']]['retry'].append(item['case_id'])
    state['inflight']=None


def mark_migrated_active(state,case_id,checkpoint,iterations):
    case=state['cases'][case_id]
    case.update(cg_checkpoint=checkpoint,iterations=iterations,status='provisional')
    for group in state['strata']:
        if case_id in group['new']:
            group['new'].remove(case_id);group['retry'].append(case_id)
            return
    raise ValueError('Migrated case is absent from catalog')
