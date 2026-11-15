"""Frozen paired validation and actual finite cancellation; no new fit or selection."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import torch


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(path)


def ci(x):
    rng=np.random.default_rng(161130);x=np.asarray(x)
    return np.quantile(x[rng.integers(len(x),size=(1000,len(x)))].mean(1),[.025,.975]).tolist()


def main(out,wait):
    deadline=time.time()+wait*60
    while True:
        if (out/'run_failure.json').exists():raise RuntimeError('Main pilot unresolved; no scientific failure conclusion')
        if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('stage')=='conditional_compensation_complete':break
        if time.time()>deadline:raise TimeoutError('Audit wait budget')
        time.sleep(5)
    models=json.loads((out/'all_candidates_frozen.json').read_text())['models'];lookup={m['id']:m for m in models}
    records=json.loads((out/'finite_results.json').read_text());result=[]
    def load(p,ident,task):return torch.load(out/f'private/finite_p{p}_{ident}_{task}.pt',map_location='cpu',weights_only=True)
    for model in models:
        if model['kind']!='conditional_pair' or not model['id'].endswith('_r1'):continue
        ident=model['id'];prefix=ident[:-3];k=model['anchor']
        comparators=[prefix+'_costmatched_single',prefix+'_4D']
        comparators += [m['id'] for m in models if m['anchor']==k and m['kind'] in ['full304_upper','global2D_local']
                        and m.get('tau')==model['tau'] and abs(m['radius']-model['radius'])<1e-12]
        for p in [4,5]:
            pair={t:load(p,ident,t) for t in ['DTD','Cars']}
            baseline=load(p,f'a{k}_baseline','DTD');driver=load(p,prefix+'_driver_only','DTD');comp=load(p,prefix+'_compensator_only','DTD')
            N=len(baseline['labels']);phi=lambda z:2*z.softmax(-1).sqrt()/np.sqrt(N)
            origin=phi(baseline['logits']);dm=phi(driver['logits'])-origin;dn=phi(comp['logits'])-origin;dj=phi(pair['DTD']['logits'])-origin
            entry=dict(id=ident,anchor=k,probe=p,groups=model['groups'],tau=model['tau'],
                nonlinear_DTD_separate_energy=float((dm*dm).sum()+(dn*dn).sum()),
                nonlinear_DTD_joint_energy=float((dj*dj).sum()),
                finite_DTD_energy_reduction=float((dm*dm).sum()+(dn*dn).sum()-(dj*dj).sum()),
                finite_response_additivity_error=float((dj-dm-dn).norm()/max(float(dj.norm()),1e-30)),
                conditional_pair_control_dimension=model['control_dimension'],comparisons=[])
            for reference in comparators:
                comparison=dict(reference_id=reference,kind=lookup[reference]['kind'])
                for t in ['DTD','Cars']:
                    data=load(p,reference,t);delta=(pair[t]['loss']-data['loss']).numpy()
                    comparison[t]=dict(CE_difference=float(delta.mean()),CI95=ci(delta),
                      accuracy_difference=float((pair[t]['logits'].argmax(-1)==pair[t]['labels']).double().mean()-(data['logits'].argmax(-1)==data['labels']).double().mean()))
                entry['comparisons'].append(comparison)
            result.append(entry);write(out/'paired_compensation_effects.json',dict(points=result,
              no_new_selection=True,individual_CI_not_multiple_testing_corrected=True,
              finite_energy_reduction_includes_nonlinear_interaction_not_only_cross_inner_product=True))
    write(out/'audit_complete.json',dict(done=True,no_extra_model_evaluation=True,pair_probe_records=len(result)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--wait-minutes',type=float,default=480);a=p.parse_args()
    try:main(a.output,a.wait_minutes)
    except Exception:
        write(a.output/'audit_failure.json',dict(traceback=traceback.format_exc(),main_untouched=True));raise
