"""Read-only audit of saved masks, budgets, seeds and held-data separation."""
import argparse,json
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output
    r=json.loads((out/'report.json').read_text());c=r['config']
    masks=torch.load(out/'private/masks.pt',map_location='cpu',weights_only=True)
    assert len(masks)==len(r['methods'])==12
    assert not set(c['selection_probe_ids'])&set(c['held_ids'])
    assert not set(sum(c['gradient_ids'],[]))&set(c['held_ids']+c['selection_probe_ids'])
    a=json.loads(Path('/media/shangxueyuan/data1/DTD_fixed10_support_20261114/config.json').read_text())
    assert not set(sum(a['probe_ids'],[]))&set(c['held_ids'])
    joint=masks['joint_logdet'].numpy();results={};all_task=0
    joint_counts=next(m['matrix_counts'] for m in r['methods'] if m['method']=='joint_logdet')
    for method in r['methods']:
        name=method['method'];indices=masks[name].numpy();k=c['budget']
        assert len(indices)==len(np.unique(indices))==k and indices.min()>=0 and indices.max()<84934656
        assert sum(method['matrix_counts'])==k
        if name.startswith('matrix_matched'):assert method['matrix_counts']==joint_counts
        cross=len(np.intersect1d(joint,indices,assume_unique=True))
        assert len(method['response_control'])==32 and len(method['task_control'])==6
        assert json.loads((out/'controls'/f'{name}.json').read_text())==method
        for row in method['response_control']:assert row['applied_parameter_norm']<=.010001
        for row in method['task_control']:
            assert abs(row['applied_parameter_norm']-row['radius'])<1e-6
            all_task+=row['metrics']['held']['CE_delta']<0
        results[name]=dict(intersection_fraction_with_joint=cross/k,Jaccard_with_joint=cross/(2*k-cross))
    payload=dict(mask_budget=c['budget'],unique_masks_valid=True,matched_matrix_counts_valid=True,
        held_ids_disjoint_from_both_A_probes=True,response_checks=384,task_checks=72,
        negative_task_CE_checks=int(all_task),overlap=results,
        caveat='Small negative CE changes can be near numerical noise; checks share images and gradients, not72 independent statistical repetitions')
    (out/'mask_audit.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload))


if __name__=='__main__':main()
