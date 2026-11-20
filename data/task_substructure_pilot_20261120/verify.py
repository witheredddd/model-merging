"""Verify complete-pilot metrics and selected raw arrays without the model."""
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent;repo=root.parents[1]
sys.path.insert(0,str(repo/'scripts/task_substructure_20261120'))
sys.path.insert(0,str(repo/'scripts/self_fiber_flow_20261119'))
from math_core import class_weights,metric_stats
from numerics import sphere_log

def main():
    meta=json.loads((root/'MANIFEST.json').read_text(encoding='utf8'))
    for row in meta['files']:
        path=(root/row['file']).resolve()
        if root not in path.parents:raise ValueError('Unsafe manifest path')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Hash mismatch: '+row['file'])
    cases=0;cells=0;records=0
    for path in sorted((root/'evaluations').glob('*.json.gz')):
        with gzip.open(path,'rt',encoding='utf8') as f:r=json.load(f)
        records+=1;file=root/r['arrays']
        if not file.exists():continue
        cases+=1
        with np.load(file,allow_pickle=False) as data:
            for task in ('DTD','Cars'):
                for probe in (0,2,3):
                    key=f'{task}_p{probe}';z=data[key+'_logits'];base=data[key+'_baseline'];y=data[key+'_labels']
                    weights=class_weights(z.shape[1],r['demand'] if task=='DTD' else 'mean')
                    actual=metric_stats(z,y,weights,base);expected=r['metrics'][key]
                    for metric in ('CE','weighted_CE','Top1','delta_CE','delta_weighted_CE','delta_Top1','prediction_flip_rate','probability_L2','margin_mean'):
                        np.testing.assert_allclose(actual[metric],expected[metric],rtol=1e-10,atol=1e-12)
                    pred=data[key+'_predicted_response'];log=data[key+'_actual_Log'];den=np.linalg.norm(pred)
                    if den:np.testing.assert_allclose(np.linalg.norm(log-pred)/den,expected['relative_Log_JVP_error'],rtol=1e-10,atol=1e-12)
                    def phi(logits):
                        a=logits-logits.max(1,keepdims=True);p=np.exp(a);p/=p.sum(1,keepdims=True)
                        return 2*np.sqrt(p)/np.sqrt(len(y))
                    np.testing.assert_allclose(sphere_log(phi(base),phi(z)).reshape(-1),log,rtol=1e-9,atol=1e-12)
                    cells+=1
    if records!=meta['evaluations']:raise ValueError('Case count mismatch')
    print(json.dumps(dict(verified_files=len(meta['files']),all_case_records=records,recomputed_array_cases=cases,recomputed_task_probe_cells=cells,
        large_campaign_included=False),indent=2))

if __name__=='__main__':main()
