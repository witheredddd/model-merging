"""Verify this snapshot and reproduce completed-step functional errors."""
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parents[1]/'scripts/self_fiber_flow_20261119'))
from numerics import sphere_log

def read_json(name):
    with gzip.open(root/name,'rt',encoding='utf8') as f:return json.load(f)

def main():
    meta=json.loads((root/'MANIFEST.json').read_text(encoding='utf8'))
    for entry in meta['files']:
        p=(root/entry['file']).resolve()
        if root not in p.parents:raise ValueError('Unsafe path')
        if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Hash mismatch: '+entry['file'])
    state=read_json('snapshot.json.gz');rows=[]
    for name in state['transitions']:
        r=read_json(name+'.gz');h=r['accepted_step_size'];errors={}
        for label,key,expected in (
            ('learning','functional_arrays',r['trials'][-1]['relative_Log_JVP_error']),
            ('heldout','independent_functional_arrays',r['independent_Log_JVP_error'])):
            with np.load(root/r[key],allow_pickle=False) as z:
                actual=sphere_log(z['base_phi'],z['next_phi']).reshape(-1)
                np.testing.assert_allclose(actual,z['actual_Log'],rtol=1e-10,atol=1e-13)
                pred=h*z['JVP']
                error=float(np.linalg.norm(actual-pred)/np.linalg.norm(pred))
                np.testing.assert_allclose(error,expected,rtol=1e-10,atol=1e-13)
                errors[label]=error
        rows.append(dict(path=r['path_id'],step=r['step'],**errors))
    print(json.dumps(dict(verified_files=len(meta['files']),generation=meta['generation'],
        completed_transitions=rows,pending_is_not_complete=bool(state.get('pending_transition'))),indent=2))

if __name__=='__main__':main()
