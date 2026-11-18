"""Rebuild a sampled-coordinate protection graph from the published data."""
import argparse,gzip,json,sys
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--position',type=int,required=True)
p.add_argument('--target',required=True)
p.add_argument('--drivers',required=True,help='Comma-separated actual region IDs, not a mode-count rule')
p.add_argument('--demand',type=Path,help='Optional matching position/probe/task demand NPZ (key y)')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--code-dir',type=Path,help='Optional location of protective_mechanism source')
a=p.parse_args();root=Path(__file__).resolve().parent;repo=root.parents[1]
sys.path.insert(0,str(a.code_dir or repo/'scripts/protective_mechanism_20261118'))
from graphs import mechanism_report
from io_state import plain
with gzip.open(root/f'mechanism/position{a.position}.json.gz','rt',encoding='utf8') as f:r=json.load(f)
cfg=json.loads((root/'numeric_configuration.json').read_text(encoding='utf8'))
rows=[];norms=[];ids=r['region_ids']
for region in ids:
    with np.load(root/f'run/regional_coupling/p{a.position}/{region:05d}.npz',allow_pickle=False) as z:
        rows.append({k:z[k].copy() for k in z.files if k!='parameter_norm_squared'})
        norms.append(float(z['parameter_norm_squared']))
Y={t:np.column_stack([row[t] for row in rows]) for t in rows[0]};weights={t:1. for t in Y if t!=a.target}
rho=cfg['gamma']*sum(cfg['scales']['scales'][t] for t in weights)
requested=list(map(int,a.drivers.split(',')));drivers=[ids.index(i) for i in requested]
if a.demand:
    with np.load(a.demand,allow_pickle=False) as z:y=z['y'].copy()
else:y=Y[a.target]@np.ones(len(ids))
if y.shape!=(Y[a.target].shape[0],):raise ValueError('Demand shape does not match task coordinates')
report=mechanism_report(Y,np.diag(norms),a.target,y,weights,rho,drivers)
report.update(position=a.position,actual_region_ids=ids,driver_region_ids=requested,rho=rho,
    target_source=str(a.demand) if a.demand else 'Observed regional response sum; not the original requested task demand',
    scope='Restricted region-motion directions, not complete parameter regions or automatic grouping')
a.output.write_text(json.dumps(plain(report),ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(output=str(a.output),rho=rho,driver_regions=requested)))
