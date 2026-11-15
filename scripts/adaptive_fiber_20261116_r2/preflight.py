"""Verify restoration and exact model equality before the r1 service is stopped."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from run_a100 import setup,cached_geometry,write_json
from fiber_core import ConstrainedSolver
from resume_state import restore,digest

def main(checkpoint,out):
    out.mkdir(parents=True,exist_ok=True);c,s,norm=setup(out)
    payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True);geom=cached_geometry(payload['anchor'])
    groups={g['key']:g['indices'] for g in s.groups if g['indices']}
    r=restore(checkpoint,groups,ConstrainedSolver('/home/shangxueyuan/merging/tsv/scripts/conditional_compensation_20261116'),geom.budget_references)
    point=s.point(torch.tensor(r['a'],device='cuda',dtype=torch.float64)).cpu()
    assert torch.equal(point,r['point']),'Restored visual parameters differ from accepted checkpoint'
    assert all(np.array_equal(m.vector,np.array(old['vector'])) and m.map_matrix==old['map_matrix'] for m,old in zip(r['policy'].modes,payload['modes']))
    assert r['next_step']==payload['step']+1 and len(r['policy'].actual_responses)==len(payload['trajectory'])
    audit=dict(passed=True,checkpoint=str(checkpoint),checkpoint_sha256=r['checkpoint_sha256'],
      exact_full_visual_equality=True,parameters=point.numel(),next_step=r['next_step'],restored_modes=len(r['policy'].modes),
      restored_actual_trajectories=len(r['policy'].actual_responses),budget_references=r['references'],time_unix=time.time())
    write_json(out/'resume_verified.json',audit);print(json.dumps(audit))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.checkpoint,a.output)
