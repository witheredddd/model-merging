"""Pure CPU lift-cost diagnostic using already saved responses; no new models."""
from pathlib import Path
import json
import numpy as np

def main(out):
    p=Path('/media/shangxueyuan/data1/anchor0_coevolution_20261116')
    r=Path('/media/shangxueyuan/data1/adaptive_fiber_20261116_r2/archived_other_anchors/anchor0/adaptive')
    states=json.loads((p/'CONTROL_STATE.json').read_text());run=json.loads((r/'FAST_FINAL.json').read_text());lock=json.loads((r/'fast_lock.json').read_text())
    with np.load(p/'actual_trajectory_p0.npz') as f:Y=f['joint_anchor_scaled']
    rows=[]
    for t in range(1,5):
        n=len(states[t-1]['modes']);B=np.array([m['vector'] for m in states[t]['modes']]).T[:,:n]
        with np.load(p/f'mode_response_step{t}.npz') as f:
            A=f['Psi_anchor_scaled'][:,:n];z=np.linalg.lstsq(A,Y[:,t],rcond=1e-10)[0];v=B@z
            target=run['trajectory'][t]['scenario']['target'];tau=run['trajectory'][t]['scenario']['tau']
            src=f['J_D_B' if target=='Cars' else 'J_C_B'][:,:n]@z
            rows.append(dict(step=t,target=target,tau=tau,old_modes=n,residual=float(np.linalg.norm(Y[:,t]-A@z)/np.linalg.norm(Y[:,t])),
              radius_ratio=float(np.linalg.norm(v)/lock['R']),protected_response_budget_ratio=float(np.linalg.norm(src)/(lock['R']*tau*lock['source_references'][target]))))
    (out/'projection_cost_check.json').write_text(json.dumps(dict(scope='Unconstrained old-mode response projection, diagnostic only; no CE-sign or finite-feasibility certification',rows=rows),indent=2))

if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
