"""Refine only linear support solves; preserve the first solver audit."""
import json,hashlib
import numpy as np
from run import OLD,write,linear_support,root,spectrum
from pathlib import Path

def main(out):
    rows=json.loads((out/'programs.json').read_text())
    if not (out/'first_solver_attempt.json').exists():write(out/'first_solver_attempt.json',rows)
    repaired=[]
    for k in range(3):
        with np.load(OLD/f'private/anchor{k}/geometry.npz') as f:KD=f['K_DTD'];KC=f['K_Cars'];hc=f['h_Cars'];hd=f['h_DTD']
        LD=root(KD);LC=root(KC)
        state=__import__('torch').load('/media/shangxueyuan/data1/compromise_geometry_20261115/private/space.pt',map_location='cpu',mmap=True,weights_only=True)
        G=np.linalg.qr(np.stack([state['a_D'].numpy(),state['a_C'].numpy()],1))[0]
        selected=json.loads((OLD/f'anchor{k}_selected_pair_lock.json').read_text())['selected'];Ws={'full304':np.eye(304),'global2D':G}
        for j,r in enumerate(selected):
            if r['kind']=='conditional_pair':
                W=np.zeros((304,2));W[r['ids']]=np.array(r['W']);Ws[f'pair_{j}']=W
        rr=[r for r in rows if r['anchor']==k and r['space']!='best_cached_conditional_plane']
        for r in rr:
            if r['problem'].startswith('linear'):
                W=Ws[r['space']];x,a=linear_support(W.T@KD@W,W.T@hc,r['radius'],r['epsilon']);v=W@x
                r.update(v=v.tolist(),solver=a,linear_Cars_gain=float(-hc@v),GN_Cars_gain=float(-hc@v-.5*v@KC@v),
                   source_response=float(np.linalg.norm(LD@v)),Cars_response=float(np.linalg.norm(LC@v)),parameter_norm=float(np.linalg.norm(v)),DTD_CE_slope=float(hd@v))
        repaired+=rr
        for r in [r for r in rr if r['space']=='full304']:
            cs=[s for s in rr if s['space'].startswith('pair_') and s['solver']['passed'] and all(s[key]==r[key] for key in ['tau','factor','problem'])]
            best=min(cs,key=lambda s:s['solver']['predicted_Cars_CE'])
            repaired.append({**best,'space':'best_cached_conditional_plane','selected_plane':best['space']})
    gaps=[];banks=[]
    for k in range(3):
        with np.load(OLD/f'private/anchor{k}/geometry.npz') as f:LD=root(f['K_DTD']);LC=root(f['K_Cars'])
        full=[r for r in repaired if r['anchor']==k and r['space']=='full304' and r['solver']['passed']]
        B=np.stack([np.array(r['v'])/r['radius'] for r in full],1)
        for name,A in [('parameter',np.eye(304)),('DTD_response',LD),('Cars_response',LC),('joint_response',np.r_[LD,LC])]:
            banks.append(dict(anchor=k,metric=name,spectrum=spectrum(A@B),scope='Sampled single-Cars-objective optima, not entire reachable-set dimension'))
        for r in full:
            case={key:r[key] for key in ['anchor','tau','factor','problem']}
            for s in repaired:
                if s['space'] not in ['global2D','best_cached_conditional_plane'] or not all(s[key]==v for key,v in case.items()):continue
                gain='linear_Cars_gain' if r['problem'].startswith('linear') else 'GN_Cars_gain'
                gaps.append(dict(**case,reference=s['space'],full_gain=r[gain],reference_gain=s[gain],gain_fraction=s[gain]/r[gain],full_solver_passed=True,reference_solver_passed=s['solver']['passed']))
    write(out/'programs.json',repaired);write(out/'capacity_gaps.json',gaps);write(out/'optimal_motion_spectra.json',banks)
    record=json.loads((out/'complete.json').read_text());record.update(program_records=len(repaired),failed=sum(not r['solver']['passed'] for r in repaired),linear_support_refined_with_scalar_dual=True,original_failed_records_preserved=True)
    write(out/'complete.json',record)

if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
