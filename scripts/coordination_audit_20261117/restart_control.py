"""Isolate CG restart from any benefit of a learned correction."""
import json
import shutil
import time

import numpy as np
import torch

from run_audit import (SOURCE, OUT, HERE, CASE, load_json, load_state, cg_from,
                       load_eight_geometry, SchurSession, FullFunctionalGeometry,
                       evaluate, atomic_json, plain, digest, relative, Profiler)


def run():
    report=load_json(OUT/'REPORT.json')
    assert report['stage']=='complete'
    protocol=load_json(SOURCE/'PROTOCOL.json')
    g,identity=load_eight_geometry(SOURCE/'runtime',protocol['inventory'],protocol['checkpoint'])
    weights={t:1. for t in g.tasks if t!='SUN397'}
    scales=load_json(SOURCE/'SCALES.json')['scales']
    rho=.1*sum(scales[t] for t in weights)
    with np.load(SOURCE/'PRECONDITIONER.npz') as z:
        pd=np.concatenate([1+z[t]/rho for t in weights]);td=np.where(z['SUN397']>0,z['SUN397']/rho,1.)
    y=g.numpy_to_tensor(np.load(SOURCE/'demands'/CASE/'response.npy'))
    session=SchurSession(g,'SUN397',weights,rho,y,pd,td)
    cg=cg_from(load_state(report['inputs']['14']))
    cg['p']=cg['r'].copy()
    torch.cuda.synchronize();start=time.perf_counter()
    _,_,final=session.step(cg,protocol['solver_rtol'],1,None)
    torch.cuda.synchronize()
    row=dict(CG_one_step_seconds=time.perf_counter()-start,
             role='Same historical step14 x and r, p reset to r, no learned correction',
             structure_count=0,source_state_sha256=digest(report['inputs']['14']))
    row['audit']=session.audit(final)
    fresh=session.b-session.apply(final['x'])
    row.update(recursive_true_relative_difference=relative(fresh-final['r'],session.b),
               full_scaled_schur_relative_residual=relative(fresh,session.b),
               protected_energy_sum=sum(row['audit']['protected_response_energies'].values()))
    np.savez(OUT/'restart_only_cg.npz',**final)
    atomic_json(OUT/'RESTART_CONTROL.json',plain(dict(stage='linear_complete',result=row)))
    print('restart_linear_complete',flush=True)
    held=FullFunctionalGeometry(g.c,g.point,1,precision='float64')
    motion,_=session.materialize(final)
    row['finite']=evaluate(held,motion)
    report['branches']['restart_only']=row
    report['restart_control_added']=True
    atomic_json(OUT/'RESTART_CONTROL.json',plain(dict(stage='complete',result=row,geometry_identity=identity)))
    atomic_json(OUT/'REPORT.json',plain(report))
    target=OUT/'executed_source'/'restart_control.py'
    shutil.copy2(HERE/'restart_control.py',target)
    atomic_json(OUT/'executed_source'/'SOURCE_HASHES.json',{p.name:digest(p) for p in (OUT/'executed_source').glob('*.py')})
    print('restart_control_complete',flush=True)


if __name__=='__main__':
    run()
