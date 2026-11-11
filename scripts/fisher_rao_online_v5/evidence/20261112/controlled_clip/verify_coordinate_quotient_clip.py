"""Read-only real-checkpoint regression; pass the fixed geometry file as argv[1]."""
from pathlib import Path
from dataclasses import replace
import sys,json,hashlib,importlib.util
import torch
root=Path('/media/shangxueyuan/data1/fisher_rao_online_v5_DTD_controlled_seed11_20261112')
source=Path('/home/shangxueyuan/merging/tsv/scripts/fisher_rao_online_v5_3a57857')
sys.path.insert(0,str(source))
from clip_runtime import ClipTask
from state import ModelState,Group
from geometry import refresh,fr_preconditioned_direction as original_direction
from work import Ledger,MeasuredOracle
fixed=Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('fixed_geometry',fixed)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
qa=root/'posthoc_quotient';qa.mkdir(exist_ok=True)
raw=ClipTask('/home/shangxueyuan/merging/tsv',qa,128,32)
stored=torch.load(root/'final_state.pt',map_location='cpu',weights_only=True,mmap=True)
if stored['base_sha256']!=raw.rt.input['base_sha256']:raise ValueError('Base asset changed')
state=ModelState(raw.theta0,stored['S'].to(raw.theta0.device),tuple(Group(g['name'],g['Z'].to(raw.theta0.device),g['a'].to(raw.theta0.device)) for g in stored['groups']),stored['dictionary_version']).validate()
oracle=MeasuredOracle(raw,Ledger());geo=refresh(state,oracle)
_,gradient=oracle.gradient(state,raw.sample(11,999999,()))
gx=state.S.double().T@gradient.double()
_,old=original_direction(state,geo,gx)
delta,info=module.fr_preconditioned_direction(state,geo,gx)
Q=torch.linalg.svd(state.G,full_matrices=False).U
reference=-Q@torch.linalg.solve(Q.T@(geo.A.T@geo.A)@Q+(geo.damping+1e-8)*torch.eye(Q.shape[1],device=Q.device,dtype=Q.dtype),Q.T@gx)
error=float((state.G@delta-reference).norm()/reference.norm())
duplicated=replace(state,groups=state.groups+(Group('duplicate_check',state.G,torch.zeros_like(state.a)),))
repeated,repeated_info=module.fr_preconditioned_direction(duplicated,geo,gx)
repeat_error=float((duplicated.G@repeated-state.G@delta).norm()/(state.G@delta).norm())
result=dict(status='passed' if info['fallback'] is None and repeated_info['fallback'] is None and error<1e-8 and repeat_error<1e-8 else 'failed',
    fixed_geometry_sha256=hashlib.sha256(fixed.read_bytes()).hexdigest(),
    original_condition=old['condition'],original_fallback=old['fallback'],
    reduced_condition=info['condition'],reduced_fallback=info['fallback'],
    coordinate_count=state.total_rank,effective_dimension=info['effective_system_dimension'],coordinate_nullity=info['coordinate_nullity'],
    direct_subspace_step_relative_error=error,duplicate_coordinates_step_relative_error=repeat_error,
    diagnostic_online_images=oracle.ledger.used,no_optimizer_updates=True,training_result_modified=False,
    full_fixed_solver_training_complete=False)
(qa/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result))
