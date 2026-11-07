"""Actual subspace comparison, computed from saved tensors without D-by-D projectors."""
import hashlib,json,math,time
from pathlib import Path
import torch
OUT=Path('/home/shangxueyuan/merging/tsv/results/functional_geometry_dtd_v2_2')

if __name__=='__main__':
    torch.set_num_threads(8)
    initial=torch.load(OUT/'initial_basis.pt',map_location='cpu',weights_only=True)['B']
    ck=torch.load(OUT/'best.pt',map_location='cpu',weights_only=True);selected=ck['B'];cross=torch.zeros(initial.shape[1],selected.shape[1],dtype=torch.float64)
    h0=hashlib.sha256();h1=hashlib.sha256()
    for a,b in zip(initial.split(262144),selected.split(262144)):
        cross+=a.double().T@b.double();h0.update(a.numpy().tobytes());h1.update(b.numpy().tobytes())
    angles=torch.linalg.svdvals(cross).clamp(0,1).acos()*180/math.pi
    row=dict(time=time.time(),checkpoint_epoch=ck['epoch'],checkpoint_validation=ck['validation'],rank=initial.shape[1],
        principal_angles_degrees=angles.tolist(),squared_projector_distance=2*initial.shape[1]-2*float(cross.square().sum()),
        initial_B_tensor_sha256=h0.hexdigest(),selected_B_tensor_sha256=h1.hexdigest(),
        interpretation='Nonzero principal angles verify a genuinely changed parameter subspace; they do not establish functional task structure or frozen low-dimensional control.')
    (OUT/'span_check.json').write_text(json.dumps(row,indent=2));print(json.dumps(row,indent=2))
