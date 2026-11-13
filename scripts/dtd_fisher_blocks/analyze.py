"""CPU-only endpoint-update allocation and small-matrix controls."""
from pathlib import Path
import sys,json
import numpy as np
import torch
torch.set_num_threads(4)
root=Path(sys.argv[1]);r=json.loads((root/'results.json').read_text());m=np.load(root/'block_matrices.npz')
parts=torch.load(root/'complete_update_decomposition.pt',map_location='cpu',weights_only=True,mmap=True)
Q=torch.load(root/'common_basis.pt',map_location='cpu',weights_only=True,mmap=True)['Q_rows']
assets=json.loads((root/'assets.json').read_text());names=assets['active']
energy=[]
for coefficients in parts['group_coordinates']:
    offset=0;row=[]
    for parameter in names:
        length=int(np.prod(parameter['shape']));block=coefficients@Q[:,offset:offset+length]
        row.append(float(block.square().sum()));offset+=length
    if offset!=Q.shape[1]:raise ValueError('Parameter layout mismatch')
    energy.append(row)
spectra=np.linalg.eigvalsh(m['F'])[:,::-1]
reference=m['F'][r['training_state_indices'][0]]
_,O=np.linalg.eigh(reference)
off_diagonal=[]
for A in m['F']:
    B=O.T@A@O;off_diagonal.append(float(np.linalg.norm(B-np.diag(np.diag(B)))/np.linalg.norm(B)))
value=dict(matrix_names=[v['name'] for v in names],group_update_parameter_matrix_energy=energy,
    group_update_energy_fractions=(np.array(energy).sum(1)/float(parts['endpoint_tau'].square().sum())).tolist(),
    fisher_spectra=spectra.tolist(),single_fit_state_eigenbasis_offdiagonal_leakage=off_diagonal,
    full_residual_norm=float(parts['full_residual'].norm()),
    note='Allocation is descriptive, not semantic attribution. Shared Q uses the full known trajectory. Single-state eigenbasis is only a control.')
(root/'update_allocation_and_controls.json').write_text(json.dumps(value,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in value.items() if k in ('group_update_energy_fractions','full_residual_norm','single_fit_state_eigenbasis_offdiagonal_leakage')}))
