from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm

root=Path(sys.argv[1]);data=np.load(root/'block_matrices.npz');result=json.loads((root/'results.json').read_text())
fig,axes=plt.subplots(3,3,figsize=(12,11),constrained_layout=True)
labels=data['labels'];boundaries=np.where(np.diff(labels)!=0)[0]+.5
for i,(axis,step,A) in enumerate(zip(axes.ravel(),data['states'],data['rotated_F'])):
    norm=np.linalg.norm(A);image=axis.imshow(A/max(norm,1e-30),cmap='coolwarm',norm=SymLogNorm(linthresh=1e-5,vmin=-1,vmax=1))
    for b in boundaries:axis.axhline(b,color='black',lw=1);axis.axvline(b,color='black',lw=1)
    axis.set_title(f'Step {step} | '+('fit' if i in result['training_state_indices'] else 'held Fisher'))
    axis.set_xlabel('Shared rotated axis');axis.set_ylabel('Shared rotated axis')
    np.savetxt(root/f'raw_fisher_step_{step:03d}.csv',data['F'][i],delimiter=',')
fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.7,label='Signed matrix entries / Frobenius norm')
fig.suptitle('DTD temporal Fisher family | blocks '+str(result['block_sizes'])+' | held support '+str(result['held_out_support']))
fig.savefig(root/'fisher_family.png',dpi=160)
print(root/'fisher_family.png')
