"""Symmetric commutant candidates with held-out leakage; never force blocks."""
import numpy as np

def leakage(matrices,labels):
    mask=np.not_equal.outer(labels,labels)
    return [float(np.linalg.norm(A*mask)/max(np.linalg.norm(A),1e-30)) for A in matrices]

def discover(matrices,train_indices,leakage_limit=.05):
    train=np.array([matrices[i]/max(np.linalg.norm(matrices[i]),1e-30) for i in train_indices])
    d=train.shape[1];basis=[]
    # Orthonormal traceless diagonal basis, then symmetric off-diagonal basis.
    for k in range(1,d):
        diag=np.r_[np.ones(k),-k,np.zeros(d-k-1)]/np.sqrt(k*(k+1))
        basis.append(np.diag(diag))
    for i in range(d):
        for j in range(i):
            B=np.zeros((d,d));B[i,j]=B[j,i]=1/np.sqrt(2);basis.append(B)
    if not basis:return dict(Q=np.eye(d),labels=np.zeros(d,dtype=int),commutator_spectrum=[],candidates=[])
    basis=np.array(basis)
    operator=np.stack([np.concatenate([(A@B-B@A).ravel() for A in train]) for B in basis],1)
    _,s,Vt=np.linalg.svd(operator,full_matrices=False)
    X=np.einsum('i,ijk->jk',Vt[-1],basis)
    eig,Q=np.linalg.eigh(X);gaps=np.diff(eig)
    candidates=[]
    transformed=np.array([Q.T@A@Q for A in matrices])
    for threshold in sorted(set(float(x) for x in gaps if x>1e-12),reverse=True):
        labels=np.r_[0,np.cumsum(gaps>=threshold)]
        if len(set(labels))<2:continue
        values=leakage(transformed[train_indices],labels)
        candidates.append(dict(labels=labels.tolist(),blocks=len(set(labels)),train_leakage=values,gap_threshold=threshold))
    admissible=[c for c in candidates if max(c['train_leakage'])<=leakage_limit]
    # Fit-only choice; held states never select the partition.
    selected=max(admissible,key=lambda c:c['blocks']) if admissible else None
    labels=np.array(selected['labels']) if selected else np.zeros(d,dtype=int)
    return dict(Q=Q if selected else np.eye(d),labels=labels,commutator_spectrum=s.tolist(),
        candidate_commutant_eigenvalues=eig.tolist(),candidates=candidates,selection=selected)
