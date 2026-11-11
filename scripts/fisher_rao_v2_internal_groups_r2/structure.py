"""Within-task unsigned-axis clustering; signed effects and overlaps remain visible."""
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage,cut_tree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score,silhouette_score
from fr_math import decoder,qf


def orth_dictionary(columns,cap=16):
    norms=torch.tensor([float(v.double().norm()) for v in columns],dtype=torch.double)
    assert bool((norms>1e-12).all()),'Zero parameter candidate'
    n,D=len(columns),columns[0].numel();gram=torch.zeros(n,n,dtype=torch.double)
    for start in range(0,D,262144):
        part=torch.stack([v[start:start+262144] for v in columns]).double()/norms[:,None];gram+=part@part.T
    ev,V=torch.linalg.eigh(gram);ids=(ev>ev[-1]*1e-8).nonzero().flatten()[-cap:].flip(0)
    coeff=V[:,ids]/ev[ids].sqrt()/norms[:,None]
    S=torch.empty(D,len(ids))
    for start in range(0,D,262144):S[start:start+262144]=torch.stack([v[start:start+262144] for v in columns],1).double()@coeff
    return S,coeff


def reachable(A,error=0.):
    ev,V=torch.linalg.eigh(A.double().T@A.double());idx=ev.argsort(descending=True);s=ev[idx].clamp_min(0).sqrt();V=V[:,idx]
    keep=s>max(float(s[0])*1e-3,10*error)
    if not keep.any():raise ValueError('No stable reachable image')
    return A.double()@V[:,keep]/s[keep],s,V[:,keep]


def cosine(R):
    norms=R.norm(dim=0);return R.T@R/(norms[:,None]*norms[None,:]).clamp_min(1e-30)


def labels_for(R,M):
    if M==1:return np.zeros(R.shape[1],dtype=int)
    D=(1-cosine(R).abs()).clamp(0,2).cpu().numpy();np.fill_diagonal(D,0)
    return cut_tree(linkage(squareform(D,checks=False),method='complete'),n_clusters=[M]).ravel()


def group_bases(L,R,labels,rank_per_group=2):
    result=[]
    for group in sorted(set(labels)):
        ids=np.where(labels==group)[0];B=L.T@R[:,ids]
        u,s,_=torch.linalg.svd(B,full_matrices=False);rank=min(rank_per_group,int((s>s[0]*1e-3).sum()))
        if rank==0:raise ValueError('Unreachable group')
        result.append(L@u[:,:rank])
    if sum(U.shape[1] for U in result)>L.shape[1]:raise ValueError('Coordinate rank exceeds stable reachable rank')
    return result


def overlaps(groups):
    rows=[]
    for i,U in enumerate(groups):
        for j in range(i):
            V=groups[j];singular=torch.linalg.svdvals(U.T@V).clamp(0,1)
            rows.append(dict(i=j,j=i,principal_angles=singular.acos().tolist(),
                 projection_i_into_j=float((V.T@U).square().sum()/U.shape[1]),
                 projection_j_into_i=float((V.T@U).square().sum()/V.shape[1])))
    return rows


def choose_structure(R,Rprime,L,metadata,classes,bootstrap=100):
    n=R.shape[1];c=cosine(R);cp=cosine(Rprime);choices=[]
    for M in range(1,min(4,n//2)+1):
        labels=labels_for(R,M);review=labels_for(Rprime,M)
        support=all(sum(metadata[i]['kind']=='minibatch' for i in np.where(labels==m)[0])>=2 for m in range(M))
        try:bases=group_bases(L,R,labels)
        except ValueError:continue
        reachable_retention=[];chart_retention=[]
        for group,U in enumerate(bases):
            target=R[:,labels==group];den=target.square().sum().clamp_min(1e-30)
            reachable_retention.append(float((L.T@target).square().sum()/den))
            chart_retention.append(float((U.T@target).square().sum()/den))
        ov=overlaps(bases);equivalent=any(min(r['projection_i_into_j'],r['projection_j_into_i'])>=.95 for r in ov)
        ari=float(adjusted_rand_score(labels,review));sil=0.
        if M>1:
            D=(1-c.abs()).clamp_min(0).cpu().numpy();np.fill_diagonal(D,0)
            sil=float(silhouette_score(D,labels,metric='precomputed'))
        stability=1.
        if M>1 and support:
            rng=np.random.default_rng(20261111);count=R.shape[0]//classes;sample=R.view(count,classes,n)
            values=[]
            for _ in range(bootstrap):
                ids=rng.integers(count,size=count);sampled=sample[ids].reshape(-1,n)
                values.append(adjusted_rand_score(labels,labels_for(sampled,M)))
            stability=float(np.mean(np.asarray(values)>=.7))
        qualified=(M==1 or (support and not equivalent and ari>=.7 and sil>=.15 and stability>=.8)) and min(reachable_retention)>=.7
        choices.append(dict(M=M,labels=labels.tolist(),support=support,independent_view_ARI=ari,silhouette=sil,
             stability=stability,equivalent_groups=equivalent,overlap=ov,qualified=bool(qualified),
             ranks=[U.shape[1] for U in bases],reachable_retention=reachable_retention,chart_retention=chart_retention))
    return dict(candidates=choices,signed_cosine=c.tolist(),independent_signed_cosine=cp.tolist(),
                dissimilarity='1-abs(cosine)',bootstrap=bootstrap,grouping_objects='same-task candidate updates',
                selected_M=None,selection='qualified structures require subsequent Vcal and necessity comparisons')


def parameter_labels(vectors,M):
    norms=torch.tensor([float(v.double().norm()) for v in vectors]);n=len(vectors);D=vectors[0].numel()
    gram=torch.zeros(n,n,dtype=torch.double)
    for start in range(0,D,262144):
        p=torch.stack([v[start:start+262144] for v in vectors]).double()/norms[:,None];gram+=p@p.T
    if M==1:return np.zeros(n,dtype=int)
    distance=(1-gram.abs()).clamp_min(0).numpy();np.fill_diagonal(distance,0)
    return cut_tree(linkage(squareform(distance,checks=False),method='complete'),n_clusters=[M]).ravel()


def transport(q,p,U,classes):
    a=q.reshape(-1,classes);b=p.reshape(-1,classes);v=U.reshape(-1,classes,U.shape[1])
    denominator=a.square().sum(-1)+(a*b).sum(-1)
    return (v-((v*b[:,:,None]).sum(1)/denominator[:,None])[:,None,:]*(a+b)[:,:,None]).reshape_as(U)


def match_groups(left,right):
    affinity=np.array([[float((U.T@V).square().sum()/max(U.shape[1],V.shape[1])) for V in right] for U in left])
    a,b=linear_sum_assignment(-affinity)
    return dict(matches=[dict(left=int(i),right=int(j),affinity=float(affinity[i,j])) for i,j in zip(a,b)],
                minimum_affinity=float(affinity[a,b].min()),group_count_matches=len(left)==len(right))
