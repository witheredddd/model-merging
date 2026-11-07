"""Original CLIP/data utilities, independent of the discarded algorithm."""
from types import SimpleNamespace
import torch


def build_model(u,head):
    enc=u.ImageEncoder('ViT-B-16').cuda()
    state=torch.load(u.ROOT/'models/checkpoints/ViT-B-16/pretrained.pt',map_location='cpu',weights_only=True)
    expected=enc.state_dict()
    if not any(n.startswith('model.') for n in state):
        state={'model.'+n:v for n,v in state.items() if 'model.'+n in expected}
    for n,t in expected.items():assert torch.equal(t.cpu(),state[n].cpu()),n
    enc.load_state_dict(state);model=u.ImageClassifier(enc,head.cuda()).cuda().float()
    found=[]
    for n,m in enc.named_modules():
        if 'visual.transformer.resblocks.' not in n:continue
        a='in_proj_weight' if isinstance(m,torch.nn.MultiheadAttention) else 'weight' if isinstance(m,torch.nn.Linear) else None
        if a:found.append((n+'.'+a,getattr(m,a)))
    assert len(found)==48
    return enc,model,found,torch.cat([p.detach().flatten() for _,p in found])


def training_data(u,enc,seed):
    from torchvision.datasets import ImageFolder
    root=u.ROOT/'datasets/dtd/train'
    tr=ImageFolder(str(root),transform=enc.train_preprocess)
    ev=ImageFolder(str(root),transform=enc.val_preprocess)
    a,b=torch.utils.data.random_split(tr,[3384,376],generator=torch.Generator().manual_seed(0))
    ea=torch.utils.data.Subset(ev,a.indices);eb=torch.utils.data.Subset(ev,b.indices)
    ed=u.EpochData(a,seed)
    return ed,u.loader(ed,True,seed),u.loader(eb),SimpleNamespace(train_dataset=ea,test_dataset=eb)
