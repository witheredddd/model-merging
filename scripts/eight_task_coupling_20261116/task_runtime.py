"""Eight-task complete visual model, data probes and expert demand sources."""
import hashlib,json,sys
from importlib.metadata import version
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from prepare import COORDINATES,ROOT,TASKS,labels,write,sha


def verify_inventory_assets(inventory):
    checked={}
    for row in inventory:
        for kind in ('expert','head'):
            path=Path(row[f'{kind}_path'])
            expected=row[f'{kind}_sha256']
            actual=sha(path)
            if actual!=expected:
                raise ValueError(f"{row['task']} {kind} SHA-256 mismatch")
            checked.setdefault(row['task'],{})[kind]=dict(path=str(path),sha256=actual)
    return checked


def probe_signatures(context,probe):
    signatures={}
    for task in context.task_names:
        digest=hashlib.sha256()
        for images,labels_tensor in context.batches[task,probe]:
            for tensor in (images,labels_tensor):
                array=tensor.detach().cpu().contiguous().numpy()
                digest.update(repr((array.shape,str(array.dtype))).encode())
                digest.update(memoryview(array).cast('B'))
        signatures[task]=digest.hexdigest()
    return signatures


def balanced_ids(y,count,rng,excluded):
    classes=np.unique(y)
    pools={int(c):list(map(int,rng.permutation([i for i in np.flatnonzero(y==c) if int(i) not in excluded]))) for c in classes}
    chosen=[]
    while len(chosen)<count:
        for cls in classes:
            if pools[int(cls)]:chosen.append(pools[int(cls)].pop())
            if len(chosen)==count:break
        if not any(pools.values()) and len(chosen)<count:raise ValueError('Insufficient disjoint training images')
    return chosen


class ExpertSources:
    def __init__(self,owner,inventory):self.owner=owner;self.inventory=inventory
    def __getitem__(self,index):
        row=self.inventory[index]
        x=torch.load(row['expert_path'],map_location='cpu',mmap=True,weights_only=False)['visual']
        return torch.cat([x[k].reshape(-1) for k in self.owner.keys]).double()


class EightTaskContext:
    def __init__(self,out,inventory):
        self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.task_names=tuple(TASKS)
        sys.path[:0]=[str(ROOT),str(ROOT/'scripts/fr_hst')]
        from src.models.modeling import ImageEncoder,ImageClassifier,ClassificationHead
        from task_data import train_set
        torch.set_grad_enabled(False);torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.enc=ImageEncoder('ViT-B-16').cuda()
        pretrained=ROOT/'models/checkpoints/ViT-B-16/pretrained.pt'
        coordinates=json.loads(COORDINATES.read_text())
        self.pretrained_sha256=sha(pretrained)
        if self.pretrained_sha256!=coordinates['base_sha256']:
            raise ValueError('Pretrained checkpoint SHA-256 mismatch')
        state=torch.load(pretrained,map_location='cpu',weights_only=True)
        expected=self.enc.state_dict()
        if not any(k.startswith('model.') for k in state):
            state={'model.'+k:v for k,v in state.items() if 'model.'+k in expected}
        self.enc.load_state_dict(state)
        self.heads=[ClassificationHead.load(r['head_path']).cuda().requires_grad_(False) for r in inventory]
        self.model=ImageClassifier(self.enc,self.heads[0]).cuda().float().requires_grad_(False).eval()
        items=list(self.enc.model.visual.named_parameters())
        self.keys=[k for k,_ in items];self.sizes=[p.numel() for _,p in items]
        self.shapes=[p.shape for _,p in items]
        self.names=['image_encoder.model.visual.'+k for k in self.keys]
        self.base=torch.cat([p.detach().cpu().reshape(-1) for _,p in items]).double()
        self.D=len(self.base);self.experts=ExpertSources(self,inventory)
        if self.keys!=coordinates['parameter_keys'] or self.D!=86192640:
            raise ValueError('Complete visual parameter order/count differs from audited coordinates')
        if [list(s) for s in self.shapes]!=coordinates['parameter_shapes']:
            raise ValueError('Complete visual parameter shapes differ')
        if tuple(r['task'] for r in inventory)!=self.task_names:
            raise ValueError('Task inventory and head order differ')
        self.batches={};locks={};rng=np.random.default_rng(1)
        lockfile=self.out/'probes.json'
        old=json.loads(lockfile.read_text()) if lockfile.exists() else None
        for task in self.task_names:
            ds=train_set(task,self.enc.val_preprocess);y=labels(ds);classes=np.unique(y)
            # Data/compute probe size only. It never sets a structural dimension.
            count=300 if task=='Cars' else max(200,len(classes))
            if old:ids=old['tasks'][task]['ids']
            else:
                first=balanced_ids(y,count,rng,set())
                second=balanced_ids(y,count,rng,set(first));ids=[first,second]
            if set(ids[0])&set(ids[1]):raise ValueError('Learning/validation images overlap')
            locks[task]=dict(ids=ids,classes=classes.tolist(),training_side=True,
                      sampling='Balanced stratified locked numerical probes; not parameter support or mode counts')
            for probe,indices in enumerate(ids):
                self.batches[task,probe]=[(x.cuda(),z.cuda()) for x,z in
                    DataLoader(Subset(ds,indices),batch_size=32,num_workers=2)]
        if not old:write(lockfile,dict(tasks=locks,data_split_seed=1,official_test_opened=False,
              expert_training_unseen=False,probe_policy='Reuse original DTD200/Cars300 development scale; others cover at least all classes. This is an empirical eight-task development geometry.'))

    def predict(self,point,x,task):
        self.model.classification_head=self.heads[self.task_names.index(task)]
        parameters={name:v.view(shape) for name,v,shape in zip(self.names,point.split(self.sizes),self.shapes)}
        return torch.func.functional_call(self.model,parameters,(x,)).float()

    def baseline(self,task,probe,point):
        zs=[];ys=[]
        for x,y in self.batches[task,probe]:
            zs.append(self.predict(point,x,task).detach().double().cpu());ys.append(y.cpu())
        return torch.cat(zs),torch.cat(ys)


def load_eight_geometry(out,inventory_file,checkpoint,probe=0,*,expected=None):
    from geometry import FullFunctionalGeometry
    probe_path=Path(out)/'probes.json'
    locked=dict(checkpoint_sha256=sha(checkpoint),inventory_sha256=sha(inventory_file))
    if probe_path.exists():
        locked['probe_sha256']=sha(probe_path)
    if expected is not None and any(locked.get(key)!=expected[key] for key in
                                   ('checkpoint_sha256','inventory_sha256','probe_sha256')):
        raise ValueError('Checkpoint, inventory or probe SHA-256 mismatch')
    inventory=json.loads(Path(inventory_file).read_text())['tasks']
    if any(row['status']!='coordinate_head_and_training_metadata_checked' for row in inventory):
        raise ValueError('An expert/head resource has not passed coordinate checks')
    resources=verify_inventory_assets(inventory)
    context=EightTaskContext(out,inventory)
    state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
    point=state['full_visual_flat'].cuda().double()
    geometry=FullFunctionalGeometry(context,point,probe,precision='float64')
    probe_digests=probe_signatures(context,probe)
    if expected is not None and 'transformed_probe_sha256' in expected:
        if probe_digests!=expected['transformed_probe_sha256']:
            raise ValueError('Transformed probe SHA-256 mismatch')
    module_names=('task_runtime','geometry','precision','prepare','task_data',
                  'src.models.modeling','src.utils.variables_and_paths',
                  'src.datasets.cars','src.datasets.gtsrb','src.datasets.resisc45','src.datasets.svhn')
    code={name:sha(sys.modules[name].__file__) for name in module_names}
    return geometry,dict(parameter_dimension=geometry.dimension,visual_tensors=len(context.keys),
        tasks=list(geometry.tasks),checkpoint=str(checkpoint),checkpoint_sha256=locked['checkpoint_sha256'],
        source='Existing anchor0 complete visual state; not an asserted eight-task optimum',
        probe_file=str(Path(out)/'probes.json'),probe_sha256=sha(Path(out)/'probes.json'),
        inventory_file=str(inventory_file),inventory_sha256=locked['inventory_sha256'],
        verified_assets=resources,pretrained_sha256=context.pretrained_sha256,
        coordinates_sha256=sha(COORDINATES),transformed_probe_sha256=probe_digests,
        runtime_source_sha256=code,versions=dict(torch=torch.__version__,numpy=np.__version__,
            torchvision=sys.modules['torchvision'].__version__,open_clip=version('open_clip_torch'),
            cuda=torch.version.cuda),
        dictionary_projection=False,predefined_support=False,numerical_precision='float64')
