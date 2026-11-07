"""Model reconstruction API: theta=theta0+Bz; no fine-tuned weights required."""
import torch
from engine import Engine,OUT,TASKS

class GeometryBundle:
    def __init__(self,microbatch=32):
        self.engine=Engine(micro=microbatch);self.engine.basis_load()
        self.coordinates=torch.load(OUT/'selected_experts.pt',map_location='cpu',weights_only=False)['z'].cuda()
    def expert(self,task):
        k=TASKS.index(task);self.engine.place(self.coordinates[k]);return self.engine.models[k]
    def at(self,task,z):
        """Return task head on theta0+Bz. Returned models share one visual encoder."""
        p=torch.zeros(self.engine.reserve,device='cuda');z=torch.as_tensor(z,device='cuda',dtype=torch.float32)
        p[:len(z)]=z;self.engine.place(p);return self.engine.models[TASKS.index(task)]
    def merged(self,task):
        z=torch.load(OUT/'merged_coordinates.pt',map_location='cuda',weights_only=False)['z']
        return self.at(task,z)
