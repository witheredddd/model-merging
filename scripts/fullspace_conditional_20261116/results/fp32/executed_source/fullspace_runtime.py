"""Load complete visual coordinates directly; never instantiate the legacy Space."""
import hashlib
import shutil
import sys
from pathlib import Path
import torch
from geometry import FullFunctionalGeometry


def load_full_geometry(*,output,checkpoint,probe_lock,probe,precision='native'):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    # Reuse only complete-model prediction/coordinate loading and the locked data.
    sys.path.insert(0,'/home/shangxueyuan/merging/tsv/scripts/projected_fiber_1b_20261115_r1')
    from gpu_runtime import Context
    destination=output/'probes.json'
    if destination.exists() and destination.read_bytes()!=Path(probe_lock).read_bytes():
        raise ValueError('Existing output uses a different data lock')
    if not destination.exists():shutil.copy2(probe_lock,destination)
    c=Context(output)
    state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
    point=state['full_visual_flat'].to(device='cuda',dtype=torch.float32)
    geometry=FullFunctionalGeometry(c,point,probe,precision=precision)
    return geometry,dict(parameter_dimension=geometry.dimension,
                         tensor_keys=list(c.keys),tensor_sizes=list(c.sizes),
                         complete_coordinate_count=sum(c.sizes),
                         point_source=str(checkpoint),probe_lock=str(probe_lock),
                         probe_lock_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                         dictionary_projection=False,predefined_support_filter=False,
                         numerical_precision=precision,point_dtype=str(geometry.point.dtype),
                         layernorm_precision_replacements=getattr(geometry.c,'layernorm_precision_replacements',[]))
