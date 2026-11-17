"""Read-only real normal-training positions. Never interpolates or optimizes them."""
import json
from pathlib import Path
import torch
from bootstrap import R3
from io_state import digest


class TrainingPositions:
    def __init__(self,trace,parameter_names,output):
        self.trace=Path(trace);self.names=list(parameter_names);self.output=Path(output)
        self.manifest=self.trace/'trace.json';self.data=json.loads(self.manifest.read_text())
        self.rows=self.data['rows']
        steps=[r['step'] for r in self.rows]
        if len(steps)<2 or any(a>=b for a,b in zip(steps,steps[1:])):raise ValueError('Strictly ordered real positions are required')
        self.manifest_sha256=digest(self.manifest)

    def verify_normal_recipe(self):
        proof_file=self.trace.parent/'comparison.json'
        proof=json.loads(proof_file.read_text())
        if not proof.get('exact_tensor_match'):raise ValueError('Normal-training replay was not reproduced')
        source=proof['source'];path=Path(source['best_checkpoint'])
        if digest(path)!=source['checkpoint_sha256']:raise ValueError('Original trained expert identity changed')
        trained=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
        provenance=trained['provenance']
        if provenance.get('optimizer')!='AdamW' or provenance.get('visual_scope')!='full':
            raise ValueError('This carrier is not the recorded ordinary full-visual training recipe')
        last=self.raw(len(self.rows)-1)
        if not all(torch.equal(last['visual'][k],trained['visual'][k]) for k in self.names):
            raise ValueError('Recorded endpoint differs from original ordinary-trained parameters')
        return dict(proof_file=str(proof_file),proof_sha256=digest(proof_file),
            original_expert_sha256=source['checkpoint_sha256'],exact_endpoint_tensor_match=True,
            original_training_provenance=provenance,
            sampling_note='Only recorded positions are used; no unrecorded intermediate step is invented')

    def raw(self,index):
        row=self.rows[index];p=(self.trace/row['checkpoint']).resolve()
        if not p.is_relative_to(self.trace.resolve()):raise ValueError('Unsafe source checkpoint path')
        if digest(p)!=row['sha256']:raise ValueError('Training position checksum mismatch')
        obj=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
        if obj['step']!=row['step']:raise ValueError('Step identity mismatch')
        if set(obj['visual'])!=set(self.names):raise ValueError('Full parameter keys differ')
        return obj

    def flat(self,index):
        obj=self.raw(index)
        return torch.cat([obj['visual'][k].reshape(-1) for k in self.names])

    def prepare(self,index):
        folder=self.output/'carrier';folder.mkdir(exist_ok=True,parents=True)
        path=folder/f'position{index:05d}_step{self.rows[index]["step"]:06d}.pt'
        point=self.flat(index)
        if path.exists():
            stored=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
            if stored['source_step']!=self.rows[index]['step'] or not torch.equal(stored['full_visual_flat'],point):
                raise ValueError('Derived carrier differs from the actual recorded position')
        else:
            temp=path.with_suffix('.tmp')
            torch.save(dict(full_visual_flat=point,source_step=self.rows[index]['step']),temp);temp.replace(path)
        return dict(file=str(path.relative_to(self.output)),sha256=digest(path),serial=index,
            source_step=self.rows[index]['step'],source_checkpoint=str(self.trace/self.rows[index]['checkpoint']),
            source_sha256=self.rows[index]['sha256'],role='immutable normal-training carrier position; not an optimized anchor')

    def direction_interval(self,index):
        # At the final position use the actual incoming motion as a reference,
        # never invent a closing jump back to the beginning of training.
        return (index,index+1) if index+1<len(self.rows) else (index-1,index)
