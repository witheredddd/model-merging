"""Scalar policies shared by unattended baseline generation."""
import math
import hashlib,json
from pathlib import Path

def intervention_scale(norm,radius,original_scale=False):
    if not math.isfinite(norm) or norm<0:raise ValueError('Invalid parameter norm')
    if not math.isfinite(radius) or radius<=0:raise ValueError('Invalid radius')
    if original_scale:return 1.
    return radius/norm if norm else 0.

def archive_completed_controls(root,anchor):
    """Remove only derived control payloads after retaining their complete recipes."""
    root=Path(root).resolve();index=root/f'CONTROL_ARCHIVE_a{anchor}.json'
    entries=json.loads(index.read_text())['records'] if index.exists() else []
    known={x['vector']['file'] for x in entries}
    for f in sorted((root/'evaluations').glob('*.json')):
        case=json.loads(f.read_text())
        if case['anchor']!=anchor:continue
        ref=case['vector'];path=(root/ref['file']).resolve()
        if not path.is_relative_to(root):raise ValueError('Control archive outside experiment')
        if path.parent.name not in ('controls','compatibility','expert_baselines'):continue
        if ref['file'] in known or not path.exists():continue
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
        if h.hexdigest()!=ref['sha256']:raise ValueError('Control checksum changed before archival')
        entries.append(dict(vector=ref,case=case['case'],case_file=str(f.relative_to(root)),
            reconstruction='Use retained gradients/bases/solver solutions and the case-specific CPU reconstruction helper'))
        known.add(ref['file'])
        temporary=index.with_suffix('.tmp');temporary.write_text(json.dumps(dict(records=entries,
            original_inputs_untouched=True,all_gradients_bases_solver_motions_and_observations_retained=True),indent=2));temporary.replace(index)
        path.unlink()
