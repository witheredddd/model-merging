"""Atomic snapshots of committed online state; no optimizer/geometry changes."""
from dataclasses import asdict
import os
import random
from pathlib import Path
import numpy as np
import torch
from state import ModelState,Group
from geometry import Geometry
from trainer import OnlineTrainer
from evidence import Persistence
from work import Ledger,MeasuredOracle


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    if isinstance(value,list):return [cpu(v) for v in value]
    return value


def model_payload(state):
    return dict(S=cpu(state.S),groups=[dict(name=g.name,Z=cpu(g.Z),a=cpu(g.a)) for g in state.groups],
                dictionary_version=state.dictionary_version)


def load_model(payload,raw):
    device=raw.theta0.device
    return ModelState(raw.theta0,payload['S'].to(device),tuple(Group(g['name'],g['Z'].to(device),g['a'].to(device))
        for g in payload['groups']),payload['dictionary_version']).validate()


def atomic_save(path,payload):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def archive_snapshot(output,payload):
    output=Path(output);folder=output/'checkpoints';folder.mkdir(exist_ok=True)
    cost=sum(payload['ledger']['images'].values())
    archive=folder/f"step_{payload['iteration']:06d}_images_{cost:09d}.pt"
    if not archive.exists():atomic_save(archive,payload)
    # A hard link keeps every historical snapshot without duplicating the latest
    # multi-GB tensor data. Never write to an already published snapshot inode.
    latest=output/'recovery_state.pt';temporary=output/'recovery_state.link.tmp'
    try:
        if temporary.exists():temporary.unlink()
        os.link(archive,temporary);os.replace(temporary,latest)
    except OSError:
        atomic_save(latest,payload)
    return archive


def capture(engine,reporter,identity,selected,elapsed_seconds=0.):
    if reporter.pending is not None:raise ValueError('Flush committed reporter before checkpoint')
    geo=engine.geometry;geo.require_current_dictionary(engine.state)
    numpy_state=np.random.get_state()
    return dict(format_version=1,identity=identity,settings=asdict(engine.settings),
        state=model_payload(engine.state),geometry={k:cpu(getattr(geo,k)) for k in geo.__dataclass_fields__
            if k not in ('dictionary_pointer','dictionary_mutation_version')},
        radius=engine.radius,iteration=engine.iteration,window=engine.window,
        dictionary_replacement_failures=engine.dictionary_replacement_failures,
        persistence=dict(reference=cpu(engine.persistence.reference),last=cpu(engine.persistence.last),
            windows=[dict(ids=sorted(w['ids']),basis=cpu(w['basis'])) for w in engine.persistence.windows]),
        ledger=dict(limit=engine.ledger.limit,images=dict(engine.ledger.images),operations=dict(engine.ledger.operations)),
        events=cpu(engine.events),geometry_events=cpu(engine.geometry_events),initialization=cpu(engine.initialization),
        reporter=dict(costs=reporter.costs,rows=cpu(reporter.rows),next=reporter.next,best=reporter.best,pending=None),
        selected=cpu(selected),runtime_calls=dict(getattr(getattr(engine.raw,'rt',None),'calls',{})),
        rng=dict(python=random.getstate(),numpy=(numpy_state[0],numpy_state[1].tolist(),numpy_state[2:]),
            torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
        elapsed_seconds=elapsed_seconds)


def restore(payload,raw,settings,identity,reporter):
    if payload['format_version']!=1 or payload['identity']!=identity or payload['settings']!=asdict(settings):
        raise ValueError('Checkpoint source/assets/task/settings identity mismatch')
    if payload['reporter']['costs']!=reporter.costs:raise ValueError('Checkpoint reporting grid mismatch')
    engine=OnlineTrainer.__new__(OnlineTrainer);engine.raw=raw;engine.settings=settings
    engine.state=load_model(payload['state'],raw);device=raw.theta0.device
    geometry={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in payload['geometry'].items()}
    engine.geometry=Geometry(**geometry,dictionary_pointer=engine.state.S.data_ptr(),
                             dictionary_mutation_version=engine.state.S._version)
    engine.geometry.require_current_dictionary(engine.state)
    engine.ledger=Ledger(**payload['ledger']);engine.oracle=MeasuredOracle(raw,engine.ledger)
    for name in ('radius','iteration','window','dictionary_replacement_failures','initialization','events','geometry_events'):
        setattr(engine,name,payload[name])
    engine.persistence=Persistence(settings.persistence_windows,overlap=settings.recurrence_overlap)
    record=payload['persistence'];engine.persistence.reference=None if record['reference'] is None else record['reference'].to(device)
    engine.persistence.last=record['last']
    engine.persistence.windows=[dict(ids=set(w['ids']),basis=w['basis'].to(device)) for w in record['windows']]
    engine.snapshots=[]
    for name in ('rows','next','best','pending'):setattr(reporter,name,payload['reporter'][name])
    if hasattr(raw,'rt'):raw.rt.calls=dict(payload['runtime_calls'])
    rng=payload['rng'];random.setstate(rng['python'])
    np.random.set_state((rng['numpy'][0],np.array(rng['numpy'][1],dtype=np.uint32),*rng['numpy'][2]))
    torch.set_rng_state(rng['torch'])
    if rng['cuda']:torch.cuda.set_rng_state_all(rng['cuda'])
    engine.ledger.on_progress=reporter.before_progress
    return engine
