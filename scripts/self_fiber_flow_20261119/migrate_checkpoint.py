"""Explicit, auditable migration of the initial discovery checkpoint only."""
import json,time,fcntl
from pathlib import Path
import bootstrap
from io_state import Transactions,digest

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    lock=(a.output/'WRITER.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    store=Transactions(a.output);s=store.load()
    if s['phase']!='DISCOVER' or s['transitions']:
        raise ValueError('Migration is restricted to discovery before physical transitions')
    for path,h in s['inputs'].items():
        if digest(path)!=h:raise ValueError('Frozen input changed: '+path)
    codes={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')}
    s.setdefault('source_migrations',[]).append(dict(time=time.time(),before=s['codes'],after=codes,
        reason='Finish V1 path sampling, independent functional innovation and finite vertical diagnostics; retain all discovery state'))
    s['codes']=codes;store.save(s)
    print(json.dumps(dict(generation=s['generation'],phase=s['phase'],basis_size=len(s['basis']))))
