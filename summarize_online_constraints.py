"""Validate archived hashes and regenerate the interim experiment summary."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).parent / 'scripts/fr_hst/diagnostics/20261112/online_constraints'
manifest = json.loads((root / 'manifest.json').read_text())
summary = []
for name, entry in manifest['runs'].items():
    run = root / 'runs' / name
    for filename, expected in entry['files'].items():
        data = (run / filename).read_bytes()
        assert len(data) == expected['bytes'], (name, filename)
        assert hashlib.sha256(data).hexdigest() == expected['sha256'], (name, filename)
    def read(filename):
        p = run / filename
        return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
    validation = read('validation.jsonl')
    geometry = read('geometry.jsonl')
    controls = read('controls.jsonl')
    audit = [r['audit'] for r in controls if r['accepted']]
    row = dict(run=name, complete=entry['complete'],
               selected=min(validation, key=lambda r: r['ce']) if validation else None,
               final=validation[-1] if validation else None,
               geometry_windows=len(geometry), stable_windows=sum(r['stable'] for r in geometry),
               attempted_controls=len(controls), accepted_controls=len(audit),
               audit_ce_violations=sum(r['delta_ce'] > .005 for r in audit),
               audit_fr_violations=sum(r['fr2'] > .0025 for r in audit))
    summary.append(row)
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('Validated archived hashes for', len(summary), 'runs')
for row in summary:
    s = row['selected']
    print(row['run'], row['complete'], s['epoch'], round(100*s['accuracy'], 2), round(s['ce'], 4))
