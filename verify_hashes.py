"""Verify published files; no dependencies, network, or model assets needed."""
import hashlib
from pathlib import Path

root=Path(__file__).resolve().parent
count=0
for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
    expected,name=line.split('  ',1)
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('unsafe manifest path')
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:raise SystemExit('HASH MISMATCH: '+name)
    count+=1
print(f'PASS: {count} SHA256 file hashes')
