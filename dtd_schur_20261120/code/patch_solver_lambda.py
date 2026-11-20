"""Fail-closed lambda-bound patcher for a COPY of schur_resumable.py.

Never edit the original frozen source in place. Supply --source and --destination.
Run the original dense tests against the patched copy before deploying it.
"""
from __future__ import annotations

import argparse
import py_compile
from pathlib import Path

REPLACEMENTS = (
    ('Stops when 0.5*residual^2 <= rtol*|q|',
     'Stops when 0.5*residual^2/lam <= rtol*|q|'),
    ('objective_gap_upper=.5 * residual * residual,',
     'objective_gap_upper=.5 * residual * residual / lam,'),
    ('objective_relative_gap=(.5 * residual * residual\n',
     'objective_relative_gap=(.5 * residual * residual / lam\n'),
    ('if .5 * residual * residual <= rtol * max(abs(q), np.finfo(float).tiny):',
     'if .5 * residual * residual / lam <= rtol * max(abs(q), np.finfo(float).tiny):'),
    ('gap = .5 * residual * residual\n',
     'gap = .5 * residual * residual / lam\n'),
)


def patch_content(text: str) -> str:
    for old, new in REPLACEMENTS:
        if text.count(old) != 1:
            raise ValueError(f'Cannot patch: expected exactly one source occurrence of {old!r}, found {text.count(old)}')
        text = text.replace(old, new, 1)
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    a = p.parse_args()
    if a.source.resolve() == a.destination.resolve():
        raise ValueError('Destination must differ from frozen source')
    if a.destination.exists():
        raise FileExistsError(a.destination)
    updated = patch_content(a.source.read_text(encoding='utf8'))
    a.destination.write_text(updated,encoding='utf8')
    py_compile.compile(str(a.destination),doraise=True)
    print('Patched copy:', a.destination)

if __name__ == '__main__':main()
