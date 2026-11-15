"""Validate source files without running model evaluation."""
import ast
from pathlib import Path
names=['fiber_core.py','run_a100.py','run_comparators_a100.py','test_fiber_core.py','test_runner_mock.py']
for n in names:
    file=Path(__file__).with_name(n)
    ast.parse(file.read_text(encoding='utf8'))
    print(n, 'syntax OK')
