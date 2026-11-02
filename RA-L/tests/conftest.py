"""Run the tests from the repository root: `.venv/bin/python -m pytest -q`. Puts the root on sys.path so that
`tdcr`, `analysis` and `hardware` import as packages, and makes relative data paths resolve from the root."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
