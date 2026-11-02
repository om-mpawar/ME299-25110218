"""Run the tests from anywhere: `.venv/bin/python -m pytest -q`. Relative data paths in the tests resolve from the
repository root."""
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
