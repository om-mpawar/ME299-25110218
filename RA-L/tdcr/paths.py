"""Repository locations, in one place. Every script builds its paths from these."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
MESHES = os.path.join(ROOT, "data", "meshes")
CABLES = os.path.join(ROOT, "data", "cables")
FIGS = os.path.join(ROOT, "poster", "figs")
PAPER = os.path.join(ROOT, "paper")
