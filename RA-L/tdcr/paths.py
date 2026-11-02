"""Repository locations, in one place. Every script builds its paths from these."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
MESHES = os.path.join(DATA, "meshes")              # not in the repository: see data/README.md
CABLES = os.path.join(DATA, "cables")
CONTACTS = os.path.join(DATA, "contacts")
CAD = os.path.join(DATA, "cad")
HARDWARE = os.path.join(DATA, "hardware")          # bench measurements, command lists, servo zeros
PAPER = os.path.join(ROOT, "paper")
FIGURES = os.path.join(PAPER, "figures")


def sofa_root():
    """SOFA installation (SOFA 25.06 with SofaPython3, SoftRobots, STLIB), from the SOFA_ROOT environment variable."""
    root = os.environ.get("SOFA_ROOT")
    if not root:
        raise RuntimeError("set SOFA_ROOT to the SOFA 25.06 installation (see sim/README.md)")
    return root
