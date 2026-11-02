"""
Level-1 virtual sensor: SOFA cylinder FEM scene, extended with a headless
batch mode for the digital-tuning DOE sweep.

This is tdcr_scene.py's createScene() unchanged (same solver stack, same
cable setup, same interactive keyboard controller for GUI use) plus a
`run_batch()` entry point that:
  1. builds the scene once (no GUI, visual STL skipped for speed),
  2. locates the tip node once,
  3. for each (dl1, dl2, dl3) actuation sample: sets the three
     CableConstraint values, resets mechanical state (Sofa.Simulation.reset)
     so samples don't carry hysteresis from the previous one, steps the
     solver for `settle_steps` to reach the quasi-static equilibrium, and
     logs the tip position,
  4. writes everything to a CSV that digitaltuning.py consumes.

runSofa loads this file by calling createScene() directly and never runs the
`if __name__ == "__main__"` block below, so it can't be used to invoke
--batch (confirmed on the v25.06.00 install this project uses -- runSofa
silently ignores the argparse flags and just runs its own default batch
loop). Both interactive and batch mode instead run as a plain script, using
system python3 with SOFA's own Python bindings (built for this exact python3
version) added to PYTHONPATH/LD_LIBRARY_PATH -- no runSofa binary involved:

    export SOFA_ROOT=/home/pawar/SOFA/v25.06.00
    export PYTHONPATH="$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$SOFA_ROOT/plugins/SofaPython3/lib:$SOFA_ROOT/plugins/SoftRobots/lib/python3/site-packages:$SOFA_ROOT/plugins/STLIB/lib/python3/site-packages:$SOFA_ROOT/lib"
    export LD_LIBRARY_PATH="$SOFA_ROOT/lib:$SOFA_ROOT/plugins/SofaPython3/lib:$LD_LIBRARY_PATH"

    python3 sim/cylinder_scene.py                                          # interactive GUI
    python3 sim/cylinder_scene.py --batch results/configs.csv --out results/results.csv    # headless batch
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tdcr`
import argparse
import csv
import math
import os
import sys

import Sofa.Core

# The multithreaded FEM component spins up a thread pool per process; several
# processes at once oversubscribe the cores and one can stall. For parallel
# sweeps set TDCR_SERIAL_FEM=1 to use the single-threaded component (same
# corotational FEM, same equations) and run one process per core.
FEM_COMPONENT = ("TetrahedronFEMForceField" if os.environ.get("TDCR_SERIAL_FEM")
                 else "ParallelTetrahedronFEMForceField")
import Sofa.constants.Key as Key
from tendon import PullingCable  # pull-only (minForce = 0): sim/tendon.py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdcr.cc_kinematics import TENDON_R, BACKBONE_L  # single source of truth, shared with Level-0

# ── Mesh paths (relative to this script) ─────────────────────────────────────
from tdcr.paths import ROOT as _DIR  # paths below are relative to the repo root
MESH_VTK = os.path.join(_DIR, "data", "meshes", "backbone_4k.vtk")   # swap for make_cylinder_vtk.py output to sweep resolution
MESH_STL = os.path.join(_DIR, "data", "meshes", "backbone bashbhai.STL")

# ── Motor pulley (unused in batch mode, kept for interactive controller) ─────
PULLEY_DIA_MM = 10
PULLEY_R_M = (PULLEY_DIA_MM / 2.0) / 1000.0

Y_VALS = list(range(5, 220, 20))


def cable_xz(angle_deg):
    a = math.radians(angle_deg)
    return TENDON_R * math.cos(a), TENDON_R * math.sin(a)


C1_X, C1_Z = cable_xz(0)
C2_X, C2_Z = cable_xz(120)
C3_X, C3_Z = cable_xz(240)

CABLE_PTS = [
    [[C1_X, y, C1_Z] for y in Y_VALS],
    [[C2_X, y, C2_Z] for y in Y_VALS],
    [[C3_X, y, C3_Z] for y in Y_VALS],
]
PULL_PTS = [
    [C1_X, -10.0, C1_Z],
    [C2_X, -10.0, C2_Z],
    [C3_X, -10.0, C3_Z],
]
FIX_BOX = [-13, -1, -13, 13, 4, 13]

YOUNG_MODULUS = 50000
POISSON_RATIO = 0.45
TOTAL_MASS = 0.03


class TDCRController(Sofa.Core.Controller):
    """Interactive keyboard controller -- unchanged from tdcr_scene.py, used
    only in GUI mode. See tdcr_scene.py for the key bindings."""

    def __init__(self, cables, root, step=5.0):
        super().__init__()
        self.name = "TDCRController"
        self.root = root
        self.cables = cables
        self.step = step

    def _adjust(self, idx, delta):
        c = self.cables[idx].CableConstraint.value[0]
        new = max(0.0, c + delta)
        self.cables[idx].CableConstraint.value = [new]

    def onKeypressedEvent(self, event):
        key = event['key']
        if key == '1': self._adjust(0, +self.step)
        elif key == '!': self._adjust(0, -self.step)
        elif key == '2': self._adjust(1, +self.step)
        elif key == '@': self._adjust(1, -self.step)
        elif key == '3': self._adjust(2, +self.step)
        elif key == '#': self._adjust(2, -self.step)
        elif key == '4':
            for i in range(3): self._adjust(i, +self.step)
        elif key == '$':
            for i in range(3): self._adjust(i, -self.step)
        elif key == '0':
            for cab in self.cables: cab.CableConstraint.value = [0]


def createScene(rootNode, mesh_vtk=MESH_VTK, mesh_stl=MESH_STL,
                 include_visual=True, add_controller=True):
    rootNode.addObject('RequiredPlugin', pluginName=[
        "SoftRobots",
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Geometry",
        "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.StateContainer",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual",
        "MultiThreading",
        "Sofa.GL.Component.Rendering3D",
    ])

    rootNode.gravity.value = [0.0, -9.81, 0.0]
    rootNode.dt.value = 0.01
    rootNode.animate.value = True
    # DefaultVisualManagerLoop is required when the scene is built through the
    # Python API: runSofa adds one implicitly, but launching via Sofa.Gui does
    # not, and every frame then fails with "VisualLoop expected at the root
    # node". Harmless in batch mode, which never calls updateVisual.
    rootNode.addObject('DefaultVisualManagerLoop')
    rootNode.addObject('VisualStyle', displayFlags='showBehaviorModels showForceFields showCollisionModels showVisual')
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('GenericConstraintSolver', maxIterations=500, tolerance=1e-6,
                        computeConstraintForces=True)

    tdcr = rootNode.addChild('TDCR')
    tdcr.addObject('EulerImplicitSolver', rayleighMass=0.1, rayleighStiffness=0.1)
    # CGLinearSolver (matrix-free) segfaults under LinearSolverConstraintCorrection,
    # which needs an assembled matrix -- tdcr_scene.py flagged this same fix in a
    # comment but never applied it. SparseLDLSolver is a direct solver with an
    # assembled CompressedRowSparseMatrix, which LinearSolverConstraintCorrection
    # can actually use.
    tdcr.addObject('SparseLDLSolver', name='linear_solver', template='CompressedRowSparseMatrixMat3x3d')

    tdcr.addObject('MeshVTKLoader', name='loader', filename=mesh_vtk)
    tdcr.addObject('TetrahedronSetTopologyContainer', src='@loader', name='topo')
    tdcr.addObject('MechanicalObject', name='dofs', template='Vec3d')
    tdcr.addObject('UniformMass', totalMass=TOTAL_MASS)
    tdcr.addObject(FEM_COMPONENT,
                    youngModulus=YOUNG_MODULUS, poissonRatio=POISSON_RATIO, method='large')

    tdcr.addObject('BoxROI', name='fixBox', box=FIX_BOX, drawBoxes=True)
    tdcr.addObject('RestShapeSpringsForceField', points='@fixBox.indices', stiffness=1e8)
    tdcr.addObject('LinearSolverConstraintCorrection')

    if include_visual:
        visu = tdcr.addChild('visual')
        visu.addObject('MeshSTLLoader', name='stlLoader', filename=mesh_stl)
        visu.addObject('OglModel', name='oglModel', src='@stlLoader', color=[0.4, 0.7, 1.0, 0.8])
        visu.addObject('BarycentricMapping')

    cables = []
    names = ["Cable_0deg", "Cable_120deg", "Cable_240deg"]
    for i in range(3):
        cab = PullingCable(tdcr, names[i], pullPointLocation=PULL_PTS[i], cableGeometry=CABLE_PTS[i])
        cables.append(cab)

    if add_controller:
        rootNode.addObject(TDCRController(cables, rootNode, step=2.0))

    return rootNode, tdcr, cables


def find_tip_index(dofs, target=(0.0, BACKBONE_L, 0.0)):
    """Index of the mesh node nearest the ideal (undeformed) tip point."""
    positions = dofs.position.value
    best_i, best_d = 0, float("inf")
    tx, ty, tz = target
    for i, (x, y, z) in enumerate(positions):
        d = (x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def run_batch(configs, mesh_vtk=MESH_VTK, out_csv=os.path.join(_DIR, "results", "results.csv"),
              settle_steps=300, dt=0.01):
    """
    configs: list of (dl1, dl2, dl3) tuples, in mm pulled-in, matching
             cc_kinematics.py's sign convention.
    Writes out_csv with columns: dl1,dl2,dl3,sim_x,sim_y,sim_z,settle_steps
    """
    import Sofa.Simulation

    root = Sofa.Core.Node("root")
    _, tdcr, cables = createScene(root, mesh_vtk=mesh_vtk, include_visual=False, add_controller=False)
    Sofa.Simulation.init(root)
    root.dt.value = dt

    tip_idx = find_tip_index(tdcr.dofs)
    print(f"[cylinder_scene] tip node index = {tip_idx}, "
          f"rest position = {tuple(tdcr.dofs.position.value[tip_idx])}")

    rows = []
    for n, (dl1, dl2, dl3) in enumerate(configs):
        # Fresh mechanical state per sample -- avoids hysteresis carrying
        # over between DOE points (this is a quasi-static sweep, not a
        # trajectory).
        Sofa.Simulation.reset(root)
        cables[0].CableConstraint.value = [dl1]
        cables[1].CableConstraint.value = [dl2]
        cables[2].CableConstraint.value = [dl3]

        for _ in range(settle_steps):
            Sofa.Simulation.animate(root, root.dt.value)

        tip = tuple(tdcr.dofs.position.value[tip_idx])
        rows.append((dl1, dl2, dl3, *tip))
        print(f"[{n+1}/{len(configs)}] dl=({dl1:.2f},{dl2:.2f},{dl3:.2f}) -> tip={tip}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dl1", "dl2", "dl3", "sim_x", "sim_y", "sim_z"])
        w.writerows(rows)
    print(f"[cylinder_scene] wrote {len(rows)} rows to {out_csv}")
    return rows


def _load_configs_csv(path):
    configs = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            configs.append((float(row["dl1"]), float(row["dl2"]), float(row["dl3"])))
    return configs


if __name__ == "__main__":
    # argv after "--" when launched as: runSofa -l SofaPython3 sim/cylinder_scene.py -- --batch configs.csv
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", metavar="CONFIGS_CSV", default=None,
                     help="run headless over actuation samples from this CSV (columns dl1,dl2,dl3)")
    ap.add_argument("--out", default=os.path.join(_DIR, "results", "results.csv"))
    ap.add_argument("--mesh", default=MESH_VTK)
    ap.add_argument("--settle-steps", type=int, default=300)
    args, _unknown = ap.parse_known_args(argv)

    if args.batch:
        configs = _load_configs_csv(args.batch)
        run_batch(configs, mesh_vtk=args.mesh, out_csv=args.out, settle_steps=args.settle_steps)
    else:
        # Interactive GUI mode, same as tdcr_scene.py
        import time
        import Sofa.Gui
        import Sofa.Simulation
        import SofaRuntime

        root = Sofa.Core.Node("root")
        createScene(root)
        Sofa.Simulation.init(root)
        root.animate.value = True

        try:
            SofaRuntime.importPlugin("SofaGLFW")
        except Exception as e:
            print(f"[WARN] SofaGLFW: {e}")

        supported = Sofa.Gui.GUIManager.ListSupportedGUI(",")
        gui = "qt" if "qt" in supported else "batch"
        Sofa.Gui.GUIManager.Init("tdcr", gui)
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1280, 960)
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
