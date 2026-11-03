"""
Twist and bending test of one std6 module in SOFA: identifies the torsional compliance the model lacks (physics audit
C3) and checks the test against the bending compliance already identified from the tendon sweeps.

The module is clamped as in the sweeps (FIX_BOX, y in [-1, 4]), gravity off, no tendons, no contact. A pure couple is
applied to the nodes of the top face (y >= 159 mm), as self-balanced nodal forces: a torque about the backbone axis
(twist) or a moment about x or z (bending). Static solve, polar corotational FEM with the sweep material (E, nu from
tdcr/params.py). Every slab of nodes 2 mm thick around y = 20, 40, ..., 160 is fitted rigidly (Kabsch) to its rest
position; the rotation vectors are written out, per case and load.

    export SOFA_ROOT=<SOFA 25.06>; source sim/sofa_env.sh
    .venv/bin/python sim/twist_test.py --out results/twist_test_std6.json        # ~2 min, one SOFA process

The model side (plane-section prediction, identified bending) is scripts/torsion_identify.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
os.environ.setdefault("TDCR_SERIAL_FEM", "1")
os.environ.setdefault("TDCR_FEM_METHOD", "polar")
os.environ["TDCR_GRAVITY"] = "0"
os.environ["TDCR_CONTACT"] = "0"

import Sofa                                    # noqa: E402
import Sofa.Simulation                         # noqa: E402
import two_section_scene as ts                 # noqa: E402
from tdcr.paths import ROOT                    # noqa: E402

MESH = os.path.join(ROOT, "data", "meshes", "std6_one_section_sym.vtk")
TOP = 159.0                                    # nodes with y >= TOP carry the load
LEVELS = np.arange(20.0, 160.0 + 1e-9, 20.0)   # slabs [y - 1, y + 1] (the last one [159, 160])
AXES = {"twist": 1, "bend_x": 0, "bend_z": 2}  # couple about scene axis (y = backbone)


def couple_forces(P, axis, M):
    """Nodal forces on points P (k, 3), centroid-relative, whose resultant is zero and whose moment about the centroid
    is M along `axis` (mN mm)."""
    r = P - P.mean(0)
    e = np.zeros(3)
    e[axis] = 1.0
    if axis == 1:                              # torque: tangential forces, f_i = M (e x r_i) / sum |r_perp|^2
        rp = r.copy()
        rp[:, 1] = 0.0
        f = M * np.cross(e, rp) / (rp ** 2).sum()
    else:                                      # bending: axial forces linear across the face
        c = 2 if axis == 0 else 0              # moment about x from y-forces linear in z, about z linear in x
        sign = 1.0 if axis == 0 else -1.0      # (r x f)_x = -z f_y ; (r x f)_z = x f_y
        f = np.zeros_like(r)
        f[:, 1] = -sign * M * r[:, c] / (r[:, c] ** 2).sum()
    f -= f.mean(0)                             # exactly self-balanced
    return f


def build(P_rest, axis, M):
    root = Sofa.Core.Node("root")
    root.addObject("RequiredPlugin", pluginName=[
        "Sofa.Component.Engine.Select", "Sofa.Component.LinearSolver.Direct", "Sofa.Component.MechanicalLoad",
        "Sofa.Component.ODESolver.Backward", "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.SolidMechanics.Spring", "Sofa.Component.StateContainer", "Sofa.Component.IO.Mesh",
        "Sofa.Component.Topology.Container.Constant", "Sofa.Component.Topology.Container.Dynamic"])
    root.gravity.value = [0.0, 0.0, 0.0]
    root.addObject("DefaultAnimationLoop")
    body = root.addChild("body")
    body.addObject("NewtonRaphsonSolver", name="newton", maxNbIterationsNewton=10,
                   relativeInitialStoppingThreshold=1e-9, relativeSuccessiveStoppingThreshold=1e-9,
                   warnWhenLineSearchFails=False)
    body.addObject("StaticSolver", newtonSolver="@newton")
    body.addObject("SparseLDLSolver", name="linear_solver", template="CompressedRowSparseMatrixMat3x3d")
    body.addObject("MeshVTKLoader", name="loader", filename=MESH)
    body.addObject("TetrahedronSetTopologyContainer", src="@loader", name="topo")
    body.addObject("MechanicalObject", name="dofs", template="Vec3d")
    body.addObject(ts.FEM_COMPONENT, youngModulus=ts.YOUNG_MODULUS, poissonRatio=ts.POISSON_RATIO,
                   method=ts.FEM_METHOD)
    body.addObject("BoxROI", name="fixBox", box=ts.FIX_BOX)
    body.addObject("RestShapeSpringsForceField", points="@fixBox.indices", stiffness=1e8)
    top = np.flatnonzero(P_rest[:, 1] >= TOP)
    body.addObject("ConstantForceField", indices=top.tolist(),
                   forces=couple_forces(P_rest[top], axis, M).tolist())
    return root, body


def run_case(P_rest, axis, M, steps):
    root, body = build(P_rest, axis, M)
    Sofa.Simulation.init(root)
    top = np.flatnonzero(P_rest[:, 1] >= TOP)
    history = []                               # top rotation after every static solve: convergence check
    for _ in range(steps):
        Sofa.Simulation.animate(root, 1.0)
        X = np.array(body.dofs.position.value)
        history.append(ts._rotvec(ts.rigid_fit(P_rest[top], X[top])[:3, :3]).tolist())
    out = {"_history": history, "_status": str(body.newton.status.value)}
    for y in LEVELS:
        sel = np.flatnonzero(np.abs(P_rest[:, 1] - min(y, 159.0)) <= 1.0 + 1e-9) if y < 160 else \
            np.flatnonzero(P_rest[:, 1] >= TOP)
        T = ts.rigid_fit(P_rest[sel], X[sel])
        out[f"{y:g}"] = ts._rotvec(T[:3, :3]).tolist()
    Sofa.Simulation.unload(root)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "twist_test_std6.json"))
    ap.add_argument("--loads", type=float, nargs="+", default=[2000.0, 10000.0], help="couples (mN mm)")
    ap.add_argument("--steps", type=int, default=4, help="static solves (each up to 10 Newton iterations)")
    a = ap.parse_args()
    P = ts.read_vtk_points(MESH)
    res = dict(mesh=os.path.relpath(MESH, ROOT), young_kpa=ts.YOUNG_MODULUS, poisson=ts.POISSON_RATIO,
               fem=f"{ts.FEM_COMPONENT} {ts.FEM_METHOD}", fix_box=ts.FIX_BOX, top_y=TOP,
               n_top=int((P[:, 1] >= TOP).sum()), cases={})
    for name, axis in AXES.items():
        for M in a.loads:
            t0 = time.time()
            rot = run_case(P, axis, M, a.steps)
            res["cases"].setdefault(name, {})[f"{M:g}"] = rot
            print(f"{name:7s} M {M:7g} mN mm: rotation at the top {np.round(rot['160'], 7).tolist()} rad, change in "
                  f"the last solve {np.abs(np.diff(rot['_history'], axis=0)).max() if a.steps > 1 else 0:.1e}, "
                  f"{rot['_status']} ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
