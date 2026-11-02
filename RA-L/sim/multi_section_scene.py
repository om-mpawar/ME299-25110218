"""
N-section TDCR scene: the ROUTED two-section scene (two_section_scene.createSceneRouted)
generalised to any number of stacked sections, including one (the single-section CAD
sweep uses it too, so every section count runs the same scene).

Everything physical is the same as the two-section scene -- solver, constraint
handling, corotational FEM, material, base clamp, joint spring web, tendons as
cable constraints from a verified cables.json. The only generalisations:

  * the mesh holds N disconnected bodies (tools/stack_sections.py); the title
    line's "offsets=" says where each body's nodes start;
  * a joint spring web joins every pair of neighbouring sections, at each
    section boundary y = k * L;
  * 3N tendons, section k's three running from the base servos to section k's
    top through the SAME holes as every other section's (no routing twist);
  * total mass scales with N.

    from multi_section_scene import createSceneRoutedN
    createSceneRoutedN(root, mesh=".../four_section_cad.vtk",
                       cables_json=".../four_section_cad.cables.json", n_sections=4)
"""
import os

import numpy as np

import two_section_scene as ts  # noqa: E402  (shared constants and helpers)
from tendon import PullingCable  # noqa: E402  (pull-only: sim/tendon.py)
from tdcr.make_cables_json import load_cables_json  # noqa: E402
from tdcr.paths import ROOT  # noqa: E402


def read_offsets(path):
    """Start index of every body's nodes, from the mesh title line."""
    with open(path) as f:
        f.readline()
        title = f.readline()
    for tok in title.split():
        if tok.startswith("offsets="):
            return [int(v) for v in tok.split("=")[1].split(",")]
    try:
        return [0, ts.read_merge_offset(path)]      # two-section meshes from merge_vtk.py
    except RuntimeError:
        return [0]                                  # a single body (one section)


def joint_springs(pts, offsets, section_length, tol=ts.INTERFACE_TOL, k=ts.JOINT_NEIGHBOURS,
                  ks=ts.JOINT_STIFFNESS, kd=ts.JOINT_DAMPING):
    """Spring webs across every section boundary; same rule as the two-section joint:
    each interface node of the lower body is tied to its k nearest interface nodes
    of the upper body, rest length = initial separation."""
    idx = np.arange(len(pts))
    bounds = list(offsets) + [len(pts)]
    springs, report = [], []
    for s in range(len(offsets) - 1):
        yb = (s + 1) * section_length
        near = np.abs(pts[:, 1] - yb) < tol
        low = np.where(near & (idx >= bounds[s]) & (idx < bounds[s + 1]))[0]
        up = np.where(near & (idx >= bounds[s + 1]) & (idx < bounds[s + 2]))[0]
        if len(low) == 0 or len(up) == 0:
            raise RuntimeError(f"no interface nodes at y={yb} (lower {len(low)}, upper {len(up)})")
        for i in low:
            d = np.linalg.norm(pts[up] - pts[i], axis=1)
            # the k nearest, plus any tied with the k-th: on a mirror-symmetric mesh this
            # keeps the spring web exactly symmetric (ties are broken the same on both sides)
            dk = np.sort(d)[min(k, len(d)) - 1]
            for j in up[d <= dk + 1e-9]:
                springs.append([int(i), int(j), ks, kd, float(np.linalg.norm(pts[j] - pts[i]))])
        report.append(f"y={yb:g}: {len(low)}x{len(up)}")
    return springs, report


def createSceneRoutedN(rootNode, mesh, cables_json, n_sections, section_length=None,
                       joint_k=ts.JOINT_STIFFNESS):
    """Batch (no visual, no controller) N-section scene. -> (rootNode, body, offsets, cables)."""
    if section_length is None:
        from tdcr.cc_kinematics import BACKBONE_L
        section_length = BACKBONE_L
    rootNode.addObject('RequiredPlugin', pluginName=[
        "SoftRobots", "Sofa.Component.AnimationLoop",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Engine.Select", "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Mapping.Linear", "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.StateContainer", "Sofa.Component.IO.Mesh",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Topology.Mapping", "MultiThreading",
    ])
    rootNode.gravity.value = ts.gravity_vector()
    rootNode.dt.value = 0.01
    rootNode.animate.value = True
    if ts.CONTACT:
        ts.add_contact_pipeline(rootNode)
    rootNode.addObject('FreeMotionAnimationLoop')
    cs = dict(maxIterations=int(os.environ.get("TDCR_CS_ITER", 500)), tolerance=1e-6,
              computeConstraintForces=True)
    if os.environ.get("TDCR_CS_REG"):          # regularisation*Id added to the compliance W
        cs["regularizationTerm"] = float(os.environ["TDCR_CS_REG"])
    if os.environ.get("TDCR_CS_METHOD"):       # ProjectedGaussSeidel | UnbuiltGaussSeidel | NonsmoothNonlinearConjugateGradient
        cs["resolutionMethod"] = os.environ["TDCR_CS_METHOD"]
    rootNode.addObject('GenericConstraintSolver', name='constraint_solver', **cs)

    tdcr = rootNode.addChild('TDCR')
    ode = os.environ.get("TDCR_ODE", "euler")          # euler (default) | static
    if ode == "static":
        tdcr.addObject('StaticSolver', newton_iterations=int(os.environ.get("TDCR_NEWTON", 1)))
    else:
        rm, rs = (float(v) for v in os.environ.get("TDCR_RAYLEIGH", "0.1,0.1").split(","))
        tdcr.addObject('EulerImplicitSolver', rayleighMass=rm, rayleighStiffness=rs)
    lin = os.environ.get("TDCR_LINSOLVER", "SparseLDLSolver")     # benchmark switches
    kw = dict(name='linear_solver', template=os.environ.get("TDCR_LINTEMPLATE", 'CompressedRowSparseMatrixMat3x3d'))
    if os.environ.get("TDCR_PARALLEL_INVERSE") == "1":
        kw["parallelInverseProduct"] = True
    order = os.environ.get("TDCR_ORDERING")                        # AMD (SOFA's fallback) | METIS | COLAMD
    if order:
        metis = order.upper() == "METIS"                           # class is MetisOrderingMethod
        tdcr.addObject('RequiredPlugin', pluginName=['Sofa.Component.LinearSolver.Ordering']
                       + (['Sofa.Metis'] if metis else []))
        tdcr.addObject('MetisOrderingMethod' if metis else f'{order}OrderingMethod', name='ordering')
        kw["orderingMethod"] = '@ordering'
    if lin in ("SparseCholeskySolver", "SparseLUSolver"):
        tdcr.addObject('RequiredPlugin', pluginName=['CSparseSolvers'])
    tdcr.addObject(lin, **kw)
    tdcr.addObject('GenericConstraintCorrection', linearSolver='@linear_solver')

    body = tdcr.addChild('body')
    body.addObject('MeshVTKLoader', name='loader', filename=mesh)
    body.addObject('TetrahedronSetTopologyContainer', src='@loader', name='topo')
    if ts.CONTACT:
        body.addObject('TetrahedronSetTopologyModifier')
    body.addObject('MechanicalObject', name='dofs', template='Vec3d')
    _m = ts.mass_component(ts.MASS_MODEL, n_sections * ts.SECTION_MASS, mesh)
    body.addObject(_m[0], **_m[1])
    body.addObject(ts.FEM_COMPONENT, youngModulus=ts.YOUNG_MODULUS,
                   poissonRatio=ts.POISSON_RATIO, method=ts.FEM_METHOD)
    body.addObject('BoxROI', name='fixBox', box=ts.FIX_BOX)
    body.addObject('RestShapeSpringsForceField', points='@fixBox.indices', stiffness=1e8)

    pts = ts.read_vtk_points(mesh)
    offsets = read_offsets(mesh)
    if len(offsets) != n_sections:
        raise ValueError(f"{mesh} holds {len(offsets)} bodies, expected {n_sections}")
    springs, report = joint_springs(pts, offsets, section_length, ks=joint_k,
                                    kd=float(os.environ.get("TDCR_JOINT_KD", ts.JOINT_DAMPING)))
    print(f"[multi] {n_sections} sections, joints {'; '.join(report)} -> {len(springs)} springs @ k={joint_k:g}")
    body.addObject('StiffSpringForceField', name='joint', object1='@dofs', object2='@dofs', spring=springs)

    if ts.STRUTS:
        n = ts.add_notch_struts(body, mesh)
        print(f"[multi] notch struts on: {n} push-only constraints, gaps may close to {ts.STRUT_FRAC:.0%}")
    if ts.STOPS:
        n = ts.add_notch_stops(body, mesh)
        print(f"[multi] notch stops on: {n} pairs, engage at {ts.STOP_FRAC:.0%} of the gap, k={ts.STOP_K:g}")
    if ts.CONTACT:
        ts.add_self_contact(body, mesh)
        print(f"[multi] disc contact on: alarm {ts.CONTACT_ALARM} mm, separation {ts.CONTACT_DIST} mm")

    tendons = load_cables_json(cables_json)
    if len(tendons) != 3 * n_sections:
        raise ValueError(f"{cables_json}: expected {3 * n_sections} tendons, got {len(tendons)}")
    cables = [PullingCable(body, f"Cable_{t['name']}", pullPointLocation=t["servo"].tolist(),
                           cableGeometry=t["points"].tolist()) for t in tendons]
    print(f"[multi] {len(cables)} tendons from {os.path.relpath(cables_json, ROOT)}")
    return rootNode, body, offsets, cables
