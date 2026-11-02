"""
Level-1 virtual sensor for the 2-section / 6-tendon backbone, with full
SE(3) pose output.

Construction (confirmed against the lab hardware, 2026-09-07):

  * TWO SEPARATE BODIES joined at the section boundary, not one continuous
    backbone. Each section is its own mesh and its own MechanicalObject;
    they are coupled by a web of stiff springs across the interface.
    JOINT_STIFFNESS is therefore a real physical parameter of the model
    (the compliance of the connector between sections), not a numerical
    fudge -- and it is a good candidate for system identification, or for
    the residual calibration to absorb.

  * ALL SIX CABLES ORIGINATE AT THE BASE, because the servos are fixed at
    the bottom of the robot. Section-1 cables terminate at the section
    boundary; section-2 cables terminate at the tip.

  * Pose, not just position. Cavity insertion needs tip orientation, so
    each frame is recovered by fitting a rigid transform (Kabsch) to a
    CLUSTER of nodes rather than reading a single node. This mirrors how
    motion capture will fit a marker plate, keeping simulation and ground
    truth directly comparable, and avoids the single-nearest-node artefact
    that muddied the earlier mesh-convergence study.

KNOWN MODELLING LIMITATION -- read before trusting the coupling numbers
----------------------------------------------------------------------
A SOFA cable attaches to ONE MechanicalObject, so with two separate bodies
a section-2 tendon cannot be routed through section-1's material. Its
points live in section 2, and the run from the base servo up to section 2
is modelled as a free chord rather than a path routed through section 1's
channels.

Consequence: the two sections are still coupled mechanically (through the
joint, and through the chord's length changing as section 1 bends), but the
*geometric* tendon coupling -- the part multisection.py's lower-triangular
T matrix models -- is only approximated. The continuous-backbone model
represents that routing more faithfully; this two-body model represents the
joint more faithfully. That is a genuine trade-off, and which one is closer
to the hardware depends on whether the dominant effect is the connector's
compliance or the tendon routing. Worth quantifying by running both.

Running it
----------
The two modes need DIFFERENT launchers, for opposite reasons.

TO LOOK AT IT -- either of these works:

    python3 sim/two_section_scene.py                       # in-process, ImGui
    runSofa -g imgui -l SofaPython3 sim/two_section_scene.py

This install (SOFA v25.06.00) registers batch, glfw and imgui -- there is NO
Qt viewer, contrary to an earlier note in this file. SofaImGui is the one to
use. GLFW is present but renders its own backdrop instead of the scene.

The in-process path works because run_gui() now imports the GUI plugins
before asking which are supported; a bare Python process registers only
'batch' until one is loaded, which is what made this look impossible before.

runSofa loads this file by calling createScene(rootNode) with no arguments,
which is why the interactive configuration (visual models, camera, lights,
keyboard controller) is the DEFAULT rather than opt-in.

TO SWEEP IT -- use plain python3 (see cylinder_scene.py's docstring for the
PYTHONPATH/LD_LIBRARY_PATH setup). runSofa cannot do this: it never
executes the __main__ block, so it silently ignores --batch.

    python3 sim/two_section_scene.py --batch results/configs_2s.csv --out results/level1_2s.csv

Batch mode passes include_visual=False, add_controller=False explicitly, so
it pays none of the rendering cost.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tdcr`
import argparse
import csv
import math
import os
import sys

import numpy as np

# The multithreaded FEM component spins up a thread pool per process; several
# processes at once oversubscribe the cores and one can stall. For parallel
# sweeps set TDCR_SERIAL_FEM=1 to use the single-threaded component (same
# corotational FEM, same equations) and run one process per core.
FEM_COMPONENT = ("TetrahedronFEMForceField" if os.environ.get("TDCR_SERIAL_FEM")
                 else "ParallelTetrahedronFEMForceField")

import Sofa.Core
from tendon import PullingCable  # pull-only (minForce = 0): sim/tendon.py
from mass import MASS_MODEL, mass_component  # TDCR_MASS: uniform (default) or volume (sim/mass.py)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdcr.cc_kinematics import TENDON_R, THETA_DEG
from tdcr.multisection import SECTION_LENGTHS, SECTION_GAMMAS

from tdcr.paths import ROOT as _DIR  # paths below are relative to the repo root
MESH_S1 = os.path.join(_DIR, "data", "meshes", "section1.vtk")
MESH_S2 = os.path.join(_DIR, "data", "meshes", "section2.vtk")
# Merged single-body meshes -- see merge_vtk.py. These are what the ROUTED
# scene uses, because a cable can only attach to one MechanicalObject.
MESH_MERGED = os.path.join(_DIR, "data", "meshes", "two_section.vtk")
MESH_MERGED_LO = os.path.join(_DIR, "data", "meshes", "two_section_lo.vtk")
# Tendon paths. Written by make_cables_json.py; see that file for the format.
CABLES_JSON = os.path.join(_DIR, "data", "cables", "two_section.cables.json")

# ── Material / solver settings ─────────────────────────────────────────────
# Units: mm, kg, s  ->  force in mN, stress in kPa, gravity in mm/s^2.
# So E = 50000 is 50 MPa, and gravity must be 9810 to be real: the historical
# value 9.81 is 1000 times too weak (effectively no gravity), which is what every
# sweep up to 2026-09-30 used. Environment variables override the defaults so a
# run can use measured values without editing this file:
#   TDCR_YOUNG (kPa), TDCR_POISSON, TDCR_SECTION_MASS (kg), TDCR_GRAVITY (mm/s^2)
YOUNG_MODULUS = float(os.environ.get("TDCR_YOUNG", 50000))
POISSON_RATIO = float(os.environ.get("TDCR_POISSON", 0.45))
SECTION_MASS = float(os.environ.get("TDCR_SECTION_MASS", 0.015))   # kg per section
GRAVITY = float(os.environ.get("TDCR_GRAVITY", 9.81))              # magnitude; direction below
# TDCR_GRAVITY_DIR: direction of gravity in the LAB frame (z along the backbone, base at the bottom):
# -z upright (default, as before), +z hanging, +-x / +-y horizontal. The scene's y-up frame is the lab frame
# rotated (x_lab = x, y_lab = -z, z_lab = y), so a lab vector (gx, gy, gz) is (gx, gz, -gy) here.
GRAVITY_DIR = os.environ.get("TDCR_GRAVITY_DIR", "-z")


def gravity_vector():
    """Gravity in the scene frame (mm/s^2) from TDCR_GRAVITY and TDCR_GRAVITY_DIR."""
    d = GRAVITY_DIR.strip().lower()
    sign = -1.0 if d.startswith("-") else 1.0
    ax = d.lstrip("+-")
    lab = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[ax]
    gx, gy, gz = (sign * GRAVITY * v for v in lab)
    return [gx, gz, -gy]
# Contact between the discs (frictionless, solved with the tendon constraints). Without it
# the notch gaps can close and the faces pass through each other, which lets a soft, heavy
# backbone fold up (found 2026-09-30: 76 of 77 two-section poses fell over that way).
#   TDCR_CONTACT=1 turns it on; alarm = detection range, contact = kept separation (mm).
# Corotational formulation of the tetrahedral FEM: 'large' (SOFA's QR-based rotation from each
# element's vertex order; what every sweep so far used) or 'polar' (polar decomposition of the
# deformation gradient: independent of vertex order, so it respects the part's mirror symmetry).
FEM_METHOD = os.environ.get("TDCR_FEM_METHOD", "large")
CONTACT = os.environ.get("TDCR_CONTACT") == "1"
CONTACT_ALARM = float(os.environ.get("TDCR_CONTACT_ALARM", 0.15))
CONTACT_DIST = float(os.environ.get("TDCR_CONTACT_DIST", 0.05))
# Hard stops across the notch gaps (tools/make_notch_stops.py): one-sided springs between
# facing points that act only once a gap has closed to STOP_FRAC of its width, so a notch
# cannot close past its faces touching. A cheap stand-in for contact.  TDCR_STOPS=1.
STOPS = os.environ.get("TDCR_STOPS") == "1"
STOP_FRAC = float(os.environ.get("TDCR_STOP_FRAC", 0.1))
STOP_K = float(os.environ.get("TDCR_STOP_K", 2.0e4))        # mN/mm per stop pair
STOP_KD = float(os.environ.get("TDCR_STOP_KD", 0.0))        # damping per stop pair
# Constraint-based stops (the stable version): each strut is a two-point CableConstraint
# that may only PUSH (maxForce = 0), letting its gap close to STRUT_FRAC of the width and
# no further. Solved by the constraint solver together with the tendons.  TDCR_STRUTS=1.
STRUTS = os.environ.get("TDCR_STRUTS") == "1"
STRUT_FRAC = float(os.environ.get("TDCR_STRUT_FRAC", 0.1))
# Which struts to use (each one costs a constraint row every step): those with a rest gap of at
# most STRUT_MAXGAP mm (the rolling surfaces, which touch first), plus those of at least
# STRUT_RIM mm (outer rim, catches large closures). Defaults keep all.
STRUT_MAXGAP = float(os.environ.get("TDCR_STRUT_MAXGAP", "inf"))
STRUT_RIM = float(os.environ.get("TDCR_STRUT_RIM", "inf"))
CONTACT_MODELS = os.environ.get("TDCR_CONTACT_MODELS", "tlp")       # t(riangle) l(ine) p(oint)
# Split the facing faces into the two sides of the gaps (faces looking up / down), each on its
# own small mapped point set, so detection only pairs faces ACROSS a gap instead of testing the
# surface (and, through the point model, every node of the body) against itself.
CONTACT_SPLIT = os.environ.get("TDCR_CONTACT_SPLIT") == "1"
CONTACT_RESPONSE = os.environ.get("TDCR_CONTACT_RESPONSE", "FrictionContactConstraint")


def add_notch_stops(body, mesh):
    """One-sided springs across the notch gaps of this mesh. -> number of stops."""
    import numpy as _np
    f = os.path.join(_DIR, "data", "contacts", os.path.basename(mesh)[:-4] + ".stops.npz")
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f}: run tools/make_notch_stops.py {mesh}")
    d = _np.load(f)
    act = _np.sqrt(d["lateral"] ** 2 + (STOP_FRAC * d["gap0"]) ** 2)   # length at which a stop engages
    springs = [[int(i), int(j), STOP_K, STOP_KD, float(a)] for i, j, a in zip(d["i"], d["j"], act)]
    body.addObject('RepulsiveSpringForceField', name='notchStops', object1='@dofs', object2='@dofs',
                   spring=springs)
    return len(springs)


def add_notch_struts(body, mesh):
    """Push-only two-point constraints across the notch gaps (tools/make_notch_stops.py --struts),
    on the body's own nodes (no mapping). -> number of struts."""
    import numpy as _np
    f = os.path.join(_DIR, "data", "contacts", os.path.basename(mesh)[:-4] + ".struts.npz")
    if not os.path.exists(f):
        raise FileNotFoundError(f"{f}: run tools/make_notch_stops.py {mesh} --struts")
    d = dict(_np.load(f))
    sel = (d["gap0"] <= STRUT_MAXGAP) | (d["gap0"] >= STRUT_RIM)
    d = {k: v[sel] for k, v in d.items()}
    L0 = _np.sqrt(d["lateral"] ** 2 + d["gap0"] ** 2)                    # rest length of each strut
    allowed = L0 - _np.sqrt(d["lateral"] ** 2 + (STRUT_FRAC * d["gap0"]) ** 2)   # shortening allowed
    for k, (i, j, a) in enumerate(zip(d["i"], d["j"], allowed)):
        body.addObject('CableConstraint', name=f'strut{k}', indices=[int(i), int(j)], hasPullPoint=False,
                       value=[float(a)], valueType='displacement', maxForce=0.0)
    return len(allowed)


def add_contact_pipeline(rootNode):
    """Collision detection and frictionless contact response, on the root node."""
    rootNode.addObject('RequiredPlugin', name='contactPlugins', pluginName=[
        "Sofa.Component.Collision.Detection.Algorithm", "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Geometry", "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.Constraint.Lagrangian.Model", "Sofa.Component.Topology.Mapping"])
    rootNode.addObject('CollisionPipeline', name='collisionPipeline')
    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('LocalMinDistance', name='proximity', alarmDistance=CONTACT_ALARM,
                       contactDistance=CONTACT_DIST, angleCone=0.0)
    rootNode.addObject('CollisionResponse', response=CONTACT_RESPONSE,
                       responseParams='mu=0' if 'Constraint' in CONTACT_RESPONSE else '')


def add_self_contact(body, mesh=None, topo='@../topo'):
    """Collision models on the backbone surface, allowed to touch each other.

    If data/contacts/<mesh>.contact.npz exists (tools/make_contact_surface.py), only the
    faces that look at another disc across a narrow gap are used: about 4 % of the surface,
    which is what makes detection affordable. Otherwise the whole surface is used."""
    import numpy as _np
    c = body.addChild('contact')
    f = os.environ.get("TDCR_CONTACT_FILE") or (os.path.join(_DIR, "data", "contacts", os.path.basename(mesh)[:-4] + ".contact.npz") if mesh else "")
    if os.path.exists(f) and CONTACT_SPLIT:
        tri = _np.load(f)["triangles"]
        pts = _np.asarray(read_vtk_points(mesh), dtype=float)
        a, b, d = (pts[tri[:, k]] for k in range(3))
        up = _np.cross(b - a, d - a)[:, 1] > 0                       # outward normal along +y at rest
        for name, sel in (('up', up), ('down', ~up)):
            nodes = _np.unique(tri[sel])
            g = c.addChild(name)
            g.addObject('MechanicalObject', name='dofs', template='Vec3d', position=pts[nodes].tolist())
            g.addObject('TriangleSetTopologyContainer', name='tri', position=pts[nodes].tolist(),
                        triangles=_np.searchsorted(nodes, tri[sel]).tolist())
            g.addObject('SubsetMapping', input='@../../dofs', output='@dofs', indices=nodes.tolist())
            for key, model in (('t', 'TriangleCollisionModel'), ('l', 'LineCollisionModel'),
                               ('p', 'PointCollisionModel')):
                if key in CONTACT_MODELS:
                    g.addObject(model, selfCollision=False)
        print(f"[contact] {len(tri)} facing triangles from {os.path.relpath(f, _DIR)}, split "
              f"{int(up.sum())} up / {int((~up).sum())} down")
        return c
    if os.path.exists(f):
        tri = _np.load(f)["triangles"]
        c.addObject('TriangleSetTopologyContainer', name='tri', position='@../loader.position',
                    triangles=tri.tolist())
        print(f"[contact] {len(tri)} facing triangles from {os.path.relpath(f, _DIR)}")
    else:
        c.addObject('TriangleSetTopologyContainer', name='tri')
        c.addObject('TriangleSetTopologyModifier')
        c.addObject('Tetra2TriangleTopologicalMapping', input=topo, output='@tri')
        print("[contact] whole surface (no contact file for this mesh)")
    for key, model in (('t', 'TriangleCollisionModel'), ('l', 'LineCollisionModel'), ('p', 'PointCollisionModel')):
        if key in CONTACT_MODELS:
            c.addObject(model, selfCollision=True)
    return c
FIX_BOX = [-13, -1, -13, 13, 4, 13]

# Inter-section joint. PLACEHOLDER -- this is a real physical property of
# the connector and should be identified from hardware. Too soft and the
# sections visibly separate; too stiff and the linear system conditions
# badly. Swap the spring web for 'AttachProjectiveConstraint' if the real
# connector is effectively rigid.
JOINT_STIFFNESS = 1.0e5       # per spring
JOINT_DAMPING = 1.0
JOINT_NEIGHBOURS = 3          # section-2 partners per section-1 interface node
INTERFACE_TOL = 6.0           # mm band either side of the boundary

TIP_CLUSTER_SIZE = 8


# ── Cable routing ─────────────────────────────────────────────────────────

def _azimuths(section_index, gammas=SECTION_GAMMAS):
    """Tendon azimuths for one section, including its routing twist."""
    Gamma = float(np.cumsum(np.asarray(gammas, dtype=float))[section_index])
    return [math.radians(t) + Gamma for t in THETA_DEG]


def cable_geometry(section_index, r=TENDON_R, section_lengths=SECTION_LENGTHS,
                   gammas=SECTION_GAMMAS, spacing=15.0):
    """Cable polylines for one section, expressed in world coordinates.

    Points span only the section's own body, because that is the body the
    cable can attach to. The run down to the base servo is represented by
    the pull point, not by cable points.
    """
    y_start = float(np.sum(section_lengths[:section_index])) + 5.0
    y_end = float(np.sum(section_lengths[:section_index + 1])) - 5.0
    y_vals = list(np.arange(y_start, y_end, spacing)) + [y_end]

    out = []
    for a in _azimuths(section_index, gammas):
        x, z = r * math.cos(a), r * math.sin(a)
        out.append([[x, y, z] for y in y_vals])
    return out


def pull_points(section_index, r=TENDON_R, gammas=SECTION_GAMMAS):
    """All six cables are pulled from below the base -- the servos are
    fixed at the bottom of the robot, for both sections."""
    return [[r * math.cos(a), -10.0, r * math.sin(a)]
            for a in _azimuths(section_index, gammas)]


def cable_geometry_routed(section_index, r=TENDON_R,
                          section_lengths=SECTION_LENGTHS,
                          gammas=SECTION_GAMMAS, spacing=15.0):
    """Cable points along the ENTIRE run from the base to this section's tip.

    This is the corrected routing. cable_geometry() places points only inside
    the section a tendon terminates in, because with two separate bodies that
    was all a SOFA cable could attach to -- leaving a 235 mm unsupported
    chord from the base servo up to section 2, passing straight through
    section 1 without touching it.

    A real distal tendon runs in section 1's channels. It therefore
      (a) changes length when section 1 bends -- the coupling the
          lower-triangular T matrix in multisection.py models, and
      (b) presses on section 1's channel walls along the way.
    A free chord does neither, and the error lands in the tip position.

    Points are laid along every section up to and including this one, inset
    5 mm from each section boundary. The resulting 10 mm gap at the joint is
    deliberate: a cable point sitting exactly on the interface plane is
    ambiguous between the two bodies, and a short chord across the connector
    is a fair description of what the hardware does there anyway.
    """
    out = []
    for a in _azimuths(section_index, gammas):
        x, z = r * math.cos(a), r * math.sin(a)
        pts = []
        for j in range(section_index + 1):
            lo = float(np.sum(section_lengths[:j])) + 5.0
            hi = float(np.sum(section_lengths[:j + 1])) - 5.0
            ys = list(np.arange(lo, hi, spacing)) + [hi]
            pts += [[x, y, z] for y in ys]
        out.append(pts)
    return out


def _joint_springs_merged(pts, y_boundary, offset, tol=INTERFACE_TOL,
                          k=JOINT_NEIGHBOURS, ks=JOINT_STIFFNESS,
                          kd=JOINT_DAMPING):
    """Joint spring web between two bodies sharing ONE MechanicalObject.

    Identical physics to _joint_springs; the only difference is that both
    ends index into the same position array, so section 1's nodes are those
    below `offset` and section 2's are those at or above it. Merging the
    meshes changes what a cable can reach, NOT how the joint behaves.
    """
    near = np.abs(pts[:, 1] - y_boundary) < tol
    top = np.where(near & (np.arange(len(pts)) < offset))[0]
    bot = np.where(near & (np.arange(len(pts)) >= offset))[0]
    if len(top) == 0 or len(bot) == 0:
        raise RuntimeError(f"no interface nodes at y={y_boundary} "
                           f"(top {len(top)}, bottom {len(bot)})")

    springs = []
    for i in top:
        d = np.linalg.norm(pts[bot] - pts[i], axis=1)
        for j in bot[np.argsort(d)[:k]]:
            springs.append([int(i), int(j), ks, kd,
                            float(np.linalg.norm(pts[j] - pts[i]))])
    return springs, len(top), len(bot)


# ── Interactive controller (GUI mode only) ────────────────────────────────

class TwoSectionController(Sofa.Core.Controller):
    """Keyboard actuation for six tendons.

        1 2 3   pull section-1 tendons   (shift: ! @ #  to release)
        4 5 6   pull section-2 tendons   (shift: $ % ^  to release)
        7 / &   pull / release all three section-1 tendons together
        8 / *   pull / release all three section-2 tendons together
        0       reset every tendon to zero

    Watch what happens when you bend section 1 alone (key 1): the tip stays
    nearly upright, because the section-2 tendons are anchored at the base
    servos and hold it there. That is the inter-section coupling, and it is
    much easier to believe once you have seen it.
    """

    def __init__(self, cables, step=1.0, max_pull=25.0):
        super().__init__()
        self.name = "TwoSectionController"
        self.cables = cables
        self.step = step
        self.max_pull = max_pull

    def _adjust(self, idx, delta):
        c = self.cables[idx].CableConstraint
        c.value = [min(self.max_pull, max(0.0, c.value[0] + delta))]

    def _report(self):
        vals = [c.CableConstraint.value[0] for c in self.cables]
        print("  tendons  s1: " + " ".join(f"{v:5.1f}" for v in vals[:3])
              + "   |  s2: " + " ".join(f"{v:5.1f}" for v in vals[3:]))

    def onKeypressedEvent(self, event):
        key = event['key']
        pulls = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5}
        releases = {'!': 0, '@': 1, '#': 2, '$': 3, '%': 4, '^': 5}

        if key in pulls:
            self._adjust(pulls[key], +self.step)
        elif key in releases:
            self._adjust(releases[key], -self.step)
        elif key == '7':
            for i in range(3): self._adjust(i, +self.step)
        elif key == '&':
            for i in range(3): self._adjust(i, -self.step)
        elif key == '8':
            for i in range(3, 6): self._adjust(i, +self.step)
        elif key == '*':
            for i in range(3, 6): self._adjust(i, -self.step)
        elif key == '0':
            for c in self.cables:
                c.CableConstraint.value = [0.0]
        else:
            return
        self._report()


# ── Scene ─────────────────────────────────────────────────────────────────

def _add_section(parent, name, mesh, mass, clamp_base, visual_color=None):
    node = parent.addChild(name)
    node.addObject('MeshVTKLoader', name='loader', filename=mesh)
    node.addObject('TetrahedronSetTopologyContainer', src='@loader', name='topo')
    node.addObject('MechanicalObject', name='dofs', template='Vec3d')
    _m = mass_component(MASS_MODEL, mass, mesh)
    node.addObject(_m[0], **_m[1])
    node.addObject(FEM_COMPONENT,
                   youngModulus=YOUNG_MODULUS, poissonRatio=POISSON_RATIO, method=FEM_METHOD)

    if visual_color is not None:
        # Render the solid by extracting the tet mesh's boundary surface.
        # There is no separate surface STL per section, so the surface is
        # derived from the volume topology rather than loaded.
        visu = node.addChild('visu')
        visu.addObject('TriangleSetTopologyContainer', name='surfTopo')
        visu.addObject('TriangleSetTopologyModifier')
        visu.addObject('Tetra2TriangleTopologicalMapping',
                       input='@../topo', output='@surfTopo')
        visu.addObject('OglModel', name='visual', color=visual_color)
        visu.addObject('IdentityMapping', input='@../dofs', output='@visual')
    if clamp_base:
        node.addObject('BoxROI', name='fixBox', box=FIX_BOX, drawBoxes=True)
        node.addObject('RestShapeSpringsForceField', points='@fixBox.indices', stiffness=1e8)
    # NOTE: no per-section constraint correction here. Both bodies share one
    # assembled matrix, and giving each its own LinearSolverConstraintCorrection
    # makes each one correct the whole global system as though it owned it --
    # which is stable with no cables but diverges violently as soon as cable
    # constraints are active (verified: two bodies + joint alone is stable;
    # adding cables sends the tip to 1e6 mm regardless of joint stiffness).
    # A single GenericConstraintCorrection on the shared parent handles the
    # combined system correctly. See createScene.
    return node


def read_vtk_points(path):
    """Node coordinates from an ASCII VTK, as an (N, 3) array.

    Read straight from the file rather than from the MechanicalObject:
    a MeshVTKLoader has not populated its DOFs yet at scene-construction
    time (that happens during Sofa.Simulation.init), so the spring web has
    to be laid out from the mesh source.
    """
    vals, n_pts = [], None
    with open(path) as f:
        for line in f:
            if n_pts is None:
                if line.upper().startswith("POINTS"):
                    n_pts = int(line.split()[1])
                continue
            if len(vals) >= 3 * n_pts:
                break
            for tok in line.split():
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
    if n_pts is None:
        raise RuntimeError(f"no POINTS section in {path}")
    return np.array(vals[:3 * n_pts], dtype=float).reshape(n_pts, 3)


def _joint_springs(p1, p2, y_boundary, tol=INTERFACE_TOL,
                   k=JOINT_NEIGHBOURS, ks=JOINT_STIFFNESS, kd=JOINT_DAMPING):
    """Spring web tying section 1's top face to section 2's bottom face.

    Each interface node of section 1 is tied to its k nearest partners in
    section 2, not just its single closest. A one-to-one, nearly axial
    pairing would resist stretching but barely resist shear or bending at
    first order; a small web of non-parallel springs transmits moment
    across the joint, which is what a physical connector does.

    Rest lengths are the initial separations, so the joint starts
    unstressed rather than yanking the two meshes together.
    """
    top = np.where(np.abs(p1[:, 1] - y_boundary) < tol)[0]
    bot = np.where(np.abs(p2[:, 1] - y_boundary) < tol)[0]
    if len(top) == 0 or len(bot) == 0:
        raise RuntimeError(f"no interface nodes found at y={y_boundary} "
                           f"(section1 {len(top)}, section2 {len(bot)})")

    springs = []
    for i in top:
        d = np.linalg.norm(p2[bot] - p1[i], axis=1)
        for j in bot[np.argsort(d)[:k]]:
            L0 = float(np.linalg.norm(p2[j] - p1[i]))
            springs.append([int(i), int(j), ks, kd, L0])
    return springs, len(top), len(bot)


def createScene(rootNode, mesh_s1=MESH_S1, mesh_s2=MESH_S2, joint_k=JOINT_STIFFNESS,
                cable_sections=(0, 1), include_visual=True, add_controller=True,
                include_camera=False):
    rootNode.addObject('RequiredPlugin', pluginName=[
        "SoftRobots",
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.StateContainer",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Topology.Mapping",
        "Sofa.Component.Visual",
        "Sofa.Component.Setting",
        "Sofa.GL.Component.Rendering3D",
        "Sofa.GL.Component.Shader",
        "MultiThreading",
    ])

    if include_visual:
        # Required for GUI mode. runSofa adds a visual loop implicitly, but a
        # scene built through the Python API does not get one, and every
        # frame then fails with "VisualLoop expected at the root node".
        rootNode.addObject('DefaultVisualManagerLoop')
        rootNode.addObject('VisualStyle',
                           displayFlags='showVisual showBehaviorModels '
                                        'showInteractionForceFields')

        # Lighting and camera are also runSofa's job, and also absent when the
        # scene is built from Python. Without a light the OglModel surfaces
        # shade black; without a camera framing it, a 220 mm object sits
        # outside the default view. Either way the window comes up empty.
        rootNode.addObject('BackgroundSetting', color=[0.16, 0.18, 0.21, 1.0])
        rootNode.addObject('LightManager')
        rootNode.addObject('DirectionalLight', direction=[1, -0.4, 1],
                           color=[1.0, 1.0, 1.0, 1.0])
        rootNode.addObject('DirectionalLight', direction=[-1, -0.2, -0.6],
                           color=[0.45, 0.5, 0.6, 1.0])

        # NO CAMERA BY DEFAULT, deliberately.
        #
        # A SOFA viewer frames the scene automatically (viewAll) only when the
        # scene does NOT define a camera. Supplying an InteractiveCamera takes
        # that job away and pins the view to whatever the scene says -- and
        # SofaImGui then rendered an empty viewport, because its viewport
        # camera is not the scene camera. Leaving it out lets the viewer do
        # what it is good at: work out the bounding box and frame it.
        #
        # Pass include_camera=True (or --camera) to go back to the explicit
        # one, which is still useful for reproducible screenshots.
        if include_camera:
            y_mid = float(np.sum(SECTION_LENGTHS)) * 0.5
            span = float(np.sum(SECTION_LENGTHS))
            rootNode.addObject('InteractiveCamera', name='camera',
                               position=[span * 1.5, y_mid * 1.35, span * 1.5],
                               lookAt=[0.0, y_mid * 0.85, 0.0],
                               fieldOfView=45, zNear=1.0, zFar=5.0 * span)

    rootNode.gravity.value = gravity_vector()
    rootNode.dt.value = 0.01
    rootNode.animate.value = True
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('GenericConstraintSolver', maxIterations=500, tolerance=1e-6,
                       computeConstraintForces=True)

    # Both bodies share one implicit solve, so the joint is resolved inside
    # the timestep rather than lagging a step behind.
    tdcr = rootNode.addChild('TDCR')
    tdcr.addObject('EulerImplicitSolver', rayleighMass=0.1, rayleighStiffness=0.1)
    tdcr.addObject('SparseLDLSolver', name='linear_solver',
                   template='CompressedRowSparseMatrixMat3x3d')
    # ONE constraint correction for the whole two-body system (see _add_section).
    tdcr.addObject('GenericConstraintCorrection', linearSolver='@linear_solver')

    s1 = _add_section(tdcr, 'section1', mesh_s1, SECTION_MASS, clamp_base=True,
                      visual_color=[0.35, 0.62, 0.95, 1.0] if include_visual else None)
    s2 = _add_section(tdcr, 'section2', mesh_s2, SECTION_MASS, clamp_base=False,
                      visual_color=[0.95, 0.55, 0.25, 1.0] if include_visual else None)

    p1 = read_vtk_points(mesh_s1)
    p2 = read_vtk_points(mesh_s2)
    y_boundary = float(SECTION_LENGTHS[0])
    springs, n_top, n_bot = _joint_springs(p1, p2, y_boundary, ks=joint_k)
    print(f"[two_section] joint: {n_top} x {n_bot} interface nodes -> {len(springs)} springs "
          f"@ k={joint_k:g}")

    tdcr.addObject('StiffSpringForceField', name='joint',
                   object1='@section1/dofs', object2='@section2/dofs',
                   spring=springs)

    # Cables, in dl1..dl6 order: section 1 first, then section 2.
    # cable_sections exists so a section's tendons can be omitted entirely --
    # the only way to test what the distal tendons contribute, since a cable
    # commanded to zero is still an active inextensibility constraint tethering
    # the distal section to the fixed base pull point, not a slack string.
    cables = []
    for sec, node in ((0, s1), (1, s2)):
        if sec not in cable_sections:
            continue
        geoms = cable_geometry(sec)
        pulls = pull_points(sec)
        for k in range(3):
            cables.append(PullingCable(
                node, f"Cable_s{sec+1}_{int(THETA_DEG[k])}deg",
                pullPointLocation=pulls[k], cableGeometry=geoms[k]))

    if add_controller:
        rootNode.addObject(TwoSectionController(cables))

    return rootNode, s1, s2, cables


def read_merge_offset(path):
    """Index of the first section-2 node in a merged mesh (from its title)."""
    with open(path) as f:
        f.readline()
        title = f.readline()
    for tok in title.split():
        if tok.startswith("offset="):
            return int(tok.split("=")[1])
    raise RuntimeError(f"{path} has no 'offset=' in its title line -- "
                       "regenerate it with merge_vtk.py")


def createSceneRouted(rootNode, mesh=MESH_MERGED, joint_k=JOINT_STIFFNESS,
                      include_visual=True, add_controller=True,
                      include_camera=False, cables_json=None):
    """Two-section scene with CORRECTLY ROUTED tendons.

    Difference from createScene: both sections live in ONE MechanicalObject,
    loaded from a merged mesh. That is the only way a SOFA cable can have
    points in both -- and without it a section-2 tendon is a 235 mm chord
    hanging in free space through section 1, which corrupts the tip position.

    The two bodies remain mechanically separate. The merged mesh has two
    disconnected tetrahedral components with no shared nodes, so the only
    thing coupling them is still the joint spring web. The joint model is
    unchanged; only the address space the cables live in.
    """
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
        "Sofa.Component.Topology.Mapping", "Sofa.Component.Visual",
        "Sofa.Component.Setting", "Sofa.GL.Component.Rendering3D",
        "Sofa.GL.Component.Shader", "MultiThreading",
    ])

    if include_visual:
        rootNode.addObject('DefaultVisualManagerLoop')
        rootNode.addObject('VisualStyle',
                           displayFlags='showVisual showBehaviorModels '
                                        'showInteractionForceFields')
        rootNode.addObject('BackgroundSetting', color=[0.16, 0.18, 0.21, 1.0])
        rootNode.addObject('LightManager')
        rootNode.addObject('DirectionalLight', direction=[1, -0.4, 1],
                           color=[1.0, 1.0, 1.0, 1.0])
        rootNode.addObject('DirectionalLight', direction=[-1, -0.2, -0.6],
                           color=[0.45, 0.5, 0.6, 1.0])
        if include_camera:
            span = float(np.sum(SECTION_LENGTHS))
            rootNode.addObject('InteractiveCamera', name='camera',
                               position=[span * 1.5, span * 0.68, span * 1.5],
                               lookAt=[0.0, span * 0.43, 0.0],
                               fieldOfView=45, zNear=1.0, zFar=5.0 * span)

    rootNode.gravity.value = gravity_vector()
    rootNode.dt.value = 0.01
    rootNode.animate.value = True
    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('GenericConstraintSolver', maxIterations=500,
                       tolerance=1e-6, computeConstraintForces=True)

    tdcr = rootNode.addChild('TDCR')
    tdcr.addObject('EulerImplicitSolver', rayleighMass=0.1, rayleighStiffness=0.1)
    tdcr.addObject('SparseLDLSolver', name='linear_solver',
                   template='CompressedRowSparseMatrixMat3x3d')
    tdcr.addObject('GenericConstraintCorrection', linearSolver='@linear_solver')

    body = tdcr.addChild('body')
    body.addObject('MeshVTKLoader', name='loader', filename=mesh)
    body.addObject('TetrahedronSetTopologyContainer', src='@loader', name='topo')
    body.addObject('MechanicalObject', name='dofs', template='Vec3d')
    _m = mass_component(MASS_MODEL, 2.0 * SECTION_MASS, mesh)
    body.addObject(_m[0], **_m[1])
    body.addObject(FEM_COMPONENT,
                   youngModulus=YOUNG_MODULUS, poissonRatio=POISSON_RATIO,
                   method=FEM_METHOD)
    body.addObject('BoxROI', name='fixBox', box=FIX_BOX, drawBoxes=True)
    body.addObject('RestShapeSpringsForceField', points='@fixBox.indices',
                   stiffness=1e8)

    if include_visual:
        visu = body.addChild('visu')
        visu.addObject('TriangleSetTopologyContainer', name='surfTopo')
        visu.addObject('TriangleSetTopologyModifier')
        visu.addObject('Tetra2TriangleTopologicalMapping',
                       input='@../topo', output='@surfTopo')
        visu.addObject('OglModel', name='visual', color=[0.35, 0.62, 0.95, 1.0])
        visu.addObject('IdentityMapping', input='@../dofs', output='@visual')

    pts = read_vtk_points(mesh)
    offset = read_merge_offset(mesh)
    springs, n_top, n_bot = _joint_springs_merged(
        pts, float(SECTION_LENGTHS[0]), offset, ks=joint_k)
    print(f"[routed] joint: {n_top} x {n_bot} interface nodes -> "
          f"{len(springs)} springs @ k={joint_k:g}")
    body.addObject('StiffSpringForceField', name='joint',
                   object1='@dofs', object2='@dofs', spring=springs)

    # Tendon paths come from a cables.json when one is given (or the default
    # one exists), so the path SOFA uses is the path you can open and read.
    # Falls back to computing them from TENDON_R / THETA_DEG otherwise.
    path = cables_json if cables_json is not None else (
        CABLES_JSON if os.path.exists(CABLES_JSON) else None)
    cables = []
    if path:
        from tdcr.make_cables_json import load_cables_json
        tendons = load_cables_json(path)
        if len(tendons) != 6:
            raise ValueError(f"{path}: expected 6 tendons (dl1..dl6), got {len(tendons)}")
        for t in tendons:
            cables.append(PullingCable(
                body, f"Cable_{t['name']}",
                pullPointLocation=t["servo"].tolist(),
                cableGeometry=t["points"].tolist()))
        print(f"[routed] cables from {os.path.relpath(path, _DIR)}: "
              + ", ".join(f"{t['name']} {len(t['points'])}pt" for t in tendons))
    else:
        for sec in (0, 1):
            geoms = cable_geometry_routed(sec)
            pulls = pull_points(sec)
            for k in range(3):
                cables.append(PullingCable(
                    body, f"Cable_s{sec+1}_{int(THETA_DEG[k])}deg",
                    pullPointLocation=pulls[k], cableGeometry=geoms[k]))
        print(f"[routed] cables computed (no cables.json): s1 "
              f"{len(cable_geometry_routed(0)[0])} pts, s2 "
              f"{len(cable_geometry_routed(1)[0])} pts each")

    if add_controller:
        rootNode.addObject(TwoSectionController(cables))

    return rootNode, body, offset, cables


# ── Rigid frame extraction ────────────────────────────────────────────────

def find_cluster(positions, target, k=TIP_CLUSTER_SIZE):
    p = np.asarray(positions, dtype=float)
    d = np.linalg.norm(p - np.asarray(target, dtype=float), axis=1)
    return np.argsort(d)[:k]


def rigid_fit(rest_pts, cur_pts):
    """Kabsch best-fit rigid transform taking rest_pts onto cur_pts.

    The same computation a motion-capture system runs on a marker cluster.
    """
    P = np.asarray(rest_pts, dtype=float)
    Q = np.asarray(cur_pts, dtype=float)
    pc, qc = P.mean(axis=0), Q.mean(axis=0)

    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    # Translation is the cluster's CURRENT CENTROID -- i.e. where the frame
    # actually is -- not the transform's translation component (qc - R @ pc),
    # which is a different quantity once there is any rotation and is not
    # comparable to a Level-0 tip position or to a mocap rigid-body origin.
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = qc
    return T


def _rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


# ── Batch sweep ───────────────────────────────────────────────────────────

def run_batch(configs, mesh_s1=MESH_S1, mesh_s2=MESH_S2,
              out_csv=os.path.join(_DIR, "results", "level1_2s.csv"),
              settle_steps=300, dt=0.01, joint_k=JOINT_STIFFNESS,
              passthrough_header=(), passthrough_rows=None):
    """configs: list of 6-tuples (dl1..dl6) in mm pulled in.

    Poses are logged as position + rotation vector (absolute, base frame).
    Rotation vectors rather than Euler angles: no wrapping or gimbal-lock
    discontinuities, which matters once these become GP targets.
    """
    import Sofa.Simulation

    root = Sofa.Core.Node("root")
    _, s1, s2, cables = createScene(root, mesh_s1=mesh_s1, mesh_s2=mesh_s2, joint_k=joint_k,
                                   include_visual=False, add_controller=False)
    Sofa.Simulation.init(root)
    root.dt.value = dt

    rest1 = np.array(s1.dofs.position.value, dtype=float)
    rest2 = np.array(s2.dofs.position.value, dtype=float)
    y_mid = float(SECTION_LENGTHS[0])
    y_tip = float(np.sum(SECTION_LENGTHS))

    mid_idx = find_cluster(rest1, (0.0, y_mid, 0.0))
    tip_idx = find_cluster(rest2, (0.0, y_tip, 0.0))
    mid_rest, tip_rest = rest1[mid_idx], rest2[tip_idx]

    print(f"[two_section] mid cluster centroid {np.round(mid_rest.mean(0), 3)}")
    print(f"[two_section] tip cluster centroid {np.round(tip_rest.mean(0), 3)}")

    rows = []
    for n, cfg in enumerate(configs):
        if len(cfg) != 6:
            raise ValueError(f"expected 6 tendon values, got {len(cfg)}")

        Sofa.Simulation.reset(root)
        for c, v in zip(cables, cfg):
            c.CableConstraint.value = [float(v)]

        for _ in range(settle_steps):
            Sofa.Simulation.animate(root, root.dt.value)

        cur1 = np.array(s1.dofs.position.value, dtype=float)
        cur2 = np.array(s2.dofs.position.value, dtype=float)
        T_mid = rigid_fit(mid_rest, cur1[mid_idx])
        T_tip = rigid_fit(tip_rest, cur2[tip_idx])

        extra = list(passthrough_rows[n]) if passthrough_rows else []
        rows.append(extra + list(cfg)
                    + list(T_mid[:3, 3]) + list(_rotvec(T_mid[:3, :3]))
                    + list(T_tip[:3, 3]) + list(_rotvec(T_tip[:3, :3])))

        print(f"[{n+1}/{len(configs)}] dl=({','.join(f'{v:.1f}' for v in cfg)}) "
              f"-> mid={np.round(T_mid[:3,3],1)} tip={np.round(T_tip[:3,3],1)} "
              f"dir={np.round(T_tip[:3,1],3)}")

    header = (list(passthrough_header)
              + ["dl1", "dl2", "dl3", "dl4", "dl5", "dl6"]
              + ["mid_x", "mid_y", "mid_z", "mid_wx", "mid_wy", "mid_wz"]
              + ["tip_x", "tip_y", "tip_z", "tip_wx", "tip_wy", "tip_wz"])

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[two_section] wrote {len(rows)} rows to {out_csv}")
    return rows


def _load_configs_csv(path):
    """-> (configs, passthrough_header, passthrough_rows).

    Anything in the configs file that is NOT a dl column is carried straight
    through to the results file. That matters because the DOE is specified in
    (theta, alpha) and the analysis downstream -- explicit_model.py,
    plot_3d.py, compare_priors.py -- needs those columns to sit beside the
    poses they produced. Joining them back on afterwards by row index works
    right up until someone reorders or reruns a partial sweep, and then it
    silently mislabels every sample.
    """
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        extra = [c for c in r.fieldnames
                 if not (c.startswith("dl") and c[2:].isdigit())]
        configs, rows = [], []
        for row in r:
            configs.append(tuple(float(row[f"dl{i+1}"]) for i in range(6)))
            rows.append([row[c] for c in extra])
    return configs, extra, rows


def run_gui(mesh_s1=MESH_S1, mesh_s2=MESH_S2, joint_k=JOINT_STIFFNESS,
            include_camera=False, routed=True, routed_mesh=MESH_MERGED,
            cables_json=None):
    """Open the interactive viewer and drive the tendons from the keyboard."""
    import Sofa.Gui
    import Sofa.Simulation
    import SofaRuntime

    root = Sofa.Core.Node("root")
    if routed:
        _, body, _off, _ = createSceneRouted(root, mesh=routed_mesh,
                                             joint_k=joint_k,
                                             include_visual=True,
                                             add_controller=True,
                                             include_camera=include_camera,
                                             cables_json=cables_json)
        bodies = [body]
    else:
        _, s1, s2, _ = createScene(root, mesh_s1=mesh_s1, mesh_s2=mesh_s2,
                                   joint_k=joint_k, include_visual=True,
                                   add_controller=True,
                                   include_camera=include_camera)
        bodies = [s1, s2]
    Sofa.Simulation.init(root)

    # PRIME THE BOUNDING BOX BEFORE HANDING THE SCENE TO THE VIEWER.
    #
    # This is why the viewport came up empty. A SOFA viewer frames the scene
    # with viewAll() exactly once, at startup, and viewAll reads the ROOT
    # node's bounding box. Straight after Sofa.Simulation.init that box is
    # still the uninitialised sentinel -- min +1.8e308, max -1.8e308, i.e.
    # inverted -- even though every child node already has a correct one.
    # The camera is therefore framed from a degenerate box, lands nowhere,
    # and never recovers, because viewAll is not called again.
    #
    # Stepping the simulation populates it (measured: invalid after init,
    # valid after 5 steps), and setting it explicitly from the actual node
    # positions makes it independent of that timing.
    for _ in range(5):
        Sofa.Simulation.animate(root, root.dt.value)
    Sofa.Simulation.updateVisual(root)

    pts = np.vstack([np.array(b.dofs.position.value, dtype=float)
                     for b in bodies])
    margin = 0.05 * float(np.ptp(pts, axis=0).max())
    root.bbox.value = [(pts.min(axis=0) - margin).tolist(),
                       (pts.max(axis=0) + margin).tolist()]
    print(f"[two_section] scene bbox for the viewer: "
          f"{np.round(root.bbox.value, 1).tolist()}")

    root.animate.value = True

    # Load the GUI backends before asking what is available: a bare Python
    # process registers only 'batch' until a GUI plugin is imported.
    for plugin in ("SofaGLFW", "SofaImGui"):
        try:
            SofaRuntime.importPlugin(plugin)
        except Exception as e:
            print(f"[WARN] {plugin}: {e}")

    print(TwoSectionController.__doc__)

    supported = Sofa.Gui.GUIManager.ListSupportedGUI(",")
    # Preference order, checked on this install (SOFA v25.06.00):
    #   imgui  SofaImGui, the current SOFA viewer -- use this
    #   qt     absent here; earlier notes in this project claimed runSofa
    #          registers it, which is NOT true of this build
    #   glfw   present but renders its own backdrop rather than the scene
    for gui in ("imgui", "qt", "glfw", "batch"):
        if gui in supported:
            break
    print(f"[two_section] available GUIs: {supported} -> using '{gui}'")
    if gui in ("glfw", "batch"):
        print("[two_section] WARNING: falling back to a backend that will not "
              "show the scene properly. Is SofaImGui installed?")

    Sofa.Gui.GUIManager.Init("two_section", gui)
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1280, 960)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", metavar="CONFIGS_CSV", default=None,
                    help="headless sweep over this CSV; omit to open the viewer")
    ap.add_argument("--out", default=os.path.join(_DIR, "results", "level1_2s.csv"))
    ap.add_argument("--mesh-s1", default=MESH_S1)
    ap.add_argument("--mesh-s2", default=MESH_S2)
    ap.add_argument("--settle-steps", type=int, default=300)
    ap.add_argument("--joint-k", type=float, default=JOINT_STIFFNESS)
    ap.add_argument("--lo", action="store_true",
                    help="coarse meshes for interactive viewing (1.9k tets vs 20.5k). "
                         "VIEWING ONLY -- never use for results without gating "
                         "them through mesh_isotropy_check.py first")
    ap.add_argument("--mid", action="store_true",
                    help="mid-resolution meshes, between --lo and the default")
    ap.add_argument("--free-chord", action="store_true",
                    help="the OLD two-body model, where section-2 tendons hang "
                         "as a 235 mm chord through section 1. Physically wrong; "
                         "kept only to quantify what that approximation costs")
    ap.add_argument("--cables", default=None,
                    help="tendon paths (cables.json). Default: "
                         "data/cables/two_section.cables.json if it exists")
    ap.add_argument("--cad", action="store_true",
                    help="CAD meshes (the real part) instead of the cylinder")
    ap.add_argument("--camera", action="store_true",
                    help="use the scene's explicit camera instead of letting "
                         "the viewer auto-frame")
    args, _ = ap.parse_known_args(argv)
    routed_mesh = MESH_MERGED
    if args.cad:
        routed_mesh = os.path.join(_DIR, "data", "meshes", "two_section_cad.vtk")
    elif args.lo:
        args.mesh_s1 = os.path.join(_DIR, "data", "meshes", "section1_lo.vtk")
        args.mesh_s2 = os.path.join(_DIR, "data", "meshes", "section2_lo.vtk")
        routed_mesh = MESH_MERGED_LO
    elif args.mid:
        args.mesh_s1 = os.path.join(_DIR, "data", "meshes", "section1_mid.vtk")
        args.mesh_s2 = os.path.join(_DIR, "data", "meshes", "section2_mid.vtk")

    if args.batch:
        cfgs, extra_hdr, extra_rows = _load_configs_csv(args.batch)
        run_batch(cfgs, mesh_s1=args.mesh_s1, mesh_s2=args.mesh_s2,
                  out_csv=args.out, settle_steps=args.settle_steps,
                  joint_k=args.joint_k, passthrough_header=extra_hdr,
                  passthrough_rows=extra_rows)
    else:
        run_gui(mesh_s1=args.mesh_s1, mesh_s2=args.mesh_s2, joint_k=args.joint_k,
                include_camera=args.camera, routed=not args.free_chord,
                routed_mesh=routed_mesh, cables_json=args.cables)
