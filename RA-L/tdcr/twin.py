"""
Real-time, sensor-free digital twin of the notched TDCR: tendon displacements + the robot's mounting -> the whole
backbone shape.

    from tdcr.twin import Twin
    tw = Twin(4, "hanging", calib_json="results/calib_<session>.json")
    s = tw.shape(dl)            # dl: (12,) mm, + wound in, lab tendon order (the servo commands)
    s.positions                 # (K, 3) mm, every 5 mm along the backbone, lab frame (z along the backbone)
    s.rotations                 # (K, 3, 3) cross-section frames, lab frame (identity when the robot is straight)

The model is the variable-section Euler-Bernoulli core (tdcr/eb_prior.py) with the module compliance identified from
SOFA FEM (tdcr/compliance.py, scripts/module_compliance.py -> results/mc_std6.json) and, optionally, the few-shot hardware calibration
(tdcr/calibrate.py). Pull-only tendons, coupling between sections and gravity from the mounting are solved together;
consecutive calls warm-start the gravity solve, so following the servos live costs a few milliseconds.

Frames. Lab: z along the backbone (base at the origin, the robot upright along +z), x towards tendon 1 of section 1
(lab azimuth 0). The model works in the SOFA scene frame (y along the backbone): lab (x, y, z) = scene (x, -z, y).
Mounting = the direction of gravity in the lab frame, given once (no sensor): MOUNTINGS or any vector; (0, 0, 0) is
gravity-free.

Joints. The printed coupler (7 mm) is gap_mm = 2 mm longer than the 5 mm web the 160 mm module already contains: a
straight rigid step at every joint (module pitch 162 mm), with the coupler + dowel + plugs as joint_mass_kg there.
Stations stay material coordinates (every 5 mm of module, 32 per module); a module's end station is its top face,
before the joint.

Friction (optional, calibration "mu" > 0): capstan friction in the channels makes the shape depend on whether each
tendon was last wound in or paid out. The twin takes that direction from its own command history (the robot starts
homed at zero, so every tendon starts wound in) or from shape(..., direction=). A tendon has to move DEADBAND_MM
from its last registered position before its direction changes, so encoder jitter does not flip it.
"""
import json
import os
import time
from dataclasses import dataclass

import numpy as np

from tdcr.compliance import G_MAG, HOLES6, MASS_STD6, ModuleModel, lab_to_scene
from tdcr.eb_prior import geometry_from_cables, predict_gravity
from tdcr.params import STD6
from tdcr.paths import CABLES, ROOT

MOUNTINGS = {"upright": (0.0, 0.0, -1.0), "hanging": (0.0, 0.0, 1.0),
             "horizontal_x": (1.0, 0.0, 0.0), "horizontal_y": (0.0, 1.0, 0.0)}
MODULE_L = STD6.module_length_mm
_CABLE_FILES = {1: "one", 2: "two", 4: "four"}
DEFAULT_COMPLIANCE = "results/mc_std6.json"
DEADBAND_MM = 0.025          # friction direction deadband: ~3 encoder ticks of a 5 mm pulley (0.0077 mm per tick)
# lab = M @ scene for vectors; body frames are re-expressed with the same change of axes: R_lab = M R_scene M^T
M_SCENE_TO_LAB = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def scene_to_lab(v):
    """Points or vectors (..., 3) from the scene frame (y along the backbone) to the lab frame (z along it)."""
    return np.asarray(v, float) @ M_SCENE_TO_LAB.T


@dataclass
class TwinShape:
    positions: np.ndarray       # (K, 3) mm, stations every 5 mm from the base
    rotations: np.ndarray       # (K, 3, 3) cross-section frames
    tensions: np.ndarray        # (3n,) tendon tension / E (mm^2), lab tendon order; 0 = slack
    converged: bool             # False: the gravity solve did not settle (e.g. a stack past its buckling load)
    solve_ms: float


class Twin:
    """Digital twin of an n-module robot in a fixed mounting. One object per robot configuration; call shape() as
    often as the servos update."""

    def __init__(self, n_modules, mounting, compliance_json=None, calib_json=None,
                 mass_kg=MASS_STD6, tol=1e-4, gap_mm=0.0, joint_mass_kg=0.0, tip_mass_kg=0.0, mu=0.0,
                 cad_json=None):
        """calib_json: path of a tdcr/calibrate.py result, or the same dict; None = the SOFA-identified module alone.
        tol: gravity fixed-point tolerance (the live default 1e-4 agrees with a tight solve to ~0.01 mm typically, up to
        0.15 mm on 2-module upright poses (FEM test commands, 29 Oct 2026); calibration
        uses ~1e-9 so that finite differences are not swamped by solver noise). gap_mm, joint_mass_kg, tip_mass_kg:
        the measured joint (see the module docstring) and any tip load; mu: capstan friction coefficient (0 = none); a
        calibration file's values take precedence. compliance_json: the module identification; None = the one the
        calibration was made against (else results/mc_std6.json); a different one than the calibration's raises."""
        self.n = int(n_modules)
        g = np.asarray(MOUNTINGS[mounting] if isinstance(mounting, str) else mounting, float)
        self.g_scene = None if not np.any(g) else lab_to_scene(G_MAG * g / np.linalg.norm(g))
        calib = calib_json if isinstance(calib_json, dict) else (json.load(open(_path(calib_json))) if calib_json else None)
        made_with = (calib or {}).get("compliance_json")
        if compliance_json is None:
            compliance_json = made_with or DEFAULT_COMPLIANCE
        elif made_with and os.path.realpath(_path(made_with)) != os.path.realpath(_path(compliance_json)):
            raise ValueError(f"the calibration was made against {made_with}, not {compliance_json}")
        model, theta = ModuleModel.from_result(json.load(open(_path(compliance_json))))
        if cad_json is not None:
            model.cad_json = _path(cad_json)
        self.tol = tol
        hw = None if calib is None else {**{k: calib[k] for k in
                                            ("log_axial", "log_bend_x", "log_bend_z", "tendon_c", "free_len_mm")},
                                         "d_arm_mm": calib.get("d_arm_mm", 0.0)}   # absent in older calibrations
        base_mass = calib.get("mass_kg", mass_kg) if calib else mass_kg  # the measured mass travels with the calibration
        self.mass = base_mass * (np.exp(calib["log_mass"]) if calib else 1.0)
        c = calib or {}
        self.gap = float(c.get("gap_mm", gap_mm))
        self.joint_mass, self.tip_mass = float(c.get("joint_mass_kg", joint_mass_kg)), float(c.get("tip_mass_kg", tip_mass_kg))
        self.mu = float(c.get("mu", mu))
        if self.n not in _CABLE_FILES:
            raise ValueError(f"no real-robot cable file for {self.n} modules (have {sorted(_CABLE_FILES)})")
        cables = os.path.join(CABLES, f"std6_{_CABLE_FILES[self.n]}_section_sym.cables.json")
        az = geometry_from_cables(cables)[1]
        self.eb = model.build(theta, self.n, "paired60", az, HOLES6, hw=hw, gap=self.gap)
        self.stations = np.arange(0.0, MODULE_L * self.n + 1e-9, 5.0)
        self.reset()

    def reset(self):
        """Forget the previous commands (warm start, friction direction): the state of a freshly homed robot."""
        self._state = None
        self._prev, self._sigma = np.zeros(3 * self.n), np.ones(3 * self.n)      # homed: zero, every tendon wound in

    def shape(self, dl, frame="lab", direction=None):
        """dl (3n,) tendon displacements in mm (+ wound in, - paid out), lab tendon order -> TwinShape.
        direction (3n,): +1 wound in / -1 paid out per tendon for the friction term; None = from the previous command."""
        if frame not in ("lab", "scene"):
            raise ValueError("frame must be 'lab' or 'scene'")
        dl = np.asarray(dl, float).reshape(1, 3 * self.n)
        if not np.all(np.isfinite(dl)):
            raise ValueError("tendon displacements must be finite")
        if direction is None:          # a tendon changes direction only after moving DEADBAND_MM from where it last
            step = dl[0] - self._prev  # registered: encoder jitter of a tick or two must not flip the friction branch
            moved = np.abs(step) > DEADBAND_MM
            direction, prev = np.where(moved, np.sign(step), self._sigma), np.where(moved, dl[0], self._prev)
        else:
            prev = dl[0].copy()
        direction = np.asarray(direction, float).reshape(3 * self.n)
        if not np.all(np.isin(direction, (-1.0, 0.0, 1.0))):
            raise ValueError("direction entries must be -1, 0, or +1")
        self._prev, self._sigma = prev, direction.copy()
        t0 = time.perf_counter()
        if self.g_scene is None and not self.mu:
            C, R, lam = self.eb.predict(dl, self.stations)
            converged = True
        else:
            g = np.zeros(3) if self.g_scene is None else self.g_scene
            cold = self._state is None     # a cold solve needs far more iterations than a warm one (4 modules upright)
            iters = (600 if cold else 60) if self.tol >= 1e-6 else (1000 if cold else 400)
            C, R, lam, info = predict_gravity(self.eb, dl, self.stations, g_vec=g, mass=self.mass,
                                              state=self._state, tol=self.tol, iters=iters,
                                              joint_mass=self.joint_mass, tip_mass=self.tip_mass, mu=self.mu,
                                              direction=self._sigma[None])
            converged = info["converged"]
            # an unconverged but finite iterate is a good start for the next call; a diverged one is not
            self._state = None if info["failed_poses"].any() else info["state"]
        ms = 1e3 * (time.perf_counter() - t0)
        C, R = C[0], R[0]
        if frame == "lab":
            C = scene_to_lab(C)
            R = M_SCENE_TO_LAB @ R @ M_SCENE_TO_LAB.T
        return TwinShape(C, R, lam[0], bool(converged), ms)


def direction_from_sequence(dl, start=None):
    """Friction direction of every pose of a command sequence (P, 3n) run in order from home (zero): +1 where a tendon
    was last wound in, -1 where it was last paid out; a tendon that does not move keeps its direction."""
    dl = np.asarray(dl, float)
    prev = np.zeros(dl.shape[1]) if start is None else np.asarray(start, float)
    sig, out = np.ones(dl.shape[1]), np.zeros_like(dl)
    for p, x in enumerate(dl):
        step = x - prev
        sig = np.where(np.abs(step) > 1e-9, np.sign(step), sig)
        out[p], prev = sig, x
    return out


def _path(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)
