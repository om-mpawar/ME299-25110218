"""
Learn the MODULE, compose by statics: a compliance profile along the module, shared by every module, identified
inside the variable-section Euler-Bernoulli prior (tdcr/eb_prior.py) from 1- and 2-section FEM data, then used
unchanged for any number of sections.

The plane-section compliances per cell (integrals of 1/A, 1/Jx, 1/Jz) are multiplied by exp(s c(y) + theta(y)): c(y)
is 1 in the cells a notch cuts and 0 in the plain tube (where plane-section theory is accurate), s a global log-scale and theta a piecewise-linear field over the height y within the module (knots every `knot` mm),
one pair each for axial, bending towards x and bending towards z. theta is penalised (ridge and first differences,
so it stays smooth and near the CAD); s is free except for a tiny ridge that fixes the overall scale when no gravity
data are used. The multipliers are positive, so the complementary energy stays a positive
quadratic: the pull-only problem stays a strictly convex QP and the model composes to N modules exactly as the
physics does (Camarillo 2009 coupling, slack, gravity). theta = 0 is the CAD / plane-section model; a small ridge
pulls theta towards it. Without gravity the overall scale of the three fields cancels (only ratios are identifiable).

The identification itself (Table I, results/mc_std6.json) is run by scripts/module_compliance.py:

    .venv/bin/python scripts/module_compliance.py --train std6_1s_sym_train_g0 std6_2s_sym_train_g0 \
        std6_1s_sym_train std6_2s_sym_train --out results/mc_std6.json
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np
from scipy.optimize import least_squares

from tdcr.backbones import load
from tdcr.eb_prior import EBPrior, predict_gravity, geometry_from_cables, timoshenko_factor, Y_END
from tdcr.params import MASS_STD, STD6
from tdcr.paths import CABLES

TESTS = ["std_1s_sym_test_g0", "std_2s_sym_test_g0", "std_4s_sym_stabletest_g0", "std_1s_sym_test", "std_2s_sym_test"]
TESTS_STD6 = ["std6_1s_sym_test_g0", "std6_2s_sym_test_g0", "std6_4s_sym_test_g0", "std6_1s_sym_test", "std6_2s_sym_test",
              "std6_1s_sym_test_ghang", "std6_2s_sym_test_ghang", "std6_1s_sym_test_gx", "std6_2s_sym_test_gx",
              "std6_1s_sym_test_gy", "std6_2s_sym_test_gy", "std6_1s_sym_rand_g0", "std6_2s_sym_rand_g0"]
MODULE_L = STD6.module_length_mm
E_FEM = STD6.youngs_kpa               # kPa, the measured TPU modulus used in the FEM; the learned field carries the rest
G_MAG = STD6.gravity_mm_s2
G_FEM = (0.0, -G_MAG, 0.0)            # gravity of the FEM sets without the _g0 suffix (upright robot)
MASS_STD6 = STD6.module_mass_kg       # kg per module (MASS_STD: the older std module, tdcr/params.py)
HOLES6 = STD6.holes_deg
CABLES_STD6 = {1: "one", 2: "two", 4: "four"}
# gravity-direction suffix of a sweep tag (sim/std6_sweep.sh) -> direction in the LAB frame (z along the backbone)
G_SUFFIX = {"_ghang": (0, 0, 1), "_gx": (1, 0, 0), "_gmx": (-1, 0, 0), "_gy": (0, 1, 0), "_gmy": (0, -1, 0)}


def lab_to_scene(v):
    """Lab vector (z along the backbone) -> SOFA scene frame (y up): (gx, gy, gz) -> (gx, gz, -gy)
    (sim/two_section_scene.py, gravity_vector)."""
    return (float(v[0]), float(v[2]), -float(v[1]))


def gravity_of(tag):
    """Gravity vector (scene frame, mm/s^2) of a sweep tag; None for the gravity-free _g0 sets."""
    if "_g0" in tag:
        return None
    for suf, lab in G_SUFFIX.items():
        if suf in tag:
            return lab_to_scene(G_MAG * np.asarray(lab, float))
    return G_FEM                                                     # no suffix: upright (lab -z)


def load_set(tag):
    """FEM set with its section count, gravity vector (None for the gravity-free _g0 sets), routing and holes.
    std6_* (the real robot, 10 Oct 2026): tendon azimuths straight from the cable file (sections 3-4 in the channels at
    60/180/300 deg), six channel holes, 33.7 g per module, gravity direction from the tag suffix (_ghang, _gx, _gy, ...).
    Older std_* sets: three shared channels at 0/120/240, no holes, 34.6 g, upright or no gravity."""
    d = load(tag)
    n = d["dl"].shape[1] // 3
    d["n"] = n
    d["g"] = gravity_of(tag)
    if tag.startswith("std6_"):
        cables = os.path.join(CABLES, f"std6_{CABLES_STD6[n]}_section_sym.cables.json")
        d["az"] = geometry_from_cables(cables)[1]
        d["holes"] = HOLES6
        d["routing"] = "paired60"
        d["mass"] = MASS_STD6
    else:
        d["routing"] = "paired60" if "_p60" in tag else "shared"   # SOFA sets before 10 Oct 2026 share 3 channels
        d["mass"] = MASS_STD
    return d


SHEAR_MODES = (None, "theory", "tied", "fit")


class ModuleModel:
    """EB prior with a learned compliance profile (3 fields x K knots) shared by all modules.
    shear (transverse gravity only, see tdcr/eb_prior.predict_gravity):
      None     Euler-Bernoulli, no shear (every result before 19 Oct 2026)
      theory   Timoshenko on the CAD cross-sections: E / (kappa G) times the plane-section integral of 1/A
      tied     the same factor times the IDENTIFIED axial compliance: the notch bridges' shear compliance assumed to
               exceed plane-section theory by the same factor as their axial compliance (no new parameter)
      fit      theory times exp(s_S) in the notched cells: one more parameter (the last entry of theta); it needs
               training data with gravity across the backbone (horizontal mounting)
    torsion: torsional compliance as a factor on plane-section theory (EBPrior.cT_plane), identified from a SOFA twist
    test of one module (scripts/torsion_identify.py); None = rigid in twist (every result before 3 Nov 2026)."""

    def __init__(self, knot=10.0, shear=None, cad_json=None, torsion=None):
        if shear not in SHEAR_MODES:
            raise ValueError(f"shear must be one of {SHEAR_MODES}")
        self.knots = np.arange(0.0, MODULE_L + 1e-9, knot)
        self.K = len(self.knots)
        self.shear = shear
        self.cad_json = cad_json
        self.torsion = torsion
        self.n_theta = 3 + 3 * self.K + (shear == "fit")
        self.base = {}

    @classmethod
    def from_result(cls, res):
        """Model and parameters from a module_compliance.py result (json dict)."""
        m = cls(res["knot"], res.get("shear"), res.get("cad_json"), res.get("torsion"))
        return m, np.array(res["theta"])

    def prior(self, n, routing="shared", az=None, holes=None):
        """az: explicit tendon azimuths (n x 3, SOFA frame, e.g. from tdcr.eb_prior.geometry_from_cables) override
        routing; holes: channel holes cut from the cross-sections (module frame)."""
        key = (n, routing, None if az is None else tuple(np.round(np.ravel(az), 6)), None if holes is None else tuple(holes))
        if key not in self.base:
            eb = EBPrior(n, routing=routing, tendon_az_deg=az, holes=holes, cad_json=self.cad_json)
            per = len(eb.yc) // n
            yl = eb.yc[:per]                                                   # cell centres within a module
            B = np.maximum(0.0, 1.0 - np.abs(yl[:, None] - self.knots[None]) / (self.knots[1] - self.knots[0]))
            cut = (eb.cA_cad[:per] > 1.0001 * eb.cA_cad[:per].min()).astype(float)   # cells where a notch removes material
            self.base[key] = (eb, eb.cA.copy(), eb.cX.copy(), eb.cZ.copy(), B, cut)
        return self.base[key]

    def split(self, theta):
        """-> global log-scales (3,), fields (3, K)."""
        return theta[:3], theta[3:3 + 3 * self.K].reshape(3, self.K)

    def build(self, theta, n, routing="shared", az=None, holes=None, hw=None, gap=0.0):
        """Physics core with the identified compliance theta. hw (hardware calibration, plan Task 1/7): log_axial,
        log_bend_x, log_bend_z are added to the notch log-scales; tendon_c (tendon compliance per mm of tendon, mm^-2)
        and free_len_mm (tendon length outside the robot) give tendon i the stretch tendon_c * (free_len_mm + Y_END +
        section(i) * (L + gap)) on the diagonal of G; d_arm_mm (optional, default 0) moves every tendon's moment arm
        from the SOFA cable radius (EBPrior r_bear, 10.55 mm) to where the real wire bears. hw=None is the
        SOFA-identified module alone. gap: rigid joint length between modules (EBPrior.set_gap)."""
        eb, cA, cX, cZ, B, cut = self.prior(n, routing, az, holes)
        sc, f = self.split(theta)
        if hw is not None:
            sc = sc + np.array([hw["log_axial"], hw["log_bend_x"], hw["log_bend_z"]])
        mA, mX, mZ = (np.tile(np.exp(sc[i] * cut + B @ f[i]), n) for i in range(3))
        eb.cA, eb.cX, eb.cZ = cA * mA, cX * mX, cZ * mZ
        kS = timoshenko_factor()
        if self.cad_json is not None:
            with open(self.cad_json) as f:
                geometry = json.load(f)
            kS = timoshenko_factor(ri=geometry["inner_dia"] / 2, ro=geometry["outer_dia"] / 2)
        if self.shear is None:
            eb.cS = np.zeros_like(eb.cA)
        elif self.shear == "theory":
            eb.cS = kS * eb.cA_cad
        elif self.shear == "tied":
            eb.cS = kS * eb.cA
        else:
            eb.cS = kS * eb.cA_cad * np.tile(np.exp(theta[-1] * cut), n)
        eb.tendon_compliance = np.zeros(3 * n)                 # reset: the prior object is cached between builds
        eb.cT = (self.torsion or 0.0) * eb.cT_plane
        if hw is not None:          # a softer print twists more too: torsion follows the mean bending scale, which
            eb.cT = eb.cT * np.exp(0.5 * (hw["log_bend_x"] + hw["log_bend_z"]))   # keeps Prop. 2's invariance exact
        eb.set_arm(eb.r_bear0 + (hw.get("d_arm_mm", 0.0) if hw is not None else 0.0))
        eb.set_gap(gap)
        if hw is not None and hw["tendon_c"] > 0:
            length = hw["free_len_mm"] + Y_END + (np.arange(3 * n) // 3) * (MODULE_L + gap)
            eb.tendon_compliance = hw["tendon_c"] * length
        eb.assemble()
        return eb

    def predict(self, theta, d):
        eb = self.build(theta, d["n"], d.get("routing", "shared"), d.get("az"), d.get("holes"))
        if d.get("g") is None:
            return eb.predict(d["dl"], d["stations"])
        return predict_gravity(eb, d["dl"], d["stations"], g_vec=d["g"], E=E_FEM, mass=d.get("mass", MASS_STD))[:3]


# ---------------------------------------------------------------- fitting (parallel finite-difference Jacobian)
_G = {}


def _resid(free_theta):
    m, sets, ridge = _G["model"], _G["sets"], _G["ridge"]
    theta = _G["theta_full"].copy()
    theta[_G["free"]] = free_theta
    out = []
    for d, w in sets:
        C, _, _ = m.predict(theta, d)
        out.append((w * (C - d["C"])[:, 1:]).ravel())                       # station 0 is the clamped base
    sc, f = m.split(theta)
    out += [np.sqrt(ridge) * f.ravel(), np.sqrt(_G["smooth"]) * np.diff(f, axis=1).ravel(), 1e-2 * sc]
    if m.shear == "fit":
        out.append(1e-2 * theta[-1:])
    return np.concatenate(out)


def _col(args):
    theta, j, h = args
    t = theta.copy(); t[j] += h
    return _resid(t)


def fit(model, train, ridge=0.02, h=1e-3, iters=40, procs=20, smooth=0.5, theta0=None, free=None):
    """free: boolean mask of the parameters to fit (default all); the others stay at theta0."""
    sets = []
    for d in train:                                                        # equal weight per pose and per station
        sets.append((d, 1.0 / np.sqrt(len(d["C"]) * (d["C"].shape[1] - 1))))
    theta0 = np.zeros(model.n_theta) if theta0 is None else np.asarray(theta0, float)
    free = np.ones(len(theta0), bool) if free is None else np.asarray(free, bool)
    _G.update(model=model, sets=sets, ridge=ridge, smooth=smooth, theta_full=theta0.copy(), free=free)
    for d in train:
        model.prior(d["n"], d.get("routing", "shared"), d.get("az"), d.get("holes"))
    pool = mp.get_context("fork").Pool(min(procs, int(free.sum())))

    def jac(theta):
        r0 = _resid(theta)
        cols = pool.map(_col, [(theta, j, h) for j in range(len(theta))])
        return np.stack([(c - r0) / h for c in cols], 1)

    t0 = time.time()
    r = least_squares(_resid, theta0[free], jac=jac, method="trf", max_nfev=iters, x_scale=1.0, verbose=0)
    pool.close()
    print(f"   fitted {int(free.sum())} of {len(theta0)} parameters in {time.time() - t0:.0f} s, {r.nfev} evaluations, "
          f"cost {r.cost:.4f}")
    theta = theta0.copy()
    theta[free] = r.x
    return theta
