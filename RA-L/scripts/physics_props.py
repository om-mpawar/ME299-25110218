"""
Physics numbers around Propositions 1-3 and Remark 1 (paper Sec. IV-B, VI) on the identified std6 module
(results/mc_std6.json), with the code's own solver:

  prop1_examples         Prop. 1's example: redistributing the compliance (lower half x3 softer, upper half x3 stiffer)
                         on an all-taut command keeps the tip angle and moves the tip (paper: 4.5 mm; the paper's figure
                         is on the plane-section profile, scripts/study_guide_figs.py; here on the identified module),
                         and the same with slack tendons
  prop1_identifiability  Jacobian of the 1-module (and 2-module) training residual w.r.t. the three notch log-scales
  slack                  how many tendons the FEM commands leave slack (Prop. 1 and Remark 1 assume all three taut)
  prop2                  scale k on every compliance and mass 1/k: invariance in the code, and with Cowper shear on
  prop3                  equal-tension pretension pulls from the identified G (paper: 1:1.5 and 1:1.75:2.25:2.5)
  stretch                tendon stretch at the 2 mm pretension (paper: 0.005 mm), steel 0.4-0.6 mm, with and without
                         the 300 mm free length the calibration assumes
  remark1                friction mu = 0.3: tip hysteresis on the 2-module test commands (every tendon +1 vs -1), and
                         the protocol's version (each command approached, sigma = sign(dl), and left, -sigma), plus
                         pretensioned all-taut poses (paper: 0.3-0.7 mm for a typical two-module pose)
  sag                    unbent 2-module horizontal tip sag (paper: 96 mm) as the paper computed it (lateral tip
                         position of Twin(2, "horizontal_x") at zero command, live tolerance) and at a tight tolerance
  mounting               a 1 deg error in the gravity direction on the 2-module horizontal test commands (paper:
                         1.3 mm), about the base axis and in the vertical plane

Ported from the physics audit (reviews/physics_audit/props.py, W3, 2 Nov 2026), which ran the same computations on
the old repository (d363e90); only the imports, the output path and the paper-definition sag were changed.

    .venv/bin/python scripts/physics_props.py                      # everything (~30 s) -> results/physics_props.json
    .venv/bin/python scripts/physics_props.py --only sag mounting  # a subset (merged into the existing file)
"""
import argparse
import json
import math
import os
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np

from tdcr.compliance import E_FEM, HOLES6, MASS_STD6, ModuleModel, lab_to_scene, load_set
from tdcr.eb_prior import geometry_from_cables, predict_gravity
from tdcr.params import STD6
from tdcr.paths import CABLES, RESULTS, ROOT
from tdcr.twin import Twin

OUT = {}
model, theta = ModuleModel.from_result(json.load(open(os.path.join(RESULTS, "mc_std6.json"))))
MODULE_L = STD6.module_length_mm
STEEL_KPA = 200e6                                                      # steel 200 GPa in kPa (mN / mm^2)


def build(n, th=None, **kw):
    az = geometry_from_cables(os.path.join(CABLES, f"std6_{ {1: 'one', 2: 'two', 4: 'four'}[n]}_section_sym.cables.json"))[1]
    return model.build(theta if th is None else th, n, "paired60", az, HOLES6, **kw)


def solve(eb, dl, g, mass=MASS_STD6, tol=1e-11, **kw):
    dl = np.atleast_2d(dl)
    st = np.arange(0.0, MODULE_L * eb.n + 1e-9, 5.0)
    if g is None and not kw.get("mu"):
        C, R, lam = eb.predict(dl, st)
        return C, R, lam
    C, R, lam, info = predict_gravity(eb, dl, st, g_vec=(0, 0, 0) if g is None else g, E=E_FEM, mass=mass, tol=tol,
                                      iters=4000, **kw)
    assert info["converged"]
    return C, R, lam


def module_end_bends(R, n):
    """Total bend (rad) of each module: relative rotation of its end frames."""
    out = []
    for j in range(n):
        Ra, Rb = R[32 * j], R[32 * (j + 1)]
        rel = Ra.T @ Rb
        out.append(math.acos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
    return np.array(out)


def slack():
    res = {}
    for tag in ["std6_1s_sym_train_g0", "std6_1s_sym_train", "std6_2s_sym_train_g0", "std6_2s_sym_train",
                "std6_1s_sym_test", "std6_2s_sym_test", "std6_4s_sym_test_g0"]:
        d = load_set(tag)
        eb = build(d["n"])
        lam = np.concatenate([solve(eb, d["dl"][i:i + 1], d["g"])[2] for i in range(len(d["dl"]))])
        sl = (lam <= 1e-12).reshape(len(lam), d["n"], 3).sum(2)          # slack tendons per section
        res[tag] = dict(poses=len(lam), all_taut=int((sl == 0).all(1).sum()),
                        slack_per_section_hist={str(k): int((sl == k).sum()) for k in range(4)})
    OUT["slack"] = res


def prop1_identifiability():
    """Jacobian of the 1-module (and 2-module) residuals w.r.t. the three notch log-scales."""
    res = {}
    for name, tags in (("1s_g0", ["std6_1s_sym_train_g0"]), ("1s_upright", ["std6_1s_sym_train"]),
                       ("2s_g0", ["std6_2s_sym_train_g0"])):
        ds = [load_set(t) for t in tags]

        def r(th):
            out = []
            for d in ds:
                eb = build(d["n"], th)
                st = d["stations"]
                if d["g"] is None:
                    C = eb.predict(d["dl"], st)[0]
                else:
                    C = predict_gravity(eb, d["dl"], st, g_vec=d["g"], E=E_FEM, mass=d["mass"], tol=1e-11, iters=4000)[0]
                out.append(C[:, 1:].ravel())
            return np.concatenate(out)
        r0 = r(theta)
        J = []
        for k in range(3):
            t = theta.copy(); t[k] += 1e-4
            J.append((r(t) - r0) / 1e-4)
        J = np.stack(J, 1)
        u, s, vt = np.linalg.svd(J, full_matrices=False)
        res[name] = dict(singular_values=s.tolist(), weakest_direction=vt[-1].tolist(), ratio_strong_weak=float(s[0] / s[-1]))
    OUT["prop1_identifiability"] = res


def prop1_examples():
    """The paper's 4.5 mm example on the identified (not the plane-section) module, and with slack tendons."""
    st = np.arange(0.0, MODULE_L + 1e-9, 0.5)
    ex = {}
    for label, dl in (("all_taut_3_0.5_0.5", [3.0, 0.5, 0.5]), ("payout_3_-1.5_-1.5", [3.0, -1.5, -1.5]),
                      ("two_taut_3_3_-2", [3.0, 3.0, -2.0])):
        eb0 = build(1)
        C0, R0, l0 = eb0.predict(np.array([dl]), st)
        eb1 = build(1)
        f = np.where(eb1.yc < 80.0, 3.0, 1.0 / 3.0)
        eb1.cX, eb1.cZ = eb1.cX * f, eb1.cZ * f
        eb1.assemble()
        C1, R1, l1 = eb1.predict(np.array([dl]), st)
        eb2 = build(1)                                             # axial compliance x2 (ratio change)
        eb2.cA = eb2.cA * 2.0
        eb2.assemble()
        C2, R2, l2 = eb2.predict(np.array([dl]), st)
        ang = lambda R: math.degrees(math.acos(np.clip(R[1, 1], -1, 1)))   # noqa: E731
        ex[label] = dict(slack=int((l0 == 0).sum()), tip_angle_deg=ang(R0[0, -1]),
                         redistribute_angle_change_deg=ang(R1[0, -1]) - ang(R0[0, -1]),
                         redistribute_tip_move_mm=float(np.linalg.norm(C0[0, -1] - C1[0, -1])),
                         axial_x2_angle_change_deg=ang(R2[0, -1]) - ang(R0[0, -1]),
                         axial_x2_tip_move_mm=float(np.linalg.norm(C0[0, -1] - C2[0, -1])))
    OUT["prop1_examples"] = ex


def prop2():
    d = load_set("std6_2s_sym_test_gx")
    dl = d["dl"][:10]
    res = {}
    for shear in (None, "theory"):
        model.shear = shear
        eb = build(2)
        C0 = solve(eb, dl, d["g"])[0]
        k = 2.0
        eb.cA, eb.cX, eb.cZ = eb.cA * k, eb.cX * k, eb.cZ * k             # every compliance but the shear one
        eb.assemble()
        C1 = solve(eb, dl, d["g"], mass=MASS_STD6 / k)[0]
        res[str(shear)] = float(np.linalg.norm(C1 - C0, axis=2).max())
        if shear:
            eb.cS = eb.cS * k
            C2 = solve(eb, dl, d["g"], mass=MASS_STD6 / k)[0]
            res["theory_with_shear_scaled"] = float(np.linalg.norm(C2 - C0, axis=2).max())
    model.shear = None
    OUT["prop2_max_shape_change_mm"] = res


def prop3():
    res = {}
    for n in (2, 4):
        eb = build(n)
        v = eb.G @ np.ones(3 * n)
        pulls = v.reshape(n, 3).mean(1)
        res[f"N{n}_gfree_ratio"] = (pulls / pulls[0]).tolist()
        res[f"N{n}_paper"] = [sum((n - j + 1) / n for j in range(1, s + 1)) for s in range(1, n + 1)]
        res[f"N{n}_within_section_spread"] = float(np.abs(v.reshape(n, 3) / pulls[:, None] - 1).max())
        # with gravity (upright): the pulls that give equal tensions lam0 (2 mm on section 1) in the gravity solve
        lam0 = 2.0 / v[0]
        g = lab_to_scene((0, 0, -9810.0))
        dl = lam0 * v
        for it in range(30):                                            # fixed point on dl: equalise the tensions
            lam = solve(eb, dl, g)[2][0]
            dl = dl + (eb.G @ (lam0 - lam))
        lam = solve(eb, dl, g)[2][0]
        res[f"N{n}_upright_pulls_mm"] = dl.tolist()
        res[f"N{n}_upright_ratio"] = (dl.reshape(n, 3).mean(1) / dl.reshape(n, 3).mean(1)[0]).tolist()
        res[f"N{n}_upright_tension_spread"] = float(np.abs(lam / lam0 - 1).max())
        res[f"N{n}_gfree_pulls_mm"] = (lam0 * v).tolist()
        # equal pulls: are the distal tendons slack?
        lam_eq = eb.tensions(np.full((1, 3 * n), 2.0))[0]
        res[f"N{n}_equal_pull_2mm_tension_N"] = (lam_eq * E_FEM / 1e3).tolist()
        res[f"N{n}_equal_tension_N"] = lam0 * E_FEM / 1e3
    OUT["prop3"] = res


def stretch():
    res = {}
    for n in (1, 2, 4):
        eb = build(n)
        v = eb.G @ np.ones(3 * n)
        lam0 = 2.0 / v[0]                                              # 2 mm pretension on section 1, equal tensions
        T = lam0 * E_FEM                                                # mN
        for dia in (0.4, 0.5, 0.6):
            A = math.pi * dia ** 2 / 4
            for free in (0.0, 300.0):
                L = free + 155.0 + 160.0 * (np.arange(3 * n) // 3)
                s = T * L / (STEEL_KPA * A)
                res[f"N{n}_d{dia}_free{int(free)}"] = dict(T_N=T / 1e3, stretch_mm_max=float(s.max()),
                                                         stretch_over_pull=float((s / (lam0 * v)).max()))
    OUT["tendon_stretch"] = res


def remark1():
    """mu = 0.3 on the 2-module test commands: (a) every tendon wound in (+1) vs paid out (-1); (b) the protocol's
    hysteresis: each command approached (sigma = sign(dl)) and left (-sigma); (c) pretensioned all-taut poses."""
    res = {}
    for tag in ["std6_2s_sym_test_g0", "std6_2s_sym_test"]:
        d = load_set(tag)
        eb = build(2)
        rows = []
        for i in range(len(d["dl"])):
            dl = d["dl"][i:i + 1]
            C0, R0, l0 = solve(eb, dl, d["g"])
            Cp, Rp, lp = solve(eb, dl, d["g"], mu=0.3, direction=np.ones((1, 6)))
            Cm, Rm, lm = solve(eb, dl, d["g"], mu=0.3, direction=-np.ones((1, 6)))
            b0, bp, bm = (module_end_bends(R[0], 2) for R in (R0, Rp, Rm))
            rows.append(dict(slack=int((l0 == 0).sum()), hyst_mm=float(np.linalg.norm(Cp[0, -1] - Cm[0, -1])),
                             bend_change_deg=float(np.degrees(np.abs(bp - bm)).max()),
                             tip_shift_vs_mu0_mm=float(np.linalg.norm(Cp[0, -1] - C0[0, -1]))))
        h = np.array([r["hyst_mm"] for r in rows])
        sl = np.array([r["slack"] for r in rows])
        bc = np.array([r["bend_change_deg"] for r in rows])
        res[tag] = dict(poses=len(rows), hyst_median=float(np.median(h)), hyst_p10_p90=np.percentile(h, [10, 90]).tolist(),
                        hyst_max=float(h.max()), all_taut_poses=int((sl == 0).sum()),
                        bend_change_all_taut_max_deg=float(bc[sl == 0].max()) if (sl == 0).any() else None,
                        bend_change_slack_median_deg=float(np.median(bc[sl > 0])) if (sl > 0).any() else None,
                        bend_change_slack_max_deg=float(bc[sl > 0].max()) if (sl > 0).any() else None)
    OUT["remark1_mu0.3"] = res
    res = {}
    for tag in ["std6_1s_sym_test", "std6_2s_sym_test_g0", "std6_2s_sym_test"]:
        d = load_set(tag)
        eb = build(d["n"])
        h = []
        for i in range(len(d["dl"])):
            dl = d["dl"][i:i + 1]
            s = np.where(dl >= 0, 1.0, -1.0)
            Ca = solve(eb, dl, d["g"], mu=0.3, direction=s)[0]
            Cl = solve(eb, dl, d["g"], mu=0.3, direction=-s)[0]
            h.append(float(np.linalg.norm(Ca[0, -1] - Cl[0, -1])))
        h = np.array(h)
        res[tag] = dict(poses=len(h), median=float(np.median(h)), p10_p90=np.percentile(h, [10, 90]).tolist(),
                        max=float(h.max()), frac_above_0p7=float((h > 0.7).mean()))
    # all-taut poses: pretension pattern (equal tensions) 2 mm on section 1 plus a bend of +-b on top
    eb = build(2)
    v = eb.G @ np.ones(6)
    base = 2.0 / v[0] * v
    rows = []
    up = lab_to_scene((0, 0, -9810.0))
    for b in (0.5, 1.0, 1.5):
        bend = np.array([b, -b / 2, -b / 2, b, -b / 2, -b / 2])
        dl = (base + bend)[None]
        lam0 = solve(eb, dl, up)[2][0]
        s = np.where(bend >= 0, 1.0, -1.0)[None]
        Ca = solve(eb, dl, up, mu=0.3, direction=s)[0]
        Cl = solve(eb, dl, up, mu=0.3, direction=-s)[0]
        rows.append(dict(bend_pull_mm=b, slack=int((lam0 == 0).sum()), hyst_mm=float(np.linalg.norm(Ca[0, -1] - Cl[0, -1]))))
    res["pretensioned_all_taut_upright"] = rows
    OUT["remark1_protocol_mu0.3"] = res


def sag():
    res = {}
    # as the paper computed it (old repo session of 26 Oct 2026): the twin at its live tolerance, lateral tip position
    for name in ("horizontal_x", "horizontal_y"):
        s = Twin(2, name).shape(np.zeros(6))
        res[f"twin_live_{name}_tip_lateral_mm"] = float(np.hypot(*s.positions[-1, :2]))
        s = Twin(2, name, tol=1e-11).shape(np.zeros(6))
        res[f"twin_tight_{name}_tip_lateral_mm"] = float(np.hypot(*s.positions[-1, :2]))
    # W3: distance of the tip from the unloaded straight tip, tight solve; 1 deg mounting error in the vertical plane
    eb = build(2)
    for name, lab in (("gx", (9810.0, 0, 0)), ("gy", (0, 9810.0, 0))):
        g = lab_to_scene(lab)
        C = solve(eb, np.zeros((1, 6)), g)[0]
        res[f"tip_sag_{name}_mm"] = float(np.linalg.norm(C[0, -1] - np.array([0, 2 * MODULE_L, 0])))
        d = load_set(f"std6_2s_sym_test_{name}")
        a = math.radians(1.0)
        lab2 = (9810.0 * math.cos(a), 0, 9810.0 * math.sin(a)) if name == "gx" else (0, 9810.0 * math.cos(a),
                                                                                    9810.0 * math.sin(a))
        e = []
        for i in range(len(d["dl"])):
            C1 = solve(eb, d["dl"][i:i + 1], g)[0]
            C2 = solve(eb, d["dl"][i:i + 1], lab_to_scene(lab2))[0]
            e.append(float(np.linalg.norm(C1[0, -1] - C2[0, -1])))
        res[f"tip_change_per_deg_{name}_median_mm"] = float(np.median(e))
        res[f"tip_change_per_deg_{name}_p90_mm"] = float(np.percentile(e, 90))
    OUT["sag"] = res


def mounting():
    """1 deg error in the direction of gravity about the base axis (lab z) for horizontal 2-module mountings."""
    eb = build(2)
    a = math.radians(1.0)
    res = {}
    for name, g0, g1 in (("gx", (9810.0, 0, 0), (9810.0 * math.cos(a), 9810.0 * math.sin(a), 0)),
                         ("gy", (0, 9810.0, 0), (-9810.0 * math.sin(a), 9810.0 * math.cos(a), 0))):
        C0 = solve(eb, np.zeros((1, 6)), lab_to_scene(g0))[0]
        C1 = solve(eb, np.zeros((1, 6)), lab_to_scene(g1))[0]
        d = load_set(f"std6_2s_sym_test_{name}")
        e = [float(np.linalg.norm(solve(eb, d["dl"][i:i + 1], lab_to_scene(g0))[0][0, -1]
                                  - solve(eb, d["dl"][i:i + 1], lab_to_scene(g1))[0][0, -1])) for i in range(len(d["dl"]))]
        res[name] = dict(unbent_mm=float(np.linalg.norm(C1[0, -1] - C0[0, -1])), test_median_mm=float(np.median(e)),
                         test_p90_mm=float(np.percentile(e, 90)))
    OUT["mounting_error_about_axis_1deg"] = res


STEPS = dict(prop1_examples=prop1_examples, prop2=prop2, prop3=prop3, stretch=stretch, sag=sag, mounting=mounting,
             remark1=remark1, prop1_identifiability=prop1_identifiability, slack=slack)

KEYS = dict(prop1_examples=["prop1_examples"], prop2=["prop2_max_shape_change_mm"], prop3=["prop3"],
            stretch=["tendon_stretch"], sag=["sag"], mounting=["mounting_error_about_axis_1deg"],
            remark1=["remark1_mu0.3", "remark1_protocol_mu0.3"], prop1_identifiability=["prop1_identifiability"],
            slack=["slack"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(STEPS), help="run these steps, merge into the existing file")
    ap.add_argument("--out", default=os.path.join(RESULTS, "physics_props.json"))
    a = ap.parse_args()
    out = os.path.join(ROOT, a.out)
    if a.only and os.path.exists(out):
        OUT.update(json.load(open(out)))
    OUT["source"] = "scripts/physics_props.py (ported from reviews/physics_audit/props.py, W3)"
    for name in a.only or STEPS:
        t0 = time.time()
        STEPS[name]()
        for k in KEYS[name]:
            print(f"{name} ({time.time() - t0:.0f} s): {k} {json.dumps(OUT[k])[:600]}", flush=True)
        json.dump(OUT, open(out, "w"), indent=1)
    print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
