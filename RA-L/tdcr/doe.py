"""
Command designs in TENDON-DISPLACEMENT space, generated the way the real robot is driven.

Mapping: the lab's hardware tendon mapping (tdcr/tendon_mapping.py, ported from the lab's hardware-control code):
for target bends (theta_s, alpha_s) per section,  dl = T(alpha) lv,  lv_s = R_D theta_s,  with the routing offsets
gamma (sections 3-4 of the real robot in the channels at +60 deg: gamma = 0, 0, 60, 0). This is pure pay-out: the
inside tendons are wound in and the outside paid out by the geometric amounts, no common offset. The robot is
tensioned by hand until it stands straight with no slack, so dl = 0 is the taut rest state (--pretension adds a
common pull on every tendon, for sensitivity tests only).

Frames and numbering: alpha and the tendon azimuths are in the LAB frame (z along the backbone, angle from +x
towards +y); tendon k of section s sits at lab azimuth offset_s + 120 k, the order of the cable files written with
tools/make_std_cables.py --lab-numbering. theta / alpha are stored only as metadata (the sweep needs the columns);
every model reads dl.

Screening (physics core with the identified module, tdcr/eb_prior.py + analysis/module_compliance.py, geometry read
from the cable file): a design is dropped when a section bends beyond --max-bend, when the axial load of any module
exceeds --max-load (tension / E, mm^2; the stable 4-section FEM set reached 1.83), or, with --screen-gravity, when
the gravity fixed point does not converge (a sign of buckling).

--random S: instead of the mapping alone, add independent noise U[-S, S] mm to every tendon and a common offset
U[-1, 3] mm per section (commands the mapping would not produce, for the generalisation test), then screen.

    TDCR_SECTION_L=160 TDCR_TENDON_R=9.5 .venv/bin/python tools/make_cmd_doe.py --sections 4 --n 100 --seed 61 \
        --cables data/cables/std6_four_section_sym.cables.json --max-theta 45 --screen-gravity \
        --out results/configs_std6_4s_test.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import csv
import json
import warnings
warnings.simplefilter("ignore")

import numpy as np

from tdcr.tendon_mapping import TendonMapping
from tdcr.eb_prior import geometry_from_cables, predict_gravity

ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
HOLES6 = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
E_FEM = 56000.0


def lab_offsets(cables):
    """Per-section lab azimuth of tendon 0, from a --lab-numbering cable file."""
    d = json.load(open(cables))
    if not str(d.get("numbering", "")).startswith("lab"):
        raise SystemExit(f"{cables}: not written with --lab-numbering")
    n = max(t["section"] for t in d["tendons"])
    off = [None] * n
    for t in d["tendons"]:
        if off[t["section"] - 1] is None:
            off[t["section"] - 1] = t["azimuth_lab_deg"]
    return np.array(off)


def mapped(theta_deg, alpha_deg, offsets, r_d):
    """(P, n) target bends (deg, lab frame) -> (P, 3n) tendon displacements by the lab mapping."""
    gam = np.diff(np.concatenate([[0.0], offsets]))            # TendonMapping cumulates gamma
    tm = TendonMapping(n_sections=len(offsets), gamma=tuple(gam), R_D=r_d)
    return np.array([tm.compute_lengths(a, tm.theta_to_lv(t)) for t, a in zip(theta_deg, alpha_deg)])


def screen(dl, cables, n, max_bend, max_load, gravity):
    """-> keep mask, per-design max bend (deg), max module load (mm^2), slack tendons, gravity iterations."""
    from analysis.module_compliance import ModuleModel
    res = json.load(open(_os.path.join(ROOT, "results", "mc_grav.json")))
    model = ModuleModel(res["knot"])
    _, az = geometry_from_cables(cables)
    eb = model.build(np.array(res["theta"]), n, az=az, holes=HOLES6)
    lam = eb.tensions(dl)
    per = len(eb.yc) // n
    T = eb.through.astype(float)
    w, _ = eb.strains(lam)
    bend = np.degrees(np.array([np.linalg.norm(w[:, s * per:(s + 1) * per].sum(1), axis=-1) for s in range(n)])).T
    load = np.array([(lam * T[:, s * per + per // 2][None]).sum(1) for s in range(n)]).T     # tension sum through mid-module
    slack = (lam < 1e-12).sum(1)
    keep = (bend.max(1) <= max_bend) & (load.max(1) <= max_load)
    its = np.zeros(len(dl), int)
    if gravity:
        st = np.arange(0.0, 160.0 * n + 1e-9, 5.0)
        for i in np.where(keep)[0]:
            info = predict_gravity(eb, dl[i:i + 1], st, g_vec=(0.0, -9810.0, 0.0), E=E_FEM, iters=80)[3]
            its[i] = info["iterations"] if info["converged"] else 80
        keep &= its < 80
    return keep, bend.max(1), load.max(1), slack, its


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", type=int, required=True)
    ap.add_argument("--cables", required=True, help="cable file of the simulated robot (routing, lab numbering)")
    ap.add_argument("--n", type=int, required=True, help="designs to keep")
    ap.add_argument("--max-theta", type=float, default=45.0, help="largest target bend per section (deg)")
    ap.add_argument("--r-d", type=float, default=9.5, help="R_D of the mapping (channel radius, mm)")
    ap.add_argument("--pretension", type=float, default=0.0, help="common pull on every tendon (mm); robot: 0")
    ap.add_argument("--random", type=float, default=0.0, help="per-tendon noise half-width (mm); 0 = mapping only")
    ap.add_argument("--max-bend", type=float, default=100.0, help="screen: predicted bend per section (deg)")
    ap.add_argument("--max-load", type=float, default=1.83, help="screen: axial load per module, tension / E (mm^2)")
    ap.add_argument("--screen-gravity", action="store_true", help="screen: upright gravity fixed point must converge")
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = a.sections
    off = lab_offsets(a.cables)
    assert len(off) == n, f"{a.cables} has {len(off)} sections"
    rng = np.random.default_rng(a.seed)
    rows, tried = [], 0
    while len(rows) < a.n and tried < 50 * a.n:
        m = 4 * (a.n - len(rows)) + 16
        th = a.max_theta * np.sqrt(rng.uniform(0, 1, (m, n)))          # uniform over the disk of (theta, alpha)
        al = rng.uniform(0, 360.0, (m, n))
        dl = mapped(th, al, off, a.r_d) + a.pretension
        if a.random > 0:
            dl += rng.uniform(-a.random, a.random, dl.shape)
            dl += np.repeat(rng.uniform(-1.0, 3.0, (m, n)), 3, axis=1)
        keep, bend, load, slack, its = screen(dl, a.cables, n, a.max_bend, a.max_load, a.screen_gravity)
        tried += m
        for i in np.where(keep)[0][: a.n - len(rows)]:
            rows.append((th[i], al[i], dl[i], bend[i], load[i], slack[i]))
    if len(rows) < a.n:
        raise SystemExit(f"only {len(rows)} of {a.n} designs passed the screen after {tried} tries")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"theta{j+1}" for j in range(n)] + [f"alpha{j+1}" for j in range(n)] + [f"dl{i+1}" for i in range(3 * n)]
                   + ["label", "pred_bend_deg", "pred_load", "pred_slack"])
        for th, al, dl, b, l, sl in rows:
            w.writerow([f"{v:.9f}" for v in np.radians(th)] + [f"{v:.9f}" for v in np.radians(al)] + [f"{v:.9f}" for v in dl]
                       + ["", f"{b:.2f}", f"{l:.4f}", int(sl)])
    meta = dict(cables=a.cables, lab_offsets=off.tolist(), mapping="tdcr/tendon_mapping.py (lab), pure pay-out",
                r_d=a.r_d, pretension=a.pretension, random=a.random, max_theta=a.max_theta, seed=a.seed,
                screen=dict(max_bend=a.max_bend, max_load=a.max_load, gravity=a.screen_gravity),
                tried=tried, kept=len(rows), frame="theta/alpha: lab frame, metadata only; dl in lab tendon order")
    json.dump(meta, open(a.out.replace(".csv", ".meta.json"), "w"), indent=1)
    b = np.array([r[3] for r in rows]); l = np.array([r[4] for r in rows]); s = np.array([r[5] for r in rows])
    print(f"{a.out}: {len(rows)} designs kept of {tried} tried ({100 * len(rows) / tried:.0f} %); predicted bend "
          f"max {b.max():.0f} deg, module load median {np.median(l):.2f} max {l.max():.2f} mm^2, slack tendons/pose "
          f"{s.mean():.1f}; dl {min(r[2].min() for r in rows):.1f} .. {max(r[2].max() for r in rows):.1f} mm")


if __name__ == "__main__":
    main()
