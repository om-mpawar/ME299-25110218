"""
Hardware evaluation: every model on every test dataset, against motion capture.

Models (MODELS):
  cc               constant curvature from dl (exact 3-tendon Fourier inversion section by section, real routing:
                   sections 3-4 at +60 deg, lab tendon order), no slack, no gravity
  physics          the twin with the SOFA-identified module only (no hardware calibration)
  calibrated       the twin after tdcr/calibrate.fit on n_poses poses of the calibration datasets
  gp               pure learning: GP from dl to module-end positions and rotation vectors, trained on the same
                   calibration poses; only for test sets with the same number of modules (otherwise not applicable)
  cosserat, sofa_mor, sofa_calibrated
                   predictions read from <pred_dir>/hw_pred_<model>_<dataset>.npz (key T_ends), made by
                   analysis/cosserat_baseline.py, sim/mor/run_reduced.py and sim/calibrate_sofa.py; skipped with a note
                   when the file is missing

Metrics (evaluate): tip = last module end; tip_mm, tip_pct (of the robot length), tip_deg (geodesic angle), ends_mm
(mean over all module ends), with 95th percentiles; invalid poses (NaN in the ground truth) are left out.

    .venv/bin/python analysis/hw_eval.py --calib data/hw/c1/dataset.npz data/hw/c2/dataset.npz \
        --test data/hw/t4_hang/dataset.npz data/hw/t2_up/dataset.npz --n-poses 15 --mass-kg <weighed> \
        --out results/hw_eval.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
import math
import os
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from analysis.model_study import gp_fit_predict
from analysis.module_compliance import MASS_STD6
from tdcr.calibrate import JOINT, STATIONS_PER_MODULE, fit, load_datasets
from tdcr.paths import RESULTS
from tdcr.twin import Twin

MODELS = ("cc", "physics", "calibrated", "gp", "cosserat", "sofa_mor", "sofa_calibrated")
FILE_MODELS = ("cosserat", "sofa_mor", "sofa_calibrated")
MODULE_L, TENDON_R = 160.0, 9.5
LAB_OFFSETS = (0.0, 0.0, 60.0, 60.0)                      # lab azimuth of tendon 1 per section (real routing)


def evaluate(pred, true, length_mm, valid=None):
    """pred, true: (P, n, 4, 4) module-end poses -> mean and 95th-percentile errors over the valid poses."""
    if pred.shape != true.shape or true.ndim != 4 or true.shape[-2:] != (4, 4):
        raise ValueError("pred and true must have matching (P, n, 4, 4) shapes")
    if not np.isfinite(length_mm) or length_mm <= 0:
        raise ValueError("length_mm must be positive and finite")
    truth_ok = np.all(np.isfinite(true), axis=(1, 2, 3))
    if valid is not None:
        valid = np.asarray(valid, bool)
        if valid.shape != (len(true),):
            raise ValueError("valid must have one entry per pose")
        truth_ok &= valid
    ok = truth_ok & np.all(np.isfinite(pred), axis=(1, 2, 3))
    counts = dict(n_poses=int(ok.sum()), n_valid_truth=int(truth_ok.sum()),
                  n_failed_predictions=int((truth_ok & ~ok).sum()), n_total=int(len(true)))
    if not ok.any():
        return dict(counts, **{k: None for k in ("tip_mm", "tip_pct", "tip_deg", "ends_mm", "tip_mm_95",
                                                "tip_deg_95", "ends_mm_95")})
    p, t = pred[ok], true[ok]
    pos = np.linalg.norm(p[:, :, :3, 3] - t[:, :, :3, 3], axis=-1)              # (P, n)
    rel = np.einsum("pji,pjk->pik", p[:, -1, :3, :3], t[:, -1, :3, :3])
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))
    tip = pos[:, -1]
    return dict(counts, tip_mm=float(tip.mean()), tip_pct=float(100 * tip.mean() / length_mm),
                tip_deg=float(ang.mean()), ends_mm=float(pos.mean()), tip_mm_95=float(np.percentile(tip, 95)),
                tip_deg_95=float(np.percentile(ang, 95)), ends_mm_95=float(np.percentile(pos.mean(1), 95)))


def _section(kappa, alpha, L):
    """Constant-curvature section in the lab frame (z along the backbone), bending towards lab azimuth alpha."""
    T = np.eye(4)
    if abs(kappa) < 1e-12:
        T[2, 3] = L
        return T
    th, rho = kappa * L, 1.0 / kappa
    u, v = rho * (1 - math.cos(th)), rho * math.sin(th)
    T[:3, 3] = (u * math.cos(alpha), u * math.sin(alpha), v)
    T[:3, :3] = Rotation.from_rotvec(th * np.array([-math.sin(alpha), math.cos(alpha), 0.0])).as_matrix()
    return T


def cc_ends(dl, n, r=TENDON_R, L=MODULE_L):
    """Constant curvature: dl (3n,) mm, lab order -> module-end poses (n, 4, 4) in the lab frame. Tendon k of section
    s sits at lab azimuth LAB_OFFSETS[s] + 120 k; a tendon passing a proximal section j shortens by r kappa_j L
    cos(psi - alpha_j); sections are recovered proximal first (exact Fourier inversion, slack and load ignored)."""
    dl = np.asarray(dl, float)
    psi = [np.radians(LAB_OFFSETS[s] + 120.0 * np.arange(3)) for s in range(n)]
    kap, alp = np.zeros(n), np.zeros(n)
    for s in range(n):
        res = dl[3 * s:3 * s + 3].copy()
        for j in range(s):
            res -= r * kap[j] * L * np.cos(psi[s] - alp[j])
        A, B = (2 / 3) * np.sum(res * np.cos(psi[s])), (2 / 3) * np.sum(res * np.sin(psi[s]))
        kap[s], alp[s] = math.hypot(A, B) / (r * L), math.atan2(B, A)
    out, T = [], np.eye(4)
    for s in range(n):
        T = T @ _section(kap[s], alp[s], L)
        out.append(T.copy())
    return np.array(out)


def twin_ends(d, calib=None, mass_kg=MASS_STD6, **joint):
    """Module-end poses of every pose of dataset d predicted by the twin (lab frame) -> (P, n, 4, 4). joint: gap_mm,
    joint_mass_kg, tip_mass_kg of the robot (a calibration's own values take precedence); friction direction from
    d["sigma"] when present."""
    tw = Twin(d["n"], d["mounting"], calib_json=calib, mass_kg=mass_kg, **joint)
    out = np.zeros((len(d["dl"]), d["n"], 4, 4))
    for p, x in enumerate(d["dl"]):
        s = tw.shape(x, direction=d["sigma"][p] if "sigma" in d else None)
        if not s.converged:
            out[p] = np.nan
            continue
        for j in range(d["n"]):
            k = STATIONS_PER_MODULE * (j + 1)
            out[p, j] = np.eye(4)
            out[p, j, :3, :3], out[p, j, :3, 3] = s.rotations[k], s.positions[k]
    return out


def gp_ends(train, d):
    """Pure GP baseline: (dl -> module-end positions and rotation vectors) learnt from the calibration poses with
    d's module count -> (P, n, 4, 4), or None if there are none."""
    X = [t["dl"][p] for t, p in train if t["n"] == d["n"]]
    Y = [np.concatenate([np.r_[t["T_ends"][p, j, :3, 3], Rotation.from_matrix(t["T_ends"][p, j, :3, :3]).as_rotvec()]
                         for j in range(t["n"])]) for t, p in train if t["n"] == d["n"]]
    if len(X) < 3:
        return None
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():              # a length scale at its bound = an input the GP finds irrelevant
        warnings.simplefilter("ignore", ConvergenceWarning)
        Yp = gp_fit_predict(np.array(X), np.array(Y), d["dl"], restarts=1).reshape(len(d["dl"]), d["n"], 6)
    out = np.tile(np.eye(4), (len(d["dl"]), d["n"], 1, 1))
    out[..., :3, 3] = Yp[..., :3]
    out[..., :3, :3] = Rotation.from_rotvec(Yp[..., 3:].reshape(-1, 3)).as_matrix().reshape(len(d["dl"]), d["n"], 3, 3)
    return out


def run_eval(calib_paths, test_paths, n_poses=15, seed=0, mass_kg=MASS_STD6, pred_dir=RESULTS, max_nfev=None,
             **fit_kw):
    """Calibrate once, then every model on every test set -> (results {test name: {model: metrics}}, table text).
    fit_kw: the measured joint and fit_friction, passed to tdcr.calibrate.fit. Prints a note for every model that is
    skipped."""
    calib_paths, test_paths = list(calib_paths), list(test_paths)
    if set(map(os.path.realpath, calib_paths)) & set(map(os.path.realpath, test_paths)):
        raise ValueError("calibration and test datasets must be separate files")
    # Session files commonly share the basename dataset.npz. Keep distinct IDs,
    # and use those same IDs for externally supplied baseline prediction files.
    stems = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]
    names = [s if stems.count(s) == 1 else os.path.basename(os.path.dirname(os.path.abspath(p))) + "__" + s
             for p, s in zip(test_paths, stems)]
    if len(set(names)) != len(names):
        raise ValueError("test dataset IDs collide; use distinct session folders or filenames")
    calib = fit(calib_paths, n_poses=n_poses, seed=seed, mass_kg=mass_kg, max_nfev=max_nfev, **fit_kw)
    cdata = load_datasets(calib_paths)
    train = [(cdata[i], p) for i, p in calib["poses_used"]]
    results = {}
    for name, d in zip(names, load_datasets(test_paths)):
        length, preds = MODULE_L * d["n"] + calib["gap_mm"] * (d["n"] - 1), {}
        preds["cc"] = np.array([cc_ends(x, d["n"]) for x in d["dl"]])
        preds["physics"] = twin_ends(d, None, mass_kg, **{k: calib[k] for k in JOINT})   # same measured joint
        preds["calibrated"] = twin_ends(d, calib)
        g = gp_ends(train, d)
        if g is None:
            print(f"{name}: gp not applicable (no calibration poses with {d['n']} modules)")
        else:
            preds["gp"] = g
        for m in FILE_MODELS:
            f = os.path.join(pred_dir, f"hw_pred_{m}_{name}.npz")
            if os.path.exists(f):
                preds[m] = np.load(f)["T_ends"]
            else:
                print(f"{name}: {m} skipped (no {f})")
        results[name] = {m: evaluate(preds[m], d["T_ends"], length, d["valid"]) for m in MODELS if m in preds}
    cols = [m for m in MODELS if any(m in r for r in results.values())]
    lines = [f"{'test set (tip mm / deg)':28s}" + "".join(f"{m:>18s}" for m in cols)]
    for name, r in results.items():
        lines.append(f"{name:28s}" + "".join(f"{r[m]['tip_mm']:9.2f} /{r[m]['tip_deg']:6.2f}"
                                             if m in r and r[m]['tip_mm'] is not None else f"{'-':>18s}"
                                             for m in cols))
        for m, metrics in r.items():
            if metrics["n_failed_predictions"]:
                lines.append(f"  {m}: {metrics['n_failed_predictions']}/{metrics['n_valid_truth']} valid poses failed; "
                             "errors above describe successful predictions only")
    return results, "\n".join(lines)


def main(argv=None):                                                    # pragma: no cover (CLI)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--n-poses", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mass-kg", type=float, default=MASS_STD6)
    ap.add_argument("--gap-mm", type=float, default=2.0)
    ap.add_argument("--joint-mass-kg", type=float, default=0.0)
    ap.add_argument("--tip-mass-kg", type=float, default=0.0)
    ap.add_argument("--fit-friction", action="store_true")
    ap.add_argument("--out", default=os.path.join(RESULTS, "hw_eval.json"))
    a = ap.parse_args(argv)
    res, table = run_eval(a.calib, a.test, a.n_poses, a.seed, a.mass_kg, gap_mm=a.gap_mm,
                          joint_mass_kg=a.joint_mass_kg, tip_mass_kg=a.tip_mass_kg, fit_friction=a.fit_friction)
    print(table)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
