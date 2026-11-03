"""
The tendon moment arm as a calibration parameter, checked against SOFA (physics audit K1).

SOFA plays the robot: its cable points sit at 10.55 mm. The twin is given a wrong prior arm, 10.15 mm (where a real
wire bearing on the channel wall would sit), and calibrated as on the rig: 15 poses drawn from 15 two-module upright
(training set) and 8 two-module hanging FEM poses, the measured mass, no joint. Tip errors are then scored on the four
FEM test sets the paper reports, for the calibration without the arm (the paper's four parameters) and with it (five).
The control run gives the twin the right arm, 10.55 mm.

    .venv/bin/python scripts/arm_calibration_fem.py          # ~1 min -> results/arm_calibration_fem.json
"""
import argparse
import json
import os
import tempfile
import warnings
warnings.simplefilter("ignore")

import numpy as np

import tdcr.calibrate as cal
import tdcr.compliance as comp
from tdcr.compliance import load_set
from tdcr.eb_prior import R_BEAR, EBPrior
from tdcr.paths import RESULTS, ROOT
from tdcr.twin import M_SCENE_TO_LAB, Twin, scene_to_lab

FOUR = ("log_axial", "log_bend_x", "log_bend_z", "tendon_c")
LAB_G = {"std6_2s_sym_train": (0, 0, -1), "std6_2s_sym_test": (0, 0, -1), "std6_2s_sym_test_ghang": (0, 0, 1)}
TESTS = {"std6_2s_sym_test": "upright", "std6_2s_sym_test_ghang": "hanging", "std6_2s_sym_test_g0": (0, 0, 0),
         "std6_4s_sym_test_g0": (0, 0, 0)}


def dataset(tag, idx, path):
    """FEM poses idx of set `tag` as a tdcr/hardware/mocap.make_dataset file (module ends, lab frame)."""
    d = load_set(tag)
    T = np.tile(np.eye(4), (len(idx), d["n"], 1, 1))
    for q, p in enumerate(idx):
        for j in range(d["n"]):
            k = cal.STATIONS_PER_MODULE * (j + 1)
            T[q, j, :3, 3] = scene_to_lab(d["C"][p, k][None])[0]
            T[q, j, :3, :3] = M_SCENE_TO_LAB @ d["R"][p, k] @ M_SCENE_TO_LAB.T
    np.savez(path, dl=d["dl"][idx], dl_cmd=d["dl"][idx], T_ends=T, valid=np.ones(len(idx), bool),
             mounting=np.array(LAB_G[tag], float))
    return path


def tip_errors(calib):
    """Mean tip error (mm) of the twin on each FEM test set."""
    out = {}
    for tag, mount in TESTS.items():
        d = load_set(tag)
        tw = Twin(d["n"], mount, calib_json=calib, tol=1e-9)
        out[tag] = float(np.mean([np.linalg.norm(tw.shape(x, frame="scene").positions[-1] - d["C"][p, -1])
                                  for p, x in enumerate(d["dl"])]))
    return out


def run(arm0, files):
    """Twin with prior arm arm0 (mm): uncalibrated, then calibrated with four and with five parameters."""
    class Prior(EBPrior):
        def __init__(self, *a, **k):
            super().__init__(*a, **{**k, "r_bear": arm0})
    base, fitted = comp.EBPrior, cal.FITTED
    comp.EBPrior = Prior
    try:
        res = dict(prior_arm_mm=arm0, uncalibrated=tip_errors(None))
        for name, names in (("four", FOUR), ("five", FOUR + ("d_arm_mm",))):
            cal.FITTED = names
            c = cal.fit(files, n_poses=15, seed=0)
            res[name] = dict(params={k: c[k] for k in names}, tip_mm=tip_errors(c),
                             calib_residual_mm=float(np.mean(c["residual_mm"])))
            if name == "five":
                res["arm_mm"] = arm0 + c["d_arm_mm"]
            print(f"prior arm {arm0} mm, {name} parameters: {json.dumps(res[name])}", flush=True)
    finally:
        comp.EBPrior, cal.FITTED = base, fitted
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(RESULTS, "arm_calibration_fem.json"))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        files = [dataset("std6_2s_sym_train", np.arange(0, 175, 12), os.path.join(tmp, "up.npz")),
                 dataset("std6_2s_sym_test_ghang", np.arange(0, 100, 14), os.path.join(tmp, "hang.npz"))]
        res = dict(sofa_arm_mm=R_BEAR, wrong=run(10.15, files), control=run(R_BEAR, files))
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"-> {os.path.relpath(os.path.join(ROOT, a.out), ROOT)}")


if __name__ == "__main__":
    main()
