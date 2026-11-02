"""
Few-shot calibration on synthetic data (paper Sec. IV-B/C): what the calibration recovers from 15 poses of two
modules, upright and horizontal, made by the twin itself with known parameters (the datasets of tests/test_calibrate.py,
copied below) and motion-capture noise.

  friction   mu = 0.25 and the other hardware terms of TRUE; up-and-down passes; mass measured; noise 0, 0.1, 0.3 mm.
             Paper: "the fit recovered the bending scale to within 0.003 and mu = 0.25 as 0.244 in one synthetic test"
             (the 0.3 mm row; old repository session of 25 Oct 2026).
  mass       the mass fitted as well (fit_mass=True), noise 0 and 0.3 mm. Paper: "a free mass drifts by 60 % in
             calibration on synthetic data with 0.3 mm noise" (old repository session of 20 Oct 2026, where the
             fitted log-mass went from the true 0.1 to 0.60 while every compliance scale fell by 0.3-0.5).

Both were one-off runs in the old repository; the computations are reproduced here with the packaged calibration.

    .venv/bin/python scripts/calib_synthetic.py          # ~1 min -> results/synthetic_calib.json
"""
import argparse
import json
import math
import os
import tempfile
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.spatial.transform import Rotation

from tdcr.calibrate import PARAMS, fit
from tdcr.compliance import MASS_STD6, load_set
from tdcr.paths import RESULTS, ROOT
from tdcr.twin import Twin

# --- synthetic datasets: the same as tests/test_calibrate.py (TRUE, _dataset, _dataset_sequence), copied so that this
# script runs without the tests on the path
TRUE = dict(log_axial=0.0, log_bend_x=0.2, log_bend_z=0.0, log_mass=0.1, tendon_c=0.01, free_len_mm=300.0)


def _dataset(path, n, mounting, dl, calib=None, noise_mm=0.0, seed=0):
    """Module-end poses (lab frame, relative to the base) predicted by the twin, plus noise -> npz like make_dataset."""
    rng = np.random.default_rng(seed)
    tw = Twin(n, mounting, calib_json=calib, tol=1e-9)
    T = np.zeros((len(dl), n, 4, 4))
    for p, x in enumerate(dl):
        s = tw.shape(x)
        for j in range(n):
            k = 32 * (j + 1)
            T[p, j] = np.eye(4)
            T[p, j, :3, :3] = Rotation.from_rotvec(rng.normal(0, 1e-3, 3)).as_matrix() @ s.rotations[k] if noise_mm \
                else s.rotations[k]
            T[p, j, :3, 3] = s.positions[k] + rng.normal(0, noise_mm, 3)
    g = {"upright": (0, 0, -1), "horizontal_x": (1, 0, 0), "hanging": (0, 0, 1)}[mounting]
    np.savez(path, dl=dl, dl_cmd=dl, T_ends=T, valid=np.ones(len(dl), bool), mounting=np.array(g, float),
             rows=np.arange(len(dl)), reps=np.zeros(len(dl), int))
    return str(path)


def _dataset_sequence(path, n, mounting, calib, seed=0, noise_mm=0.3):
    """Up-and-down passes as on the rig: every test command, then halfway back (the tendons pay out), in that order;
    the twin follows the sequence, so its friction direction is that of the commands."""
    c = load_set("std6_2s_sym_test_g0")["dl"][8 * seed:8 * seed + 8]
    seq = np.stack([c, 0.5 * c], 1).reshape(-1, c.shape[1])
    return _dataset(path, n, mounting, seq, calib, noise_mm=noise_mm, seed=seed)
# ---


def friction(tmp):
    true = {**TRUE, "mu": 0.25}
    out = []
    for noise in (0.0, 0.1, 0.3):
        files = [_dataset_sequence(os.path.join(tmp, f"u{noise}.npz"), 2, "upright", true, seed=1, noise_mm=noise),
                 _dataset_sequence(os.path.join(tmp, f"h{noise}.npz"), 2, "horizontal_x", true, seed=2, noise_mm=noise)]
        c = fit(files, n_poses=15, seed=0, mass_kg=MASS_STD6 * np.exp(TRUE["log_mass"]), fit_friction=True)
        row = dict(noise_mm=noise, mu=c["mu"], log_bend_x=c["log_bend_x"], tendon_c=c["tendon_c"],
                   bend_x_error=abs(c["log_bend_x"] - TRUE["log_bend_x"]), residual_mm=float(np.mean(c["residual_mm"])))
        out.append(row)
        print(f"friction, noise {noise} mm: mu {c['mu']:.3f} (true 0.25), bend_x {c['log_bend_x']:.3f} (true 0.2), "
              f"tendon_c {c['tendon_c']:.4f}, resid {row['residual_mm']:.2f} mm", flush=True)
    return dict(true=true, runs=out)


def mass(tmp):
    cmd = load_set("std6_2s_sym_test_g0")["dl"]
    out = []
    for noise in (0.0, 0.3):
        files = [_dataset(os.path.join(tmp, f"up{noise}.npz"), 2, "upright", cmd[:8], TRUE, noise_mm=noise, seed=1),
                 _dataset(os.path.join(tmp, f"hx{noise}.npz"), 2, "horizontal_x", cmd[8:16], TRUE, noise_mm=noise, seed=2)]
        c = fit(files, n_poses=15, seed=0, fit_mass=True)
        drift = c["log_mass"] - TRUE["log_mass"]
        row = dict(noise_mm=noise, **{k: c[k] for k in PARAMS if k != "mu"}, log_mass_error=drift,
                   mass_relative_error=math.exp(drift) - 1.0, residual_mm=float(np.mean(c["residual_mm"])))
        out.append(row)
        print(f"mass fitted, noise {noise} mm: " + ", ".join(f"{k} {c[k]:.4f}" for k in PARAMS if k != "mu")
              + f" -> mass off by {100 * row['mass_relative_error']:.0f} % (log {drift:+.3f}), "
                f"resid {row['residual_mm']:.3f} mm", flush=True)
    return dict(true=TRUE, runs=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(RESULTS, "synthetic_calib.json"))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        res = dict(friction=friction(tmp), mass=mass(tmp))
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"-> {os.path.relpath(os.path.join(ROOT, a.out), ROOT)}")


if __name__ == "__main__":
    main()
