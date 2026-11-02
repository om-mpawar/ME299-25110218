"""
Cosserat baseline: a uniform Cosserat rod with pull-only tendons (tdcr/cosserat.py, Rucker & Webster statics, slack by
a complementarity solve), driven by the same tendon displacements as the twin, for 1-2 modules.

What it is and is not (to state in the paper):
  * uniform rod: the notched module is replaced by a tube (OD 25 / ID 14) whose axial-to-bending stiffness ratio is
    the one calibrated parameter (log_ratio, 0 = the plain tube). Without gravity Young's modulus cancels from the
    shape (tensions scale with it, the shape does not), so E itself is not identifiable and is fixed at 56 MPa.
  * gravity is NOT modelled (tdcr/cosserat.py would need a shooting solve); on gravity-loaded hardware data this is part
    of its error, reported as such.
  * frames: the rod works in the SOFA scene frame with tendons at scene azimuths 0/120/240; lab tendons (t1, t2, t3) at
    lab azimuths 0/120/240 are scene 0/240/120, i.e. the rod's tendon order is lab (t1, t3, t2). Results are returned
    in the lab frame like the twin's.

    .venv/bin/python -m tdcr.baselines.cosserat_rod --calib data/hw/c1/dataset.npz --test data/hw/t2_up/dataset.npz
    -> results/hw_pred_cosserat_<test name>.npz (read by tdcr/hw_eval.py)
"""
import argparse
import math
import os
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from tdcr import cosserat
from tdcr.calibrate import load_datasets, select_poses
from tdcr.params import STD6
from tdcr.paths import RESULTS
from tdcr.twin import M_SCENE_TO_LAB, scene_to_lab

MODULE_L, TENDON_R, E_KPA, NU = STD6.module_length_mm, STD6.tendon_radius_mm, STD6.youngs_kpa, STD6.poisson
R_OUT, R_IN = STD6.outer_radius_mm, STD6.inner_radius_mm
EA_TUBE = E_KPA * math.pi * (R_OUT ** 2 - R_IN ** 2)
EI_TUBE = E_KPA * math.pi * (R_OUT ** 4 - R_IN ** 4) / 4.0
LAB_TO_ROD = [0, 2, 1]                     # rod tendon k = lab tendon LAB_TO_ROD[k] (an involution)


def _perm(n):
    return np.concatenate([3 * s + np.array(LAB_TO_ROD) for s in range(n)])


def predict(dl, mounting=(0.0, 0.0, -1.0), log_ratio=0.0):
    """dl (P, 3n) lab order, n in (1, 2) -> module-end poses (P, n, 4, 4) lab frame, tensions (P, 3n) lab order (in
    the rod's force units), solve time per pose (ms). mounting is accepted for the interface; gravity is not modelled."""
    dl = np.atleast_2d(np.asarray(dl, float))
    n = dl.shape[1] // 3
    if n not in (1, 2):
        raise ValueError(f"the Cosserat baseline covers 1-2 modules, not {n}")
    perm = _perm(n)
    lengths, gammas = (MODULE_L,) * n, (0.0,) * n
    kw = dict(E=E_KPA, nu=NU, EA=EA_TUBE * math.exp(log_ratio), EI=EI_TUBE)
    T = np.zeros((len(dl), n, 4, 4))
    tau = np.zeros_like(dl)
    ms = np.zeros(len(dl))
    for p, x in enumerate(dl):
        t0 = time.perf_counter()
        t_rod, _ = cosserat.solve_for_displacements(x[perm], lengths, TENDON_R, gammas, **kw)
        frames, _ = cosserat.integrate(t_rod, lengths, TENDON_R, gammas, n_eval=1, **kw)
        ms[p] = 1e3 * (time.perf_counter() - t0)
        tau[p] = t_rod[perm]                                          # perm is its own inverse
        for j in range(n):
            F = frames[j + 1]
            T[p, j] = np.eye(4)
            T[p, j, :3, :3] = M_SCENE_TO_LAB @ F[:3, :3] @ M_SCENE_TO_LAB.T
            T[p, j, :3, 3] = scene_to_lab(F[:3, 3])
    return T, tau, ms


def _loss(log_ratio, data, sel, lever_mm=100.0):
    err = []
    for i, p in sel:
        d = data[i]
        T, _, _ = predict(d["dl"][p][None], d["mounting"], log_ratio)
        for j in range(d["n"]):
            err.append(np.linalg.norm(T[0, j, :3, 3] - d["T_ends"][p, j, :3, 3]) ** 2)
            rv = Rotation.from_matrix(T[0, j, :3, :3].T @ d["T_ends"][p, j, :3, :3]).as_rotvec()
            err.append((lever_mm * np.linalg.norm(rv)) ** 2)
    return float(np.mean(err))


def calibrate_stiffness(datasets, n_poses=15, seed=0):
    """The rod's axial-to-bending stiffness ratio from n_poses poses of 1-2 module datasets -> {"log_ratio", ...}."""
    data = [d for d in load_datasets(datasets) if d["n"] in (1, 2)]
    sel = select_poses(data, n_poses, seed)
    r = minimize_scalar(lambda x: _loss(x, data, sel), bounds=(-4.0, 2.0), method="bounded",
                        options=dict(xatol=1e-3))
    return dict(log_ratio=float(r.x), poses_used=sel, loss=float(r.fun), gravity_modelled=False)


def write_predictions(test_paths, calib, pred_dir=RESULTS):
    """Predict every 1-2 module test dataset -> list of written hw_pred_cosserat_<name>.npz paths (others skipped)."""
    out = []
    for d in load_datasets(test_paths):
        name = os.path.splitext(os.path.basename(d["path"]))[0]
        if d["n"] not in (1, 2):
            print(f"{name}: Cosserat baseline skipped ({d['n']} modules; it covers 1-2)")
            continue
        T, tau, ms = predict(d["dl"], d["mounting"], calib["log_ratio"])
        path = os.path.join(pred_dir, f"hw_pred_cosserat_{name}.npz")
        np.savez(path, T_ends=T, tensions=tau, solve_ms=ms, log_ratio=calib["log_ratio"])
        print(f"{name}: Cosserat {np.mean(ms):.1f} ms per pose -> {path}")
        out.append(path)
    return out


def main(argv=None):                                                    # pragma: no cover (CLI)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--n-poses", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    calib = calibrate_stiffness(a.calib, a.n_poses, a.seed)
    print(f"axial / bending ratio x{math.exp(calib['log_ratio']):.3f} of the plain tube")
    write_predictions(a.test, calib)


if __name__ == "__main__":
    main()
