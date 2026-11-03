"""
Few-shot hardware calibration of the twin (layer L2): five physical numbers (six with friction) from ~15
motion-capture poses.

The module's compliance profile comes from SOFA (results/mc_std6.json); the hardware differs from the FEM in its
overall compliance (print, material), its weight (markers, print), and in the tendons, which stretch and bear at a
smaller radius than SOFA's cable points. Calibrated:

    log_axial, log_bend_x, log_bend_z   added to the identified notch log-scales (axial, bending towards x, z)
    tendon_c                            tendon compliance per mm of tendon (stretch on the diagonal of G)
    d_arm_mm                            tendon moment arm minus the SOFA cable radius (10.55 mm). In SOFA the cable
                                        points sit inside the channel wall; a 0.4-0.6 mm wire bearing on the wall of
                                        the printed robot sits near 10.1-10.2 mm. Under displacement control the bend
                                        scales as dl / arm, which the compliance scales cannot absorb (physics audit
                                        K1: 10.15 mm against the FEM leaves 2-6 mm after refitting the scales)
    mu                                  optional (fit_friction=True): capstan friction in the channels; 0 otherwise.
                                        Under displacement control it acts mainly through the tendon stretch, a
                                        sub-millimetre effect, yet it is recovered (synthetic, 0.3 mm noise) from
                                        up-and-down passes: the direction of every pose comes from the order of the
                                        commanded displacements

The module mass is MEASURED (weigh the modules with their markers) and passed as mass_kg, not fitted: shape data
alone cannot separate stiffness from weight (tendon bending depends on stiffness ratios only, gravity sag on
compliance x weight; on synthetic data with 0.3 mm noise a free mass drifts 60 % with the compliances). log_mass
stays in the calibration file (0, or fitted with fit_mass=True when the data allow it).

Loss: module-end position error (mm) and lever_mm x orientation error (rad) over every module end of the chosen poses,
least squares with bounds (+-1 on the log-scales, 0 <= tendon_c <= 0.05, -1 <= d_arm_mm <= 0.5, 0 <= mu <= 0.6),
finite-difference Jacobian
through the twin
solved to a tight tolerance. Datasets: tdcr/hardware/mocap.make_dataset files (any mix of section counts and mountings);
n_poses valid poses are drawn from their union with `seed`. The joint (gap_mm, joint_mass_kg, tip_mass_kg) is measured
and passed in like the mass; it travels in the calibration file.

    .venv/bin/python -m tdcr.calibrate --datasets data/hw/<s1>/dataset.npz data/hw/<s2>/dataset.npz \
        --n-poses 15 --out results/calib_<session>.json
"""
import argparse
import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from tdcr.compliance import MASS_STD6
from tdcr.twin import Twin, direction_from_sequence

PARAMS = ("log_axial", "log_bend_x", "log_bend_z", "log_mass", "tendon_c", "d_arm_mm", "mu")
FITTED = ("log_axial", "log_bend_x", "log_bend_z", "tendon_c", "d_arm_mm")   # default: mass measured, not fitted
BOUNDS = dict(log_axial=(-1.0, 1.0), log_bend_x=(-1.0, 1.0), log_bend_z=(-1.0, 1.0), log_mass=(-1.0, 1.0),
              tendon_c=(0.0, 0.05), d_arm_mm=(-1.0, 0.5), mu=(0.0, 0.6))
START = dict(log_axial=0.0, log_bend_x=0.0, log_bend_z=0.0, log_mass=0.0, tendon_c=1e-4, d_arm_mm=0.0, mu=0.1)
SCALE = dict(log_axial=1.0, log_bend_x=1.0, log_bend_z=1.0, log_mass=1.0, tendon_c=0.01, d_arm_mm=0.1, mu=0.1)
JOINT = ("gap_mm", "joint_mass_kg", "tip_mass_kg")                 # measured joint, passed in (see fit)
STATIONS_PER_MODULE = 32                                   # twin stations every 5 mm, 160 mm per module (tdcr.params)


def load_datasets(paths):
    """-> list of dicts (n, mounting, dl, dl_cmd, T_ends, valid, sigma) from tdcr/hardware/mocap.make_dataset files.
    sigma: friction direction of every pose from the commanded sequence (poses are stored in run order)."""
    out = []
    for p in paths:
        d = np.load(p)
        cmd = d["dl_cmd"] if "dl_cmd" in d else d["dl"]
        out.append(dict(path=str(p), n=d["T_ends"].shape[1], mounting=tuple(d["mounting"].tolist()), dl=d["dl"],
                        dl_cmd=cmd, T_ends=d["T_ends"], valid=d["valid"].astype(bool),
                        sigma=direction_from_sequence(cmd)))
    return out


def select_poses(data, n_poses, seed):
    """Draw n_poses valid poses from the union of the datasets -> [[dataset index, pose index], ...]."""
    pool = [(i, int(p)) for i, d in enumerate(data) for p in np.flatnonzero(d["valid"])]
    if not pool:
        raise ValueError("no valid poses in the datasets")
    pick = np.random.default_rng(seed).choice(len(pool), min(n_poses, len(pool)), replace=False)
    return [list(pool[k]) for k in pick]


def calib_dict(x, names, free_len_mm, mass_kg, joint=None):
    """Fitted values for `names`, 0 for the other PARAMS, plus the fixed inputs (joint: gap_mm, joint_mass_kg,
    tip_mass_kg, default 0)."""
    c = {k: 0.0 for k in PARAMS}
    c.update({k: float(v) for k, v in zip(names, x)})
    j = {**{k: 0.0 for k in JOINT}, **(joint or {})}
    return {**c, "free_len_mm": float(free_len_mm), "mass_kg": float(mass_kg), **{k: float(v) for k, v in j.items()}}


def _errors(calib, data, sel, compliance_json, lever_mm):
    """-> per pose: (residual vector over its module ends, mean module-end position error in mm)."""
    twins, out = {}, []
    for i, p in sel:
        d = data[i]
        if i not in twins:
            twins[i] = Twin(d["n"], d["mounting"], compliance_json=compliance_json, calib_json=calib, tol=1e-9)
        s = twins[i].shape(d["dl"][p], direction=d["sigma"][p])
        r, pos = [], []
        for j in range(d["n"]):
            k = STATIONS_PER_MODULE * (j + 1)
            dp = s.positions[k] - d["T_ends"][p, j, :3, 3]
            rv = Rotation.from_matrix(s.rotations[k].T @ d["T_ends"][p, j, :3, :3]).as_rotvec()
            r += [dp, lever_mm * rv]
            pos.append(np.linalg.norm(dp))
        out.append((np.concatenate(r), float(np.mean(pos))))
    return out


def fit(datasets, n_poses=15, seed=0, free_len_mm=300.0, mass_kg=MASS_STD6, fit_mass=False,
        compliance_json="results/mc_std6.json", lever_mm=100.0, max_nfev=None, fit_friction=False, gap_mm=0.0,
        joint_mass_kg=0.0, tip_mass_kg=0.0):
    """Calibrate the hardware terms on n_poses poses drawn from `datasets` -> the calibration dict (the PARAMS,
    free_len_mm, the measured mass_kg and joint, poses used, per-pose mean module-end position error after the fit,
    ...)."""
    data = load_datasets(datasets)
    sel = select_poses(data, n_poses, seed)
    names = FITTED + (("log_mass",) if fit_mass else ()) + (("mu",) if fit_friction else ())
    joint = dict(gap_mm=gap_mm, joint_mass_kg=joint_mass_kg, tip_mass_kg=tip_mass_kg)

    def resid(x):
        return np.concatenate([e[0] for e in _errors(calib_dict(x, names, free_len_mm, mass_kg, joint), data, sel,
                                                      compliance_json, lever_mm)])

    r = least_squares(resid, np.array([START[k] for k in names]),
                      bounds=(np.array([BOUNDS[k][0] for k in names]), np.array([BOUNDS[k][1] for k in names])),
                      x_scale=np.array([SCALE[k] for k in names]), diff_step=1e-3, max_nfev=max_nfev)
    calib = calib_dict(r.x, names, free_len_mm, mass_kg, joint)
    calib.update(poses_used=sel, residual_mm=[e[1] for e in _errors(calib, data, sel, compliance_json, lever_mm)],
                 datasets=[d["path"] for d in data], compliance_json=compliance_json, n_poses=len(sel), seed=seed,
                 lever_mm=lever_mm, fitted=list(names), nfev=int(r.nfev), cost=float(r.cost))
    return calib


def save(calib, path):
    with open(path, "w") as f:
        json.dump(calib, f, indent=1)
    return path


def main(argv=None):                                                    # pragma: no cover (CLI)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--n-poses", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--free-len-mm", type=float, default=300.0, help="tendon length outside the robot (measured)")
    ap.add_argument("--mass-kg", type=float, default=MASS_STD6, help="measured module mass incl. markers (kg)")
    ap.add_argument("--gap-mm", type=float, default=2.0, help="rigid joint length beyond the 5 mm web (7 mm coupler)")
    ap.add_argument("--joint-mass-kg", type=float, default=0.0, help="weighed coupler + dowel + plugs (kg)")
    ap.add_argument("--tip-mass-kg", type=float, default=0.0, help="weighed tip cap + markers (kg)")
    ap.add_argument("--fit-friction", action="store_true", help="also fit mu (needs up-and-down passes)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    calib = fit(a.datasets, a.n_poses, a.seed, a.free_len_mm, mass_kg=a.mass_kg, fit_friction=a.fit_friction,
                gap_mm=a.gap_mm, joint_mass_kg=a.joint_mass_kg, tip_mass_kg=a.tip_mass_kg)
    save(calib, a.out)
    print(json.dumps({k: calib[k] for k in (*PARAMS, "nfev")}, indent=1))
    print(f"mean module-end error on the calibration poses: {np.mean(calib['residual_mm']):.2f} mm -> {a.out}")


if __name__ == "__main__":
    main()
