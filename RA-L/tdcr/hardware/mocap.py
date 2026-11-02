"""
Motion-capture ingest: rigid-body poses -> the pose of every module end during every hold of a pose run.

Marker layout: one rigid marker cluster on the base and one on each module's solid cap. In the motion-capture
software the base body defines the lab frame (z along the backbone, x towards tendon 1 of section 1) and each module
body is defined with the same axis directions when the robot is straight; `cap_offset_mm` is the module end's
centre (on the backbone, top face of the cap) in the module body's frame. Export rigid-body poses as csv with columns
time, body, x, y, z, qx, qy, qz, qw (mm, quaternion scalar last; occluded frames may be missing or NaN). Bodies are
named "base", "m1" ... "mN" (load_rigid_bodies takes other export formats through one adapter function each).

    path = make_dataset("data/hw/<session>/servo_log.csv", "data/hw/<session>/mocap.csv", 2, "upright",
                        out="data/hw/<session>/dataset.npz")

Dataset (npz): dl (P, 3n) reached tendon displacements (mm, lab order), dl_cmd (P, 3n) commanded, T_ends (P, n, 4, 4)
module-end poses in the base (lab) frame, NaN where a body was not seen; valid (P,); mounting (3,) gravity direction
(lab); rows (P,), reps (P,).
"""
import csv
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation

MOUNTINGS = {"upright": (0.0, 0.0, -1.0), "hanging": (0.0, 0.0, 1.0),
             "horizontal_x": (1.0, 0.0, 0.0), "horizontal_y": (0.0, 1.0, 0.0)}   # as tdcr.twin.MOUNTINGS


def load_rigid_bodies(csv_path):
    """Generic rigid-body export -> {body: (t (F,), T (F, 4, 4))}, frames sorted by time; NaN rows kept as NaN."""
    rows = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows[r["body"]].append([float(r[k]) if r[k] not in ("", None) else np.nan
                                    for k in ("time", "x", "y", "z", "qx", "qy", "qz", "qw")])
    out = {}
    for name, v in rows.items():
        a = np.array(v)
        a = a[np.argsort(a[:, 0])]
        T = np.full((len(a), 4, 4), np.nan)
        ok = np.all(np.isfinite(a[:, 1:]), axis=1)
        T[ok, :3, :3] = Rotation.from_quat(a[ok, 4:8]).as_matrix()
        T[ok, :3, 3] = a[ok, 1:4]
        T[ok, 3] = (0.0, 0.0, 0.0, 1.0)
        out[name] = (a[:, 0], T)
    return out


def hold_pose(t, T, t0, t1, min_valid=0.8):
    """Mean pose over the hold [t0, t1]: mean position, chordal-mean rotation (the rotation nearest the mean matrix).
    None if fewer than min_valid of the expected frames (from the median frame interval) are valid."""
    t = np.asarray(t, float)
    finite_t = t[np.isfinite(t)]
    if len(finite_t) < 2:
        return None
    dt = np.median(np.diff(np.sort(finite_t)))
    expected = max(1, int(round((t1 - t0) / dt)) + 1)
    sel = (t >= t0) & (t <= t1)
    W = T[sel]
    if len(W) == 0:                                                     # body not seen at all during the hold
        return None
    W = W[np.all(np.isfinite(W.reshape(len(W), -1)), axis=1)]
    if len(W) < min_valid * expected:
        return None
    U, _, Vt = np.linalg.svd(W[:, :3, :3].mean(0))
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, W[:, :3, 3].mean(0)
    return out


def _read_holds(servo_log):
    """-> list of (row, rep, t0, t1, {tendon: (dl_cmd, dl_meas)}) in log order."""
    holds = {}
    with open(servo_log) as f:
        for r in csv.DictReader(f):
            key = (int(r["row"]), int(r.get("rep", 0) or 0))
            h = holds.setdefault(key, [float(r["t_start"]), float(r["t_end"]), {}])
            h[2][r["tendon"]] = (float(r["dl_cmd_mm"]), float(r["dl_meas_mm"]))
    return [(k[0], k[1], v[0], v[1], v[2]) for k, v in holds.items()]


def make_dataset(servo_log, mocap_csv, n_modules, mounting, cap_offset_mm=(0.0, 0.0, 0.0), out="dataset.npz",
                 clock_offset_s=0.0, body_names=None):
    """Pair every hold of a pose run with the motion capture -> path of the npz dataset (see module docstring).
    clock_offset_s is added to the servo-log times to get motion-capture times."""
    n = int(n_modules)
    names = body_names or ["base"] + [f"m{j}" for j in range(1, n + 1)]
    tendons = [f"s{s}t{k}" for s in range(1, n + 1) for k in range(1, 4)]
    bodies = load_rigid_bodies(mocap_csv)
    off = np.eye(4)
    off[:3, 3] = cap_offset_mm
    holds = _read_holds(servo_log)
    P = len(holds)
    T_ends = np.full((P, n, 4, 4), np.nan)
    valid = np.zeros(P, bool)
    dl, dl_cmd = np.zeros((P, 3 * n)), np.zeros((P, 3 * n))
    for p, (row, rep, t0, t1, tend) in enumerate(holds):
        dl_cmd[p] = [tend[k][0] for k in tendons]
        dl[p] = [tend[k][1] for k in tendons]
        poses = [hold_pose(*bodies[b], t0 + clock_offset_s, t1 + clock_offset_s) if b in bodies else None
                 for b in names]
        if any(m is None for m in poses):
            continue
        base_inv = np.linalg.inv(poses[0])
        T_ends[p] = [base_inv @ m @ off for m in poses[1:]]
        valid[p] = True
    g = np.asarray(MOUNTINGS[mounting] if isinstance(mounting, str) else mounting, float)
    np.savez_compressed(out, dl=dl, dl_cmd=dl_cmd, T_ends=T_ends, valid=valid, mounting=g,
                        rows=np.array([h[0] for h in holds]), reps=np.array([h[1] for h in holds]))
    return str(out)
