"""Motion-capture ingest (plan Task 6): rigid-body poses -> module-end poses per hold, on synthetic exports."""
import csv

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hardware.mocap import hold_pose, load_rigid_bodies, make_dataset

RATE = 100.0                                                      # mocap frames per second


def _T(rotvec=(0, 0, 0), pos=(0, 0, 0)):
    T = np.eye(4); T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix(); T[:3, 3] = pos
    return T


def _write_mocap(path, bodies, t_end, gaps=()):
    """bodies: {name: T (4, 4) or callable t -> T}; gaps: [(name, t0, t1)] where the body is not exported."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "body", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for t in np.arange(0.0, t_end, 1.0 / RATE):
            for name, T in bodies.items():
                if any(name == g[0] and g[1] <= t <= g[2] for g in gaps):
                    continue
                T = T(t) if callable(T) else T
                q = Rotation.from_matrix(T[:3, :3]).as_quat()           # x, y, z, w
                w.writerow([f"{t:.4f}", name, *T[:3, 3], *q])


def _write_log(path, holds, n_modules):
    """holds: [(t0, t1)] -> a servo log with one row per hold, dl = row index * 0.1 mm on every tendon."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "rep", "t_start", "t_end", "tendon", "goal_ticks", "present_ticks", "current_ma",
                    "dl_cmd_mm", "dl_meas_mm"])
        for i, (t0, t1) in enumerate(holds):
            for s in range(1, n_modules + 1):
                for k in range(1, 4):
                    w.writerow([i, 0, t0, t1, f"s{s}t{k}", 0, 0, 10.0, 0.1 * i, 0.1 * i])


def test_quaternion_convention(tmp_path):
    _write_mocap(tmp_path / "m.csv", {"base": _T((0, 0, np.pi / 2))}, 0.05)
    t, T = load_rigid_bodies(tmp_path / "m.csv")["base"]
    assert np.allclose(T[0, :3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-6)


def test_hold_average():
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 1.0, 1.0 / RATE)
    truth = _T((0.1, -0.2, 0.3), (10.0, 20.0, 300.0))
    T = np.repeat(truth[None], len(t), 0)
    T[:, :3, 3] += rng.normal(0, 0.1, (len(t), 3))
    T[:, :3, :3] = np.einsum("fij,jk->fik", Rotation.from_rotvec(rng.normal(0, 1e-3, (len(t), 3))).as_matrix(),
                             truth[:3, :3])
    m = hold_pose(t, T, 0.0, 1.0)
    assert np.linalg.norm(m[:3, 3] - truth[:3, 3]) < 0.05
    assert np.allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-9)          # a proper rotation
    T[30:] = np.nan                                                             # 70 % occluded
    assert hold_pose(t, T, 0.0, 1.0) is None


def test_missing_body_flags_pose(tmp_path):
    bodies = {"base": _T(), "m1": _T(pos=(0, 0, 160)), "m2": _T(pos=(0, 0, 320))}
    _write_mocap(tmp_path / "m.csv", bodies, 4.0, gaps=[("m2", 2.0, 3.0)])
    _write_log(tmp_path / "log.csv", [(0.5, 1.5), (2.0, 3.0)], 2)
    d = np.load(make_dataset(tmp_path / "log.csv", tmp_path / "m.csv", 2, "upright", out=tmp_path / "d.npz"))
    assert d["valid"].tolist() == [True, False]
    assert np.all(np.isnan(d["T_ends"][1]))                                    # never zeros
    assert np.allclose(d["T_ends"][0, 1], _T(pos=(0, 0, 320)), atol=1e-6)
    assert np.allclose(d["dl"][0], 0.0) and np.allclose(d["dl"][1], 0.1) and d["dl"].shape == (2, 6)
    assert np.allclose(d["mounting"], (0.0, 0.0, -1.0))


def test_base_frame(tmp_path):
    base = _T((0.2, -0.4, 1.0), (500.0, -300.0, 80.0))
    rel = [_T((0.1, 0.0, 0.05), (5.0, -2.0, 158.0)), _T((0.2, 0.1, 0.1), (15.0, -6.0, 315.0))]
    bodies = {"base": base, "m1": base @ rel[0], "m2": base @ rel[1]}
    _write_mocap(tmp_path / "m.csv", bodies, 2.0)
    _write_log(tmp_path / "log.csv", [(0.5, 1.5)], 2)
    d = np.load(make_dataset(tmp_path / "log.csv", tmp_path / "m.csv", 2, (1.0, 0.0, 0.0), out=tmp_path / "d.npz"))
    assert d["valid"].tolist() == [True]
    for j in range(2):
        assert np.allclose(d["T_ends"][0, j], rel[j], atol=1e-6)


def test_cap_offset_and_clock_offset(tmp_path):
    bodies = {"base": _T(), "m1": _T((0, np.pi / 2, 0), (0, 0, 150))}       # cluster 10 mm below the module end
    _write_mocap(tmp_path / "m.csv", bodies, 3.0, gaps=[("m1", 0.0, 0.3)])   # gap before the hold
    _write_log(tmp_path / "log.csv", [(100.5, 101.5)], 1)                     # servo-PC clock 100 s ahead
    d = np.load(make_dataset(tmp_path / "log.csv", tmp_path / "m.csv", 1, "hanging", cap_offset_mm=(-10, 0, 0),
                             clock_offset_s=-100.0, out=tmp_path / "d.npz"))
    assert d["valid"].tolist() == [True]
    assert np.allclose(d["T_ends"][0, 0][:3, 3], [0.0, 0.0, 160.0], atol=1e-6)  # offset in the cluster frame


def test_mountings_match_the_twin():
    from hardware.mocap import MOUNTINGS
    from tdcr.twin import MOUNTINGS as TWIN_MOUNTINGS
    assert MOUNTINGS == TWIN_MOUNTINGS
