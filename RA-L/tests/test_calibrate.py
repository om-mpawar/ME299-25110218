"""Few-shot calibration of the hardware terms (plan Task 7), on synthetic datasets made by the twin itself with known
parameters, in the format of hardware/mocap.make_dataset."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tdcr.compliance import MASS_STD6, load_set
from tdcr.calibrate import FITTED, PARAMS, fit
from tdcr.twin import Twin

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


@pytest.fixture(scope="module")
def commands():
    return load_set("std6_2s_sym_test_g0")["dl"]


def test_recovers_known_params(tmp_path, commands):
    files = [_dataset(tmp_path / "up.npz", 2, "upright", commands[:8], TRUE, noise_mm=0.3, seed=1),
             _dataset(tmp_path / "hx.npz", 2, "horizontal_x", commands[8:16], TRUE, noise_mm=0.3, seed=2)]
    true_mass = MASS_STD6 * np.exp(TRUE["log_mass"])                  # weighed on a scale, not fitted:
    calib = fit(files, n_poses=15, seed=0, mass_kg=true_mass)          # shape data cannot separate stiffness and mass
    assert abs(calib["log_bend_x"] - 0.2) < 0.05
    assert abs(calib["mass_kg"] * np.exp(calib["log_mass"]) - true_mass) < 1e-9
    assert abs(calib["tendon_c"] - 0.01) < 0.004
    assert set(PARAMS) <= set(calib) and calib["free_len_mm"] == 300.0


def test_recovers_the_moment_arm(tmp_path, commands):
    """A wire bearing 0.4 mm inside SOFA's cable radius: the compliance scales cannot absorb it, the arm term recovers
    it (physics audit K1)."""
    true = dict(TRUE, d_arm_mm=-0.4)
    files = [_dataset(tmp_path / "up.npz", 2, "upright", commands[:8], true, noise_mm=0.3, seed=1),
             _dataset(tmp_path / "hx.npz", 2, "horizontal_x", commands[8:16], true, noise_mm=0.3, seed=2)]
    calib = fit(files, n_poses=15, seed=0, mass_kg=MASS_STD6 * np.exp(TRUE["log_mass"]))
    assert abs(calib["d_arm_mm"] + 0.4) < 0.03
    assert abs(calib["log_bend_x"] - 0.2) < 0.05 and abs(calib["tendon_c"] - 0.01) < 0.004


def test_calibration_without_the_arm_still_loads(commands):
    """Calibration files written before the arm term (no d_arm_mm) mean the SOFA cable radius."""
    old = dict(TRUE)
    a = Twin(2, "upright", calib_json=old, tol=1e-9).shape(commands[3]).positions
    b = Twin(2, "upright", calib_json=dict(old, d_arm_mm=0.0), tol=1e-9).shape(commands[3]).positions
    c = Twin(2, "upright", calib_json=dict(old, d_arm_mm=-0.4), tol=1e-9).shape(commands[3]).positions
    assert np.array_equal(a, b) and np.linalg.norm(c[-1] - a[-1]) > 1.0


def test_arm_does_not_leak_into_the_next_build():
    """The prior object is cached between builds: a calibrated arm must not survive into a plain build."""
    from tdcr.compliance import ModuleModel
    from tdcr.eb_prior import R_BEAR
    m, theta = ModuleModel.from_result(json.load(open("results/mc_std6.json")))
    hw = dict(log_axial=0.0, log_bend_x=0.0, log_bend_z=0.0, tendon_c=0.0, free_len_mm=300.0, d_arm_mm=-0.4)
    assert np.isclose(np.linalg.norm(m.build(theta, 2, hw=hw).D[0]), R_BEAR - 0.4)
    assert np.isclose(np.linalg.norm(m.build(theta, 2).D[0]), R_BEAR)


def test_identity_when_exact(tmp_path, commands):
    files = [_dataset(tmp_path / "up.npz", 2, "upright", commands[:8]),
             _dataset(tmp_path / "hx.npz", 2, "horizontal_x", commands[8:16])]
    calib = fit(files, n_poses=15, seed=0)
    for k in FITTED:
        assert abs(calib[k]) < 1e-3, (k, calib[k])
    assert calib["log_mass"] == 0.0 and calib["mass_kg"] == MASS_STD6


def test_uses_only_n_poses(tmp_path, commands):
    f = _dataset(tmp_path / "big.npz", 2, "upright", commands[:60])
    a = fit([f], n_poses=15, seed=3, max_nfev=2)
    b = fit([f], n_poses=15, seed=3, max_nfev=2)
    assert len(a["poses_used"]) == 15 and a["poses_used"] == b["poses_used"]
    assert len(a["residual_mm"]) == 15


def test_mass_is_not_identifiable_from_shape(tmp_path, commands):
    """Why mass is measured: scaling every compliance by k and the mass by 1/k leaves the shape nearly unchanged."""
    files = [_dataset(tmp_path / "up.npz", 2, "upright", commands[:8]),
             _dataset(tmp_path / "hx.npz", 2, "horizontal_x", commands[8:16])]
    a = dict(log_axial=0.0, log_bend_x=0.0, log_bend_z=0.0, log_mass=0.0, tendon_c=0.0, free_len_mm=300.0)
    b = dict(a, log_axial=0.3, log_bend_x=0.3, log_bend_z=0.3, log_mass=-0.3)
    pa = Twin(2, "horizontal_x", calib_json=a, tol=1e-9).shape(commands[9]).positions
    pb = Twin(2, "horizontal_x", calib_json=b, tol=1e-9).shape(commands[9]).positions
    assert np.abs(pa - pb).max() < 0.5                                  # mm, for a 35 % trade in both
