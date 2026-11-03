"""Cosserat baseline (plan Task 9): a uniform Cosserat rod with pull-only tendons (tdcr/cosserat.py), driven by the
same lab-order tendon displacements as the twin. Gravity is not modelled by that rod."""
import numpy as np

from tdcr.baselines.cosserat_rod import calibrate_stiffness, predict
from tdcr.hw_eval import cc_ends
from tdcr.compliance import load_set


def test_matches_cc_without_load():
    """One module, all tendons taut, no external load: the rod is a constant-curvature arc with the same bend angle
    (tendon loads carry no shear), shortened uniformly by the mean pull (inextensible tendons, compressible rod).
    With two modules, section 2's held tendons resist section 1's bend: coupling that constant curvature lacks."""
    c = 1.0
    dl = 1.5 * np.cos(np.radians([0.0, 120.0, 240.0])) + c             # bend towards tendon 1; min pull 0.25 > 0: taut
    T, tau, ms = predict(dl[None], mounting=(0.0, 0.0, 0.0))
    assert np.all(tau[0] > 0.0)
    ref = cc_ends(dl, 1)[0]
    eps = c / 160.0
    assert np.abs(T[0, 0, :3, 3] - (1 - eps) * ref[:3, 3]).max() < 0.05
    assert np.allclose(T[0, 0, :3, :3], ref[:3, :3], atol=1e-4)


def test_tendon_1_bends_towards_lab_x():
    T, _, _ = predict(np.array([[2.0, -1.0, -1.0]]) + 0.2, mounting=(0.0, 0.0, 0.0))
    tip = T[0, -1, :3, 3]
    assert tip[0] > 1.0 and abs(tip[1]) < 0.05 * tip[0] and tip[2] > 150.0   # lab frame, z along the backbone


def test_no_negative_tensions():
    dl = load_set("std6_2s_sym_test_g0")["dl"][:4]                      # pay-out commands: some tendons go slack
    T, tau, ms = predict(dl, mounting=(0.0, 0.0, 0.0))
    assert np.all(tau >= 0.0) and np.any(tau < 1e-9) and np.all(np.isfinite(T))


def test_timing_reported():
    dl = load_set("std6_2s_sym_test_g0")["dl"][:3]
    _, _, ms = predict(dl, mounting=(0.0, 0.0, 0.0))
    assert ms.shape == (3,) and np.all(ms > 0.0)


def test_calibration_recovers_axial_to_bending_ratio(tmp_path):
    dl = load_set("std6_2s_sym_test_g0")["dl"][:6]
    T, _, _ = predict(dl, mounting=(0.0, 0.0, 0.0), log_ratio=np.log(0.3))
    np.savez(tmp_path / "d.npz", dl=dl, dl_cmd=dl, T_ends=T, valid=np.ones(len(dl), bool),
             mounting=np.zeros(3), rows=np.arange(len(dl)), reps=np.zeros(len(dl), int))
    calib = calibrate_stiffness([str(tmp_path / "d.npz")], n_poses=6)
    assert abs(calib["log_ratio"] - np.log(0.3)) < 0.05


def test_predictions_of_two_sessions_do_not_collide(tmp_path, monkeypatch):
    """Code review W2: every session's file is dataset.npz; their predictions used to overwrite each other, and
    hw_eval scored one session's predictions against another's truth."""
    import pytest
    import tdcr.baselines.cosserat_rod as cr
    from tdcr.hw_eval import load_prediction, prediction_path
    from tests.test_calibrate import _dataset
    cmd = load_set("std6_2s_sym_test_g0")["dl"]
    paths = []
    for k, s in enumerate(("t_up", "t_hx")):
        (tmp_path / s).mkdir()
        paths.append(_dataset(tmp_path / s / "dataset.npz", 1, "upright", cmd[4 * k:4 * k + 4, :3]))
    monkeypatch.setattr(cr, "predict", lambda dl, mounting, log_ratio: (np.zeros((len(dl), 1, 4, 4)),
                                                                         np.zeros((len(dl), 3)), np.ones(len(dl))))
    out = cr.write_predictions(paths, dict(log_ratio=0.0), pred_dir=str(tmp_path))
    assert len(set(out)) == 2 and out == [prediction_path(str(tmp_path), "cosserat", p) for p in paths]
    assert load_prediction(out[0], paths[0], 4).shape == (4, 1, 4, 4)
    with pytest.raises(ValueError, match="was not made from"):
        load_prediction(out[0], paths[1], 4)
