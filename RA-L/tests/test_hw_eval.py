"""Hardware evaluation (plan Task 8): metrics and the cheap baselines, end to end on synthetic datasets."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tdcr.hw_eval import MODELS, cc_ends, evaluate, run_eval
from tdcr.compliance import MASS_STD6, load_set
from tests.test_calibrate import TRUE, _dataset


def _T(rotvec=(0, 0, 0), pos=(0, 0, 0)):
    T = np.eye(4); T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix(); T[:3, 3] = pos
    return T


def test_metrics_known_offsets():
    true = np.stack([np.stack([_T(pos=(0, 0, 160)), _T(pos=(0, 0, 320))])] * 4)
    pred = true.copy()
    pred[:, -1] = [_T((0, 0, np.radians(1.0)), (2.0, 0, 320)) for _ in range(4)]
    m = evaluate(pred, true, length_mm=320.0)
    assert np.isclose(m["tip_mm"], 2.0) and np.isclose(m["tip_deg"], 1.0)
    assert np.isclose(m["tip_pct"], 100 * 2.0 / 320.0) and np.isclose(m["ends_mm"], 1.0)
    assert np.isclose(m["tip_mm_95"], 2.0)


def test_invalid_truth_and_failed_predictions_are_counted():
    true = np.tile(np.eye(4), (3, 1, 1, 1))
    pred = true.copy()
    pred[1, 0, 0, 3] = 1000  # finite but explicitly invalid ground truth
    pred[2] = np.nan         # failed prediction must remain visible in counts
    m = evaluate(pred, true, 160, valid=[True, False, True])
    assert m["tip_mm"] == 0 and m["n_poses"] == 1
    assert m["n_valid_truth"] == 2 and m["n_failed_predictions"] == 1
    m = evaluate(np.full_like(pred, np.nan), true, 160)
    assert m["n_poses"] == 0 and m["n_failed_predictions"] == 3
    assert m["tip_mm"] is None


def test_calibration_test_overlap_rejected_before_fit(tmp_path):
    p = str(tmp_path / "dataset.npz")
    with pytest.raises(ValueError, match="separate files"):
        run_eval([p], [p])


def test_copied_calibration_file_rejected_before_fit(tmp_path):
    import shutil
    a = tmp_path / "calib.npz"
    np.savez(a, x=np.arange(3))
    shutil.copy(a, tmp_path / "test.npz")
    with pytest.raises(ValueError, match="same contents"):
        run_eval([str(a)], [str(tmp_path / "test.npz")])


def _fake_eval(monkeypatch, data, failed=()):
    """run_eval on in-memory datasets: data = {path: (dl_cmd rows, n)}; the twin fails on the poses in `failed`."""
    import tdcr.hw_eval as hw
    monkeypatch.setattr(hw, "fit", lambda *a, **kw: dict(poses_used=[], gap_mm=2., joint_mass_kg=0., tip_mass_kg=0.))
    def load(paths):
        out = []
        for p in paths:
            dl = np.asarray(data[p], float)
            out.append(dict(path=p, n=1, mounting=(0, 0, 0), dl=dl, dl_cmd=dl, valid=np.ones(len(dl), bool),
                            T_ends=np.array([cc_ends(x, 1) for x in dl])))
        return out
    def twin(d, *a, **kw):
        T = d["T_ends"].copy()
        T[[k for k in failed if k < len(T)]] = np.nan
        return T
    monkeypatch.setattr(hw, "load_datasets", load)
    monkeypatch.setattr(hw, "twin_ends", twin)
    monkeypatch.setattr(hw, "gp_ends", lambda *a: None)
    return hw


def test_test_poses_repeating_a_calibration_command_are_left_out(tmp_path, monkeypatch, capsys):
    """The straight rows of the held-out list repeat the calibration list's straight rows (code review W3)."""
    straight, bent = [2.0, 2.0, 2.0], [[3.0, 1.0, 2.0], [1.0, 3.0, 2.5], [2.5, 2.0, 1.0]]
    hw = _fake_eval(monkeypatch, {"c": [straight, [4.0, 1.0, 1.0]], "t": [straight] + bent + [straight]})
    res, _ = hw.run_eval(["c"], ["t"], pred_dir=tmp_path)
    for m in res["t"].values():
        assert m["n_shared_with_calibration"] == 2 and m["n_poses"] == 3
    assert "2 of 5 poses repeat a calibration command" in capsys.readouterr().out


def test_every_model_is_scored_on_the_same_poses(tmp_path, monkeypatch):
    """A twin failure removes the pose from every model's score, not only the twin's (code review W1)."""
    hw = _fake_eval(monkeypatch, {"c": [[9.0, 0.0, 0.0]], "t": [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]},
                    failed=(1,))
    res, table = hw.run_eval(["c"], ["t"], pred_dir=tmp_path)
    r = res["t"]
    assert {m["n_poses"] for m in r.values()} == {2}
    assert r["physics"]["n_failed_predictions"] == 1 and r["cc"]["n_failed_predictions"] == 0
    assert "scored on the 2 of 3 valid poses" in table


def test_duplicate_basenames_preserve_both_sessions(tmp_path, monkeypatch):
    import tdcr.hw_eval as hw
    paths = [str(tmp_path / s / "dataset.npz") for s in ("two", "four")]
    monkeypatch.setattr(hw, "fit", lambda *a, **kw: dict(poses_used=[], gap_mm=2., joint_mass_kg=0., tip_mass_kg=0.))
    def load(paths):
        return [dict(path=p, n=1, mounting=(0, 0, 0), dl=np.zeros((1, 3)),
                     T_ends=cc_ends(np.zeros(3), 1)[None], valid=np.ones(1, bool)) for p in paths]
    monkeypatch.setattr(hw, "load_datasets", load)
    monkeypatch.setattr(hw, "twin_ends", lambda d, *a, **kw: d["T_ends"])
    monkeypatch.setattr(hw, "gp_ends", lambda *a: None)
    result, _ = run_eval([], paths, pred_dir=tmp_path)
    assert set(result) == {"two__dataset", "four__dataset"}


def test_cc_is_straight_at_zero_and_bends_towards_tendon_1():
    T = cc_ends(np.zeros(6), 2)
    assert np.allclose(T[-1][:3, 3], [0, 0, 320]) and np.allclose(T[-1][:3, :3], np.eye(3))
    T = cc_ends(np.array([2.0, -1.0, -1.0, 0, 0, 0]), 2)                 # lab tendon 1 at lab azimuth 0
    assert T[0][0, 3] > 1.0 and abs(T[0][1, 3]) < 1e-9


def test_end_to_end_synthetic(tmp_path, capsys):
    cmd2, cmd4 = load_set("std6_2s_sym_test_g0")["dl"], load_set("std6_4s_sym_test_g0")["dl"]
    calib = [_dataset(tmp_path / "c_up.npz", 2, "upright", cmd2[:8], TRUE, noise_mm=0.2, seed=1),
             _dataset(tmp_path / "c_hx.npz", 2, "horizontal_x", cmd2[8:16], TRUE, noise_mm=0.2, seed=2)]
    test = [_dataset(tmp_path / "t_hang4.npz", 4, "hanging", cmd4[:10], TRUE, noise_mm=0.2, seed=3),
            _dataset(tmp_path / "t_up2.npz", 2, "upright", cmd2[20:30], TRUE, noise_mm=0.2, seed=4)]
    res, table = run_eval(calib, test, n_poses=15, seed=0, mass_kg=MASS_STD6 * np.exp(TRUE["log_mass"]),
                          pred_dir=tmp_path)
    r4 = res["t_hang4"]
    assert r4["calibrated"]["tip_mm"] < r4["physics"]["tip_mm"] < r4["cc"]["tip_mm"]
    assert "gp" not in r4 and "gp" in res["t_up2"]                       # a 2-module GP cannot take 4-module commands
    assert "not applicable" in capsys.readouterr().out


def test_missing_baseline_skipped(tmp_path, capsys):
    cmd2 = load_set("std6_2s_sym_test_g0")["dl"]
    calib = [_dataset(tmp_path / "c.npz", 2, "upright", cmd2[:8])]
    test = [_dataset(tmp_path / "t.npz", 2, "upright", cmd2[8:12])]
    res, table = run_eval(calib, test, n_poses=8, seed=0, pred_dir=tmp_path, max_nfev=2)
    assert "sofa_mor" not in table and "sofa_mor" not in res["t"]
    assert "sofa_mor" in capsys.readouterr().out and "sofa_mor" in MODELS
