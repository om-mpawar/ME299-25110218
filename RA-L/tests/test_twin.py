"""The Twin API (plan Task 3): tendon displacements + mounting -> backbone shape, in real time, in the lab frame."""
import json

import numpy as np

from tdcr.compliance import ModuleModel, load_set
from tdcr.twin import M_SCENE_TO_LAB, MOUNTINGS, Twin, scene_to_lab


def test_straight_at_zero_gravity_free():
    s = Twin(2, (0.0, 0.0, 0.0)).shape(np.zeros(6))
    assert s.converged
    assert np.abs(s.positions[:, :2]).max() < 1e-9                    # lab frame: backbone along +z
    assert np.isclose(s.positions[-1, 2], 320.0)
    assert np.allclose(s.rotations[-1], np.eye(3), atol=1e-12)


def test_matches_module_model():
    m, th = ModuleModel.from_result(json.load(open("results/mc_std6.json")))
    d = load_set("std6_4s_sym_test_g0")
    C, _, _ = m.predict(th, {**d, "dl": d["dl"][:10]})
    tw = Twin(4, (0.0, 0.0, 0.0))
    for i in range(10):
        s = tw.shape(d["dl"][i], frame="scene")
        assert np.abs(s.positions - C[i]).max() < 1e-9


def test_all_slack_under_gravity():
    s = Twin(2, "horizontal_x").shape(-3.0 * np.ones(6))
    assert s.converged and np.all(np.isfinite(s.positions))
    assert s.positions[-1, 0] > 1.0                                     # sags towards +x (lab), the gravity direction


def test_lab_scene_round_trip():
    dl = np.array([2.0, -1.0, -1.0, 0.5, 0.0, -0.5])
    lab = Twin(2, "horizontal_y").shape(dl)                             # two cold solves: a warm-started second
    scene = Twin(2, "horizontal_y").shape(dl, frame="scene")            # call agrees only to the solver tolerance
    assert np.allclose(lab.positions, scene_to_lab(scene.positions), atol=1e-9)
    assert np.allclose(lab.rotations, M_SCENE_TO_LAB @ scene.rotations @ M_SCENE_TO_LAB.T, atol=1e-12)
    assert np.allclose(lab.rotations @ np.transpose(lab.rotations, (0, 2, 1)), np.eye(3), atol=1e-9)


def test_tendon_order_is_the_lab_order():
    tw = Twin(1, (0.0, 0.0, 0.0))
    for k, az in enumerate((0.0, 120.0, 240.0)):                        # lab tendon k + 1 sits at lab azimuth az
        dl = np.zeros(3)
        dl[k] = 2.0
        tip = tw.shape(dl).positions[-1]
        assert np.hypot(tip[0], tip[1]) > 1.0
        assert abs((np.degrees(np.arctan2(tip[1], tip[0])) - az + 180.0) % 360.0 - 180.0) < 5.0, (k, tip)
    assert set(MOUNTINGS) == {"upright", "hanging", "horizontal_x", "horizontal_y"}


def test_rate():
    d = load_set("std6_4s_sym_test_g0")
    tw = Twin(4, "hanging")
    tw.shape(np.zeros(12))                                              # compile and warm up
    ms = [tw.shape((k / 499) * d["dl"][0]).solve_ms for k in range(500)]
    print(f"\nTwin, 4 modules hanging: median {np.median(ms):.2f} ms, max {np.max(ms):.2f} ms")
    assert np.median(ms) < 10.0


def test_unsupported_module_count_is_explained():
    import pytest
    with pytest.raises(ValueError, match="cable file for 3 modules"):
        Twin(3, "upright")


def test_invalid_call_does_not_change_command_history():
    import pytest
    tw = Twin(1, (0., 0., 0.))
    for kw in (dict(dl=[1., 0., 0.], frame="bad"), dict(dl=[np.nan, 0., 0.]),
               dict(dl=[1., 0., 0.], direction=[1., np.nan, 1.])):
        with pytest.raises(ValueError):
            tw.shape(**kw)
        np.testing.assert_array_equal(tw._prev, np.zeros(3))
        np.testing.assert_array_equal(tw._sigma, np.ones(3))


def test_calibration_dict_equals_file(tmp_path):
    calib = dict(log_axial=0.1, log_bend_x=-0.1, log_bend_z=0.05, log_mass=0.2, tendon_c=0.005, free_len_mm=250.0)
    json.dump(calib, open(tmp_path / "c.json", "w"))
    dl = np.array([1.0, -0.5, -0.5, 0.5, 0.0, -0.5])
    a = Twin(2, "upright", calib_json=calib, tol=1e-9).shape(dl)
    b = Twin(2, "upright", calib_json=str(tmp_path / "c.json"), tol=1e-9).shape(dl)
    c = Twin(2, "upright", tol=1e-9).shape(dl)
    assert np.allclose(a.positions, b.positions, atol=1e-12)
    assert np.abs(a.positions - c.positions).max() > 0.1             # the calibration does change the shape


def test_reset_restores_the_homed_state():
    """After reset, the twin answers as a fresh one (warm start and friction direction forgotten)."""
    a, b = Twin(2, "upright"), Twin(2, "upright")
    x = np.array([3.0, -1.0, 0.5, 1.0, 2.0, -0.5])
    a.shape(-x)
    a.reset()
    sa, sb = a.shape(x), b.shape(x)
    assert np.array_equal(sa.positions, sb.positions)


def test_rate_4_modules_upright():
    """Code review W1/H5: the live twin must converge and keep its rate on a 4-module upright bend (it used to restart
    cold on every call and never converge)."""
    import csv
    rows = list(csv.DictReader(open("data/hardware/poses/test_4s_upright.csv")))
    cols = [f"dl{k}" for k in range(1, 13)]
    seq = np.array([[float(r[c]) for c in cols] for r in rows])
    tw = Twin(4, "upright", gap_mm=2.0)
    tw.shape(seq[0])
    out = []
    for a, b in zip(seq[:-1], seq[1:]):                                 # 0.1 s ramps at 100 Hz between rows
        out += [tw.shape(a + (k / 10) * (b - a)) for k in range(1, 11)]
    assert all(s.converged for s in out)
    assert np.median([s.solve_ms for s in out]) < 10.0


def test_encoder_jitter_does_not_flip_friction():
    """Code review H4: +-1 tick (0.0077 mm) of read noise on a held pose used to flip the friction branch and move the
    tip by 4.7 mm (mu 0.3, stretching tendons); it must move it no more than the same noise does without friction."""
    from tdcr.compliance import MASS_STD6
    cal = dict(log_axial=0.0, log_bend_x=0.0, log_bend_z=0.0, log_mass=0.0, tendon_c=0.01, free_len_mm=300.0,
               mass_kg=MASS_STD6)
    held = np.array([5.0, -2.5, -2.5, 3.0, -1.5, -1.5]) + 2.0 * np.array([1, 1, 1, 1.5, 1.5, 1.5])

    def spread(mu):
        tw = Twin(2, "upright", calib_json={**cal, "mu": mu})
        tw.shape(held)
        rng = np.random.default_rng(0)
        tips = [tw.shape(held + 0.0077 * rng.integers(-1, 2, 6)).positions[-1] for _ in range(40)]
        return np.ptp(np.array(tips), axis=0).max(), tw

    with_mu, tw = spread(0.3)
    assert with_mu < spread(0.0)[0] + 0.1                               # 0.40 vs 0.37 mm (4.7 mm before)
    tw.shape(held - 0.1)                                                # a real reversal still registers
    assert np.all(tw._sigma == -1)


def test_calibration_keeps_its_compliance():
    """Code review L7: a calibration is only valid with the identification it was made against."""
    import pytest
    calib = dict(log_axial=0.0, log_bend_x=0.0, log_bend_z=0.0, log_mass=0.0, tendon_c=0.0, free_len_mm=300.0,
                 compliance_json="results/mc_std6.json")
    Twin(2, "upright", calib_json=calib)
    with pytest.raises(ValueError, match="calibration was made against"):
        Twin(2, "upright", compliance_json="results/mc_grav.json", calib_json=calib)
