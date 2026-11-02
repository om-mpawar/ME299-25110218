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
    tip = tw.shape(np.array([2.0, 0.0, 0.0])).positions[-1]             # lab tendon 1 sits at lab azimuth 0 deg
    assert tip[0] > 1.0 and abs(tip[1]) < 0.5 * tip[0]                  # bends towards +x (lab)
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
