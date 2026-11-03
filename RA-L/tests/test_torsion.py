"""Torsion of the backbone under gravity (physics audit C3): the twisting part of the gravity moment, with the
compliance identified from the SOFA twist test (sim/twist_test.py, scripts/torsion_identify.py)."""
import json

import numpy as np

from tdcr.compliance import ModuleModel, load_set
from tdcr.eb_prior import capstan_factors

MC = json.load(open("results/mc_std6.json"))


def _model(torsion):
    m, theta = ModuleModel.from_result(MC)
    m.torsion = torsion
    return m, theta


def test_identified_factor_is_in_the_model():
    t = json.load(open("results/torsion_std6.json"))
    assert 2.0 < t["factor"] < 3.0                                   # notch bridges twist ~2.5x plane-section theory
    assert abs(t["bending_check"]["about_x"] - 1) < 0.05 and abs(t["bending_check"]["about_z"] - 1) < 0.05
    assert MC["torsion"] == t["factor"]


def test_rigid_in_twist_without_a_factor():
    m, theta = _model(None)
    d = load_set("std6_2s_sym_test_gx")
    eb = m.build(theta, d["n"], d["routing"], d["az"], d["holes"])
    assert not eb.cT.any() and eb.cT_plane.min() > 0


def test_torsion_fixes_the_horizontal_mounting():
    """Gravity across a backbone bent out of its plane twists it; without torsion the 2-module horizontal sets were the
    worst in Table I (1.2 mm mean)."""
    d = load_set("std6_2s_sym_test_gx")
    d = dict(d, dl=d["dl"][:20], C=d["C"][:20])
    err = {}
    for k in (None, MC["torsion"]):
        m, theta = _model(k)
        C, _, _ = m.predict(theta, d)
        err[k] = np.linalg.norm(C[:, -1] - d["C"][:, -1], axis=1).mean()
    assert err[MC["torsion"]] < 0.6 < err[None]


def test_upright_and_gravity_free_unchanged():
    """No twisting moment without gravity, and little upright or hanging (tip moves 0.004 mm on average)."""
    for tag, tol in (("std6_2s_sym_test_g0", 0.0), ("std6_2s_sym_test", 0.1), ("std6_2s_sym_test_ghang", 0.1)):
        d = load_set(tag)
        d = dict(d, dl=d["dl"][:10], C=d["C"][:10])
        a = _model(None)[0].predict(_model(None)[1], d)[0]
        b = _model(MC["torsion"])[0].predict(_model(MC["torsion"])[1], d)[0]
        tip = np.linalg.norm(a[:, -1] - b[:, -1], axis=1)
        assert tip.max() <= tol + 1e-12 and tip.mean() <= tol / 5 + 1e-12, (tag, tip.max(), tip.mean())


def test_twist_does_not_turn_the_tendons():
    """Capstan friction counts the bending turn only: a pure twist leaves every tension factor at 1."""
    m, theta = _model(MC["torsion"])
    d = load_set("std6_2s_sym_test_gx")
    eb = m.build(theta, d["n"], d["routing"], d["az"], d["holes"])
    w = np.zeros((1, len(eb.yc), 3))
    w[..., 1] = 0.01
    assert np.allclose(capstan_factors(eb, w, np.ones((1, 3 * eb.n)), 0.3), 1.0)


def test_calibration_scales_torsion_with_bending():
    """A uniformly softer print (every log-scale +s) twists more by the same factor: Prop. 2's scale/mass invariance
    stays exact with torsion."""
    m, theta = _model(MC["torsion"])
    d = load_set("std6_2s_sym_test_gx")
    hw = dict(log_axial=0.3, log_bend_x=0.3, log_bend_z=0.3, tendon_c=0.0, free_len_mm=300.0)
    base = m.build(theta, 2, d["routing"], d["az"], d["holes"]).cT.copy()
    assert np.allclose(m.build(theta, 2, d["routing"], d["az"], d["holes"], hw=hw).cT, base * np.exp(0.3))
