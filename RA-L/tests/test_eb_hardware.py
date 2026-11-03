"""Hardware terms in the physics core (plan Task 1): tendon stretch added to the diagonal of the tendon map G, and
hardware compliance scales added to the identified notch log-scales."""
import json

import numpy as np
import pytest

from tdcr.compliance import ModuleModel, load_set

ZERO_HW = dict(log_axial=0.0, log_bend_x=0.0, log_bend_z=0.0, tendon_c=0.0, free_len_mm=300.0)


@pytest.fixture(scope="module")
def model():
    return ModuleModel.from_result(json.load(open("results/mc_std6.json")))


def _predict(m, th, d, hw=None):
    eb = m.build(th, d["n"], d["routing"], d["az"], d["holes"], hw=hw)
    return eb.predict(d["dl"], d["stations"])[0]


def test_zero_tendon_compliance_is_identical(model):
    m, th = model
    d = load_set("std6_2s_sym_test_g0")
    d = {k: (v[:10] if k in ("C", "R", "dl", "theta", "alpha") else v) for k, v in d.items()}
    plain = _predict(m, th, d)
    zero = _predict(m, th, d, ZERO_HW)
    assert np.abs(zero - plain).max() < 1e-12
    first = m.build(th, d["n"], d["routing"], d["az"], d["holes"], gap=2.0)
    G, cX, gap = first.G.copy(), first.cX.copy(), first.gap_after.copy()
    m.build(th, d["n"], d["routing"], d["az"], d["holes"], hw={**ZERO_HW, "tendon_c": 0.01, "log_bend_x": 0.3})
    # code review L1: a later build used to overwrite the object an earlier build returned (the prior is cached)
    assert np.array_equal(first.G, G) and np.array_equal(first.cX, cX) and np.array_equal(first.gap_after, gap)
    again = _predict(m, th, d)
    assert np.abs(again - plain).max() < 1e-12


def test_prior_cache_follows_the_geometry(model):
    """Code review L1: changing cad_json must change the profile, not reuse the cached D30 one."""
    import os
    from tdcr.paths import CAD
    m, th = model
    d = load_set("std6_2s_sym_test_g0")
    a = m.build(th, d["n"], d["routing"], d["az"], d["holes"])
    m2 = ModuleModel.from_result(json.load(open("results/mc_std6.json")))[0]
    m2.base = m.base                                                    # share the cache, as a reused model would
    m2.cad_json = os.path.join(CAD, "std6_tdcr_L150_D40.json")
    b = m2.build(th, d["n"], d["routing"], d["az"], d["holes"])
    assert not np.allclose(a.cX, b.cX)


def test_stretch_reduces_bend(model):
    m, th = model
    d = load_set("std6_1s_sym_test_g0")
    dl = np.array([[2.0, 0.0, 0.0]])
    eb = m.build(th, 1, d["routing"], d["az"], d["holes"])
    tip0 = eb.predict(dl, d["stations"])[0][0, -1]
    eb = m.build(th, 1, d["routing"], d["az"], d["holes"], hw={**ZERO_HW, "tendon_c": 1e-3})
    tip1 = eb.predict(dl, d["stations"])[0][0, -1]
    lateral = lambda p: np.hypot(p[0], p[2])                            # backbone axis is +y (scene frame)
    assert lateral(tip1) < lateral(tip0)


@pytest.mark.parametrize("c", [0.0, 1e-3, 1e-1])
def test_G_stays_spd(model, c):
    m, th = model
    d = load_set("std6_4s_sym_test_g0")
    eb = m.build(th, 4, d["routing"], d["az"], d["holes"], hw={**ZERO_HW, "tendon_c": c})
    np.linalg.cholesky(eb.G)
    assert np.allclose(eb.G, eb.G.T)
