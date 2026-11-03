"""Real-time gravity solve (plan Task 2): warm start, compiled frame integration, convergence flag.
The reference below is a verbatim copy of tdcr/eb_prior.predict_gravity as committed on 21 Oct 2026 (4ed511a), kept
only here so the faster implementation can be checked against it."""
import json
import time

import numpy as np
import pytest

from tdcr.compliance import E_FEM, ModuleModel, lab_to_scene, load_set
from tdcr.eb_prior import _rodrigues, predict_gravity


@pytest.fixture(scope="module")
def model():
    return ModuleModel.from_result(json.load(open("results/mc_std6.json")))


def _eb(model, tag):
    m, th = model
    d = load_set(tag)
    return m.build(th, d["n"], d["routing"], d["az"], d["holes"]), d


def test_matches_reference(model):
    eb, d = _eb(model, "std6_2s_sym_test")
    eb.cT = 0.0 * eb.cT                    # the reference is rigid in twist (torsion: tests/test_torsion.py)
    dl, st = d["dl"][:20], d["stations"]
    # same fixed point, solved tight: since 29 Oct the iteration also waits for the axial strains, so at the live
    # tolerance it stops later than the reference (closer to the fixed point), not at the same iterate
    C0, R0, lam0, _ = reference_predict_gravity(eb, dl, st, g_vec=d["g"], E=E_FEM, mass=d["mass"], tol=1e-10, iters=400)
    # tighter here: every pose now stops at its own tolerance, the reference at the batch's slowest pose
    C1, R1, lam1, info = predict_gravity(eb, dl, st, g_vec=d["g"], E=E_FEM, mass=d["mass"], tol=1e-12, iters=1000)
    assert np.abs(C1 - C0).max() < 1e-6
    assert np.abs(R1 - R0).max() < 1e-8                 # the reference itself stops ~1e-9 from the fixed point
    assert info["converged"] and info["iterations"] > 1


def test_batch_equals_one_at_a_time(model):
    """Code review W4: a pose's prediction must not depend on the other poses solved with it."""
    eb, d = _eb(model, "std6_2s_sym_test")
    dl, st = d["dl"][:12], d["stations"]
    kw = dict(g_vec=d["g"], E=E_FEM, mass=d["mass"])
    C, R, lam, info = predict_gravity(eb, dl, st, **kw)
    one = [predict_gravity(eb, x[None], st, **kw) for x in dl]
    assert np.abs(C - np.concatenate([o[0] for o in one])).max() < 1e-12
    assert np.abs(lam - np.concatenate([o[2] for o in one])).max() < 1e-12
    assert info["converged_poses"].all() and info["iterations"] == max(o[3]["iterations"] for o in one)


def test_a_diverging_pose_does_not_stop_the_others(model):
    """Code review W4: a pose past buckling used to stop the iteration of the whole batch. Upright, 1e8 times the
    weight: the bent pose blows up (NaN, failed), the straight one (an equilibrium, if unstable) still converges."""
    eb, d = _eb(model, "std6_4s_sym_test_g0")
    kw = dict(g_vec=lab_to_scene((0.0, 0.0, -9810.0)), E=E_FEM, mass=1e8 * d["mass"])
    dl = np.zeros((2, 12))
    dl[1, 0] = 3.0
    C, R, lam, info = predict_gravity(eb, dl, d["stations"], **kw)
    assert list(info["failed_poses"]) == [False, True] and list(info["converged_poses"]) == [True, False]
    assert np.isnan(C[1]).all() and np.isnan(lam[1]).all() and np.isfinite(info["state"]["wg"]).all()
    assert np.abs(C[0] - predict_gravity(eb, dl[:1], d["stations"], **kw)[0][0]).max() < 1e-12


def test_zero_iterations():
    """Code review L9: iters=0 used to raise NameError; it returns the tendon-only shape, not converged."""
    m, th = ModuleModel.from_result(json.load(open("results/mc_std6.json")))
    d = load_set("std6_2s_sym_test")
    eb = m.build(th, d["n"], d["routing"], d["az"], d["holes"])
    C, _, _, info = predict_gravity(eb, d["dl"][:2], d["stations"], g_vec=d["g"], iters=0)
    assert not info["converged"] and info["iterations"] == 0
    np.testing.assert_allclose(C, eb.predict(d["dl"][:2], d["stations"])[0], atol=1e-12)


def test_warm_start_fewer_iterations(model):
    eb, d = _eb(model, "std6_2s_sym_test")
    base = d["dl"][0]
    seq = [base + 0.05 * k * np.sign(base) for k in range(20)]           # 0.05 mm per call
    cold = sum(predict_gravity(eb, x[None], d["stations"], g_vec=d["g"], E=E_FEM, mass=d["mass"])[3]["iterations"]
               for x in seq)
    warm, state = 0, None
    for x in seq:
        info = predict_gravity(eb, x[None], d["stations"], g_vec=d["g"], E=E_FEM, mass=d["mass"], state=state)[3]
        warm += info["iterations"]; state = info["state"]
    assert warm < cold


def test_nonconvergence_flagged(model):
    eb, d = _eb(model, "std6_4s_sym_test_g0")
    tilt = np.radians(1.0)                                              # a 1 deg imperfection: the exactly straight column
    g = lab_to_scene((9810.0 * np.sin(tilt), 0.0, -9810.0 * np.cos(tilt)))   # is an (unstable) equilibrium
    C, R, lam, info = predict_gravity(eb, np.zeros((1, 12)), d["stations"], g_vec=g, E=E_FEM,
                                      mass=5 * d["mass"])
    assert info["converged"] is False


def test_axial_gravity_must_converge_even_without_bending(model):
    eb, d = _eb(model, "std6_2s_sym_test_g0")
    dl = np.full((1, 6), -10.0)  # slack tendons, straight hanging beam
    kw = dict(g_vec=(0, 9810, 0), mass=d["mass"], tol=1e-10, iters=100)
    C, R, _, info = predict_gravity(eb, dl, d["stations"], **kw)
    exact, _, _, _ = predict_gravity(eb, dl, d["stations"], relax=1.0, **kw)
    assert info["converged"] and info["iterations"] > 1
    assert C[0, -1, 1] > d["stations"][-1]
    np.testing.assert_allclose(C, exact, atol=1e-7)
    np.testing.assert_allclose(R, np.broadcast_to(np.eye(3), R.shape), atol=1e-12)


def test_rate_4_modules_hanging(model):
    eb, d = _eb(model, "std6_4s_sym_test_g0")
    g = lab_to_scene((0.0, 0.0, 9810.0))                                # hanging: gravity along +z (lab)
    target = d["dl"][0]
    predict_gravity(eb, np.zeros((1, 12)), d["stations"], g_vec=g, E=E_FEM, mass=d["mass"])   # compile / warm caches
    state, times = None, []
    for k in range(200):
        x = (k / 199) * target
        t0 = time.perf_counter()
        info = predict_gravity(eb, x[None], d["stations"], g_vec=g, E=E_FEM, mass=d["mass"], state=state)[3]
        times.append(time.perf_counter() - t0); state = info["state"]
        assert info["converged"]
    print(f"\n4 modules hanging: mean {1e3 * np.mean(times):.2f} ms, median {1e3 * np.median(times):.2f} ms "
          f"({1 / np.mean(times):.0f} Hz)")
    assert np.mean(times) < 0.010


# ---------------------------------------------------------------- reference (21 Oct 2026, verbatim)
def reference_predict_gravity(eb, dl, stations, g_vec=(0.0, -9810.0, 0.0), E=56000.0, mass=0.0346, iters=60, relax=0.5, tol=1e-4):
    """Variable-section Euler-Bernoulli prior under gravity (units mm, kg, s; E in kPa = mN/mm^2, SOFA's units).

    Fixed point: shape -> internal moment and force of the weight distal to every cell (in the cell frame) ->
    curvature and compression = (tendon + gravity load) / (E J), / (E A) -> pull-only tensions re-solved with the
    gravity-induced tendon path change -> shape. Mass per module distributed in proportion to the CAD area.
    Shear (eb.cS > 0, Timoshenko): the transverse part of the distal weight in each cell frame shears the cell, moving
    the centreline sideways by cS * F_transverse / E relative to the cross-section frame; frames are not rotated by it
    and, to first order, tendon path lengths do not change."""
    dl = np.atleast_2d(dl)
    P, nc = len(dl), len(eb.yc)
    # mass per cell from the CAD area: A_cell * cell / (sum over a module) * mass
    area = eb.cell / eb.cA_cad                          # cell length / integral(1/A) = harmonic-mean CAD area
    per_mod = len(eb.yc) // eb.n
    m_cell = np.concatenate([mass * area[j * per_mod:(j + 1) * per_mod] / area[j * per_mod:(j + 1) * per_mod].sum()
                             for j in range(eb.n)])
    Fg = m_cell[:, None] * np.asarray(g_vec, float)[None] / E          # (nc, 3) world force / E per cell
    lam = eb.tensions(dl)
    w_t, sh_t = eb.strains(lam)
    wg, shg = np.zeros((P, nc, 3)), np.zeros((P, nc))
    sxz = np.zeros((P, nc, 2))                                              # shear offset per cell (x, z), cell frame
    T = eb.through.astype(float)
    for it in range(iters):
        # current frames at cell centres
        w, sh = w_t + wg, sh_t + shg
        Rk = _rodrigues(w.reshape(-1, 3)).reshape(P, nc, 3, 3)
        p, R = np.zeros((P, 3)), np.repeat(np.eye(3)[None], P, 0)
        pc, Rc = np.zeros((P, nc, 3)), np.zeros((P, nc, 3, 3))
        for c in range(nc):
            Rh = R @ _rodrigues(0.5 * w[:, c])
            step = Rh[:, :, 1] * (eb.cell - sh[:, c])[:, None] + Rh[:, :, 0] * sxz[:, c, :1] + Rh[:, :, 2] * sxz[:, c, 1:]
            pc[:, c], Rc[:, c] = p + 0.5 * step, Rh
            p, R = p + step, R @ Rk[:, c]
        # distal weight and its moment about each cell centre (world), then in the cell frame
        Fsum = np.cumsum(Fg[::-1], 0)[::-1]                                 # cells c.. (inclusive)
        Fdist = np.concatenate([Fsum[1:], np.zeros((1, 3))])                # cells c+1..
        mom_all = np.cumsum(np.cross(pc, Fg[None])[:, ::-1], 1)[:, ::-1]    # sum_{c'>=c} p_c' x F_c'
        mom_dist = np.concatenate([mom_all[:, 1:], np.zeros((P, 1, 3))], 1)
        Mw = mom_dist - np.cross(pc, Fdist[None])                           # sum_{c'>c} (p_c' - p_c) x F_c'
        Ml = np.einsum("pcji,pcj->pci", Rc, Mw)                             # in the cell frame
        Fl = np.einsum("pcji,cj->pci", Rc, Fdist)
        kx, kz = eb.bend(-Ml[..., 2], Ml[..., 0])                          # moments towards +x, +z (tendon convention)
        wg_new = np.stack([kz, np.zeros_like(kz), -kx], -1)
        shg_new = -Fl[..., 1] * eb.cA                                       # compression when the distal weight pushes down
        sxz_new = Fl[..., [0, 2]] * eb.cS[None, :, None]                    # shear along the transverse distal force
        # gravity-induced tendon path shortening, then pull-only tensions with that offset
        h = np.einsum("ic,pc->pi", T, shg_new) + np.einsum("ic,pc,i->pi", T, kx, eb.D[:, 0]) \
            + np.einsum("ic,pc,i->pi", T, kz, eb.D[:, 1])
        lam_new = eb.tensions(dl - h)
        w_t_new, sh_t_new = eb.strains(lam_new)
        change = np.abs(wg_new - wg).max() + np.abs(w_t_new - w_t).max()
        wg, shg = (1 - relax) * wg + relax * wg_new, (1 - relax) * shg + relax * shg_new
        sxz = (1 - relax) * sxz + relax * sxz_new
        w_t, sh_t, lam = (1 - relax) * w_t + relax * w_t_new, (1 - relax) * sh_t + relax * sh_t_new, lam_new
        if change < tol:
            break
    # final frames at the stations
    w, sh = w_t + wg, sh_t + shg
    Rk = _rodrigues(w.reshape(-1, 3)).reshape(P, nc, 3, 3)
    p, R = np.zeros((P, 3)), np.repeat(np.eye(3)[None], P, 0)
    st = np.asarray(stations, float)
    C, Rs = np.zeros((P, len(st), 3)), np.zeros((P, len(st), 3, 3))
    k_out = 0
    if st[0] <= 0:
        C[:, 0], Rs[:, 0], k_out = p, R, 1
    for c in range(nc):
        Rh = R @ _rodrigues(0.5 * w[:, c])
        p = p + Rh[:, :, 1] * (eb.cell - sh[:, c])[:, None] + Rh[:, :, 0] * sxz[:, c, :1] + Rh[:, :, 2] * sxz[:, c, 1:]
        R = R @ Rk[:, c]
        while k_out < len(st) and abs(eb.edges[c + 1] - st[k_out]) < 1e-6:
            C[:, k_out], Rs[:, k_out] = p, R
            k_out += 1
    return C, Rs, lam, it + 1


def test_station_off_the_cell_edges_raises(model):
    """Code review L3: a station between cell edges used to come back as the origin with a zero rotation matrix."""
    eb, d = _eb(model, "std6_2s_sym_test_g0")
    with pytest.raises(ValueError, match="not a cell edge"):
        eb.predict(d["dl"][:1], [0.0, 5.25])
