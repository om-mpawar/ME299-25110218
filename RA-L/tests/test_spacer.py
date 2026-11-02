"""The printed joint (7 mm coupler on a 5 mm web): a straight rigid gap of gap_mm between consecutive modules, and the
coupler + dowel + plugs as a point mass at every joint (plus an optional tip mass)."""
import numpy as np

from analysis.module_compliance import MASS_STD6
from tdcr.eb_prior import EBPrior, _rodrigues, predict_gravity
from tdcr.twin import Twin

BEND = np.array([4.0, -2.0, -2.0, 0.0, 1.5, -1.5])                     # bends both sections (mm, lab order)


def test_gap_spaces_the_module_ends():
    s = Twin(2, (0.0, 0.0, 0.0), gap_mm=2.0).shape(np.zeros(6))
    assert np.isclose(s.positions[32, 2], 160.0) and np.isclose(s.positions[64, 2], 322.0)
    assert np.abs(s.positions[:, :2]).max() < 1e-9


def test_gap_is_a_rigid_step_along_the_joint_frame():
    """Without gravity the gap translates everything distal by gap * (backbone axis at the joint), frames unchanged."""
    a = Twin(2, (0.0, 0.0, 0.0)).shape(BEND)
    b = Twin(2, (0.0, 0.0, 0.0), gap_mm=2.0).shape(BEND)
    assert np.allclose(a.positions[:33], b.positions[:33], atol=1e-12)
    step = 2.0 * a.rotations[32][:, 2]                                   # lab z = backbone axis
    assert np.allclose(b.positions[33:] - a.positions[33:], step, atol=1e-9)
    assert np.allclose(a.rotations, b.rotations, atol=1e-12)


def test_gravity_march_matches_the_plain_frames():
    """predict_gravity with no weight integrates the gap exactly as EBPrior.frames."""
    eb = EBPrior(2, gap=2.0)
    st = np.arange(0.0, 320.0 + 1e-9, 5.0)
    C0, R0, _ = eb.predict(BEND[None], st)
    C1, R1, _, _ = predict_gravity(eb, BEND[None], st, mass=0.0)
    assert np.isclose(C0[0, 64, 1], C1[0, 64, 1]) and np.abs(C0 - C1).max() < 1e-9
    assert np.abs(R0 - R1).max() < 1e-9


def test_zero_gap_and_masses_change_nothing():
    a = Twin(2, "horizontal_x").shape(BEND)
    b = Twin(2, "horizontal_x", gap_mm=0.0, joint_mass_kg=0.0, tip_mass_kg=0.0).shape(BEND)
    assert np.array_equal(a.positions, b.positions)


def test_gap_lengthens_the_lever():
    a = Twin(2, "horizontal_x").shape(np.zeros(6))
    b = Twin(2, "horizontal_x", gap_mm=2.0).shape(np.zeros(6))
    assert b.positions[-1, 0] > a.positions[-1, 0]                    # sags further (towards +x, lab gravity)


def test_joint_and_tip_mass_add_sag():
    a = Twin(2, "horizontal_x", gap_mm=2.0).shape(np.zeros(6))
    j = Twin(2, "horizontal_x", gap_mm=2.0, joint_mass_kg=0.0046).shape(np.zeros(6))
    t = Twin(2, "horizontal_x", gap_mm=2.0, joint_mass_kg=0.0046, tip_mass_kg=0.0046).shape(np.zeros(6))
    assert a.positions[-1, 0] < j.positions[-1, 0] < t.positions[-1, 0]


def test_joint_mass_sits_at_the_joint():
    """4.6 g at the one joint sags the tip more than nothing, less than 4.6 g more in every module (a longer lever)."""
    eb = EBPrior(2, gap=2.0)
    st = np.arange(0.0, 320.0 + 1e-9, 5.0)
    g = (9810.0, 0.0, 0.0)
    C1, _, _, _ = predict_gravity(eb, np.zeros((1, 6)), st, g_vec=g, mass=MASS_STD6, joint_mass=0.0046, tol=1e-10,
                                  iters=400)
    C0, _, _, _ = predict_gravity(eb, np.zeros((1, 6)), st, g_vec=g, mass=MASS_STD6, tol=1e-10, iters=400)
    C2, _, _, _ = predict_gravity(eb, np.zeros((1, 6)), st, g_vec=g, mass=MASS_STD6 + 0.0046, tol=1e-10, iters=400)
    assert C0[0, -1, 0] < C1[0, -1, 0] < C2[0, -1, 0]


def test_tendon_stretch_uses_the_module_pitch():
    calib = dict(log_axial=0.0, log_bend_x=0.0, log_bend_z=0.0, log_mass=0.0, tendon_c=0.01, free_len_mm=300.0)
    eb = Twin(2, "upright", calib_json=calib, gap_mm=2.0).eb
    assert np.allclose(eb.tendon_compliance, 0.01 * (300.0 + 155.0 + np.array([0, 0, 0, 162.0, 162.0, 162.0])))
    eb0 = Twin(2, "upright", calib_json=calib).eb                       # the cached prior is reset between builds
    assert np.allclose(eb0.tendon_compliance[3:], 0.01 * (300.0 + 155.0 + 160.0))
    assert not np.any(eb0.gap_after)


def test_calibration_carries_the_joint():
    """gap / joint / tip masses are measured, travel in the calibration dict, and the twin reads them from it."""
    calib = dict(log_axial=0.0, log_bend_x=0.0, log_bend_z=0.0, log_mass=0.0, tendon_c=0.0, free_len_mm=300.0,
                 mass_kg=MASS_STD6, gap_mm=2.0, joint_mass_kg=0.0046, tip_mass_kg=0.001)
    a = Twin(2, "horizontal_x", calib_json=calib).shape(np.zeros(6))
    b = Twin(2, "horizontal_x", gap_mm=2.0, joint_mass_kg=0.0046, tip_mass_kg=0.001).shape(np.zeros(6))
    assert np.allclose(a.positions, b.positions, atol=1e-12)


def reference_frames(eb, lam, stations):
    """EBPrior.frames as committed in 7a66ff6 (pure-Python loop over cells), kept only here as the reference for the
    compiled one."""
    w, sh = eb.strains(lam)
    P, nc = sh.shape
    Rk = _rodrigues(w.reshape(-1, 3)).reshape(P, nc, 3, 3)
    p, R = np.zeros((P, 3)), np.repeat(np.eye(3)[None], P, 0)
    C, Rs = np.zeros((P, len(stations), 3)), np.zeros((P, len(stations), 3, 3))
    st = np.asarray(stations, float)
    k_out = 0
    if st[0] <= 0:
        C[:, 0], Rs[:, 0], k_out = p, R, 1
    for c in range(nc):
        Rh = R @ _rodrigues(0.5 * w[:, c])                          # half-step rotation (midpoint rule)
        p = p + np.einsum("pij,j->pi", Rh, np.array([0.0, 1.0, 0.0])) * (eb.cell - sh[:, c])[:, None]
        R = R @ Rk[:, c]
        while k_out < len(st) and abs(eb.edges[c + 1] - st[k_out]) < 1e-6:
            C[:, k_out], Rs[:, k_out] = p, R
            k_out += 1
        if eb.gap_after[c]:
            p = p + R[:, :, 1] * eb.gap_after[c]                  # rigid joint along the end frame's axis
    return C, Rs


def test_frames_match_the_python_reference():
    """Compiled frames (via _march) reproduce the per-cell Python loop on a bent 2-module stack with a 2 mm joint,
    including the module-end station 160 (taken before the gap)."""
    eb = EBPrior(2, gap=2.0)
    st = np.arange(0.0, 320.0 + 1e-9, 5.0)
    lam = eb.tensions(np.stack([BEND, -BEND, 0.5 * BEND]))
    C0, R0 = reference_frames(eb, lam, st)
    C1, R1 = eb.frames(lam, st)
    assert np.abs(C0[:, -1, [0, 2]]).max() > 40.0                      # really bent (tip off the y axis)
    assert np.abs(C1 - C0).max() < 1e-9 and np.abs(R1 - R0).max() < 1e-9
