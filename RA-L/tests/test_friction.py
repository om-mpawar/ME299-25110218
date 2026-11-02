"""Optional capstan friction between the tendons and their channels: along tendon i the tension is
lambda_i exp(-sigma_i mu Theta_i), Theta_i the backbone turning accumulated from the base, sigma_i = +1 while the
servo winds the tendon in, -1 while it pays out. mu = 0 is the frictionless twin."""
import numpy as np
import pytest

from tdcr.compliance import MASS_STD6
from tdcr.eb_prior import EBPrior, capstan_factors, lcp_pull_only, predict_gravity
from tdcr.twin import Twin, direction_from_sequence

ST = np.arange(0.0, 320.0 + 1e-9, 5.0)
BEND = np.array([[5.0, -2.5, -2.5, 3.0, -1.5, -1.5]])                 # bends both sections the same way
CAL = dict(log_axial=0.0, log_bend_x=0.0, log_bend_z=0.0, log_mass=0.0, tendon_c=0.01, free_len_mm=300.0,
           mass_kg=MASS_STD6)


def _bend_deg(s, k=32):
    """Bend angle at station k (default the end of section 1), deg."""
    return np.degrees(np.arccos(np.clip(s.rotations[k][2, 2], -1.0, 1.0)))


def test_pull_only_solver_matches_nnls_on_the_symmetric_map():
    eb = EBPrior(2)
    for dl in (BEND[0], np.array([3.0, 1.0, -2.0, -1.0, 2.0, 0.5]), -np.ones(6)):
        assert np.allclose(lcp_pull_only(eb.G, dl), eb.tensions(dl)[0], atol=1e-10)


def test_lcp_solution_is_complementary_for_a_non_symmetric_map():
    rng = np.random.default_rng(0)
    eb = EBPrior(2)
    M = eb.G @ np.diag(rng.uniform(0.5, 1.5, 6))                      # tensions scaled per tendon: not symmetric
    b = np.array([3.0, -1.0, -2.0, 1.0, 2.0, -0.5])
    lam = lcp_pull_only(M, b)
    r = M @ lam - b
    assert lam.min() >= 0 and r.min() > -1e-10 and abs(lam @ r) < 1e-10


def test_capstan_factors_decay_while_pulling_and_grow_while_releasing():
    eb = EBPrior(2)
    w, _ = eb.strains(eb.tensions(BEND))
    up = capstan_factors(eb, w, np.ones((1, 6)), 0.3)
    down = capstan_factors(eb, w, -np.ones((1, 6)), 0.3)
    for i in range(6):
        path = eb.through[i]
        assert np.all(np.diff(up[0, i, path]) <= 1e-15) and np.all(np.diff(down[0, i, path]) >= -1e-15)
        assert up[0, i, path][0] > 0.999 and up[0, i, path][-1] < 0.98 < 1.02 < down[0, i, path][-1]
    assert up[0, 3, eb.through[3]][-1] < up[0, 0, eb.through[0]][-1]  # distal tendons wrap more of the backbone
    assert np.array_equal(capstan_factors(eb, w, np.zeros((1, 6)), 0.3), np.ones_like(up))


def test_zero_friction_is_the_frictionless_twin():
    eb = EBPrior(2)
    a = predict_gravity(eb, BEND, ST, mass=MASS_STD6)
    b = predict_gravity(eb, BEND, ST, mass=MASS_STD6, mu=0.0, direction=np.ones((1, 6)))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])


def test_no_direction_means_no_friction():
    """sigma = 0 everywhere: the friction path (non-symmetric solver) must give the frictionless shape."""
    eb = EBPrior(2)
    a = predict_gravity(eb, BEND, ST, mass=MASS_STD6, tol=1e-10, iters=400)
    b = predict_gravity(eb, BEND, ST, mass=MASS_STD6, tol=1e-10, iters=400, mu=0.3, direction=np.zeros((1, 6)))
    assert np.abs(a[0] - b[0]).max() < 1e-6


def test_hysteresis_loading_bends_less_than_unloading():
    """Displacement control: pulling, friction leaves less tension distally, so the servo needs more tension (more
    stretch) for the same travel and the section bends less; releasing, the reverse. Without tendon stretch (and
    gravity) the travel nearly fixes the bend and friction mostly moves curvature along the section."""
    up = Twin(2, (0.0, 0.0, 0.0), calib_json={**CAL, "mu": 0.3}, tol=1e-9).shape(BEND[0], direction=np.ones(6))
    free = Twin(2, (0.0, 0.0, 0.0), calib_json=CAL, tol=1e-9).shape(BEND[0])
    down = Twin(2, (0.0, 0.0, 0.0), calib_json={**CAL, "mu": 0.3}, tol=1e-9).shape(BEND[0], direction=-np.ones(6))
    assert _bend_deg(up) < _bend_deg(free) - 0.02 and _bend_deg(down) > _bend_deg(free) + 0.02
    assert up.tensions[0] > free.tensions[0] > down.tensions[0]
    rigid = {**CAL, "tendon_c": 0.0}
    a = Twin(2, (0.0, 0.0, 0.0), calib_json={**rigid, "mu": 0.3}, tol=1e-9).shape(BEND[0], direction=np.ones(6))
    b = Twin(2, (0.0, 0.0, 0.0), calib_json=rigid, tol=1e-9).shape(BEND[0])
    assert abs(_bend_deg(a) - _bend_deg(b)) < 0.01 and np.abs(a.positions - b.positions).max() > 0.1


def test_friction_converges_under_gravity():
    for g in ("upright", "hanging", "horizontal_x"):
        s = Twin(2, g, calib_json={**CAL, "mu": 0.3}).shape(BEND[0], direction=-np.ones(6))
        assert s.converged and np.all(np.isfinite(s.positions))


def test_direction_from_the_command_sequence():
    dl = np.array([[2.0, 0.0], [3.0, 0.0], [3.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
    assert np.array_equal(direction_from_sequence(dl), [[1, 1], [1, 1], [1, -1], [-1, -1], [-1, -1]])


def test_twin_tracks_the_direction_from_its_commands():
    tw = Twin(2, (0.0, 0.0, 0.0), calib_json={**CAL, "mu": 0.3}, tol=1e-9)
    a1 = _bend_deg(tw.shape(BEND[0]))                                    # from home: wound in
    tw.shape(1.5 * BEND[0])
    a2 = _bend_deg(tw.shape(BEND[0]))                                    # back down: tendon 1 paid out, others in
    a3 = _bend_deg(tw.shape(BEND[0]))                                    # same command: direction held
    assert a2 > a1 + 0.04 and a3 == pytest.approx(a2, abs=1e-6)


def test_calibration_recovers_friction(tmp_path):
    from tests.test_calibrate import TRUE, _dataset_sequence
    from tdcr.calibrate import fit
    true = {**TRUE, "mu": 0.25}
    files = [_dataset_sequence(tmp_path / "up.npz", 2, "upright", true, seed=1),          # 0.3 mm mocap noise
             _dataset_sequence(tmp_path / "hx.npz", 2, "horizontal_x", true, seed=2)]
    calib = fit(files, n_poses=15, seed=0, mass_kg=MASS_STD6 * np.exp(TRUE["log_mass"]), fit_friction=True)
    assert abs(calib["mu"] - 0.25) < 0.05 and abs(calib["log_bend_x"] - 0.2) < 0.05
