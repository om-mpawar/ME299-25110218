"""Command lists for the first hardware sessions (hardware/pose_lists.py): every list is safe for run_poses, the
pretension pulls every tendon to the same tension, and the calibration lists hold up-and-down passes for the friction
direction."""
import numpy as np
import pytest

from tdcr.hardware import pose_lists as pl
from tdcr.hardware.run_poses import read_commands
from tdcr.twin import direction_from_sequence


def _straight(x):
    return np.allclose(x, x[0] * pl.pretension_pattern(len(x) // 3))


@pytest.fixture(scope="module")
def sets(tmp_path_factory):
    out = tmp_path_factory.mktemp("poses")
    return out, pl.make_all(str(out), pretension_mm=2.0, seed=0)


def test_every_list_is_safe_and_readable(sets):
    out, made = sets
    assert {"standup_4s", "calib_2s", "heldout_2s", "test_4s_upright", "test_4s_hanging"} <= set(made)
    for name, m in made.items():
        dl = read_commands(m["csv"])
        assert dl.shape == (m["rows"], 3 * m["n"])
        assert np.abs(dl).max() <= pl.MAX_DL_MM
        assert m["pred_bend_max_deg"] <= m["max_bend_deg"] + 1e-9 and m["pred_load_max"] <= pl.MAX_LOAD


def test_every_list_stands_with_gravity(sets):
    """Every row converges in the twin with gravity, run in order as tdcr/hw_eval.py runs it, in each mounting the list
    is used in (the gravity-free screen alone let 18 of 25 4-module upright rows fold over: code review W1)."""
    out, made = sets
    for name, m in made.items():
        assert m["stands"] and all(v == m["rows"] for v in m["stands"].values()), (name, m["stands"])
    assert set(made["test_4s_upright"]["stands"]) == {"upright"}
    assert set(made["calib_2s"]["stands"]) == {"upright", "horizontal_x"}


def test_stands_rejects_a_folding_stack():
    """Row 6 of the old 4-module upright list (a target up to 25 deg per section, 2 mm pretension) folds over."""
    dl = np.array([1.649, 5.047, -0.696, 2.706, 4.959, 2.857, 6.3, 8.123, -0.888, 6.842, 10.905, -2.704])
    assert not pl.stands(4, [dl], ("upright",))[0]
    assert pl.stands(4, [dl], ("hanging",))[0]


def test_targets_gives_up_when_nothing_stands(monkeypatch):
    monkeypatch.setattr(pl, "MAX_DRAWS", 2)
    monkeypatch.setattr(pl, "stands", lambda n, dl, mountings: np.zeros(len(dl), bool))
    with pytest.raises(ValueError, match="raise the pretension or lower the bend"):
        pl.targets(4, 2, 15.0, 0, 2.0, ("upright",))


def test_pretension_loads_every_tendon_alike():
    """Equal pulls would leave the distal tendons slack (the proximal ones already shorten their paths)."""
    for n in (2, 4):
        eb = pl.screen_model(n)
        lam = eb.tensions(2.0 * pl.pretension_pattern(n))[0]
        assert lam.min() > 0 and np.ptp(lam) < 1e-9 * lam.max()
        flat = eb.tensions(np.full(3 * n, 2.0))[0]
        assert flat.min() < 1e-12 * flat.max()                          # equal pulls: a distal tendon goes slack
    assert np.allclose(pl.pretension_pattern(4)[::3], [1.0, 1.75, 2.25, 2.5], atol=0.01)   # tendons through module j


def test_standup_ramps_the_pretension_up_and_down():
    rows = pl.standup(4, p_max=4.0, step=0.5)
    dl = np.array([r["dl"] for r in rows])
    assert np.allclose(dl, dl[:, :1] * pl.pretension_pattern(4)[None])
    p = dl[:, 0]
    k = int(np.argmax(p))
    assert p[0] == 0 and p[-1] == 0 and p[k] == 4.0
    assert np.all(np.diff(p[:k + 1]) > 0) and np.all(np.diff(p[k:]) < 0)


def test_calibration_list_has_matched_up_and_down_poses(sets):
    _, made = sets
    dl = read_commands(made["calib_2s"]["csv"])
    assert 45 <= len(dl) <= 55
    sig = direction_from_sequence(dl)
    pairs = [(i, j) for i in range(len(dl)) for j in range(i + 1, len(dl)) if np.allclose(dl[i], dl[j])
             and not _straight(dl[i])]                                  # the same bent command reached twice
    assert any(np.any(sig[i] != sig[j]) for i, j in pairs)              # ... once winding in, once paying out


def test_lists_hold_the_pretension(sets):
    _, made = sets
    for name in ("calib_2s", "test_4s_upright", "test_4s_hanging"):
        dl = read_commands(made[name]["csv"])
        p0 = made[name]["pretensions"][0]                                 # starts straight at the first pretension
        assert np.allclose(dl[0], p0 * pl.pretension_pattern(made[name]["n"]))


def test_heldout_differs_from_calibration(sets):
    _, made = sets
    a, b = read_commands(made["calib_2s"]["csv"]), read_commands(made["heldout_2s"]["csv"])
    bent = [x for x in b if not _straight(x)]
    assert bent and not any(np.allclose(x, y) for x in bent for y in a)


def test_deterministic(tmp_path):
    a = pl.make_all(str(tmp_path / "a"), pretension_mm=2.0, seed=3)
    b = pl.make_all(str(tmp_path / "b"), pretension_mm=2.0, seed=3)
    assert np.array_equal(read_commands(a["calib_2s"]["csv"]), read_commands(b["calib_2s"]["csv"]))
