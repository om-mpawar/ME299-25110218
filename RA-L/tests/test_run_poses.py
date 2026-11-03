"""Pose runs (plan Task 5): drive homed tendons through a command list and log every hold, with fake motors."""
import csv
import json

import numpy as np
import pytest

from tdcr.hardware.hysteresis_probe import mm_to_ticks, ticks_to_mm
from tdcr.hardware.run_poses import TENDON_ORDER, run

HALF_TICK_MM = ticks_to_mm(0.5)


class FakeMotor:
    """Position-mode servo: rejects goals outside 0..4095 as the XM430 does (mode 3), optionally fails on a goal."""

    def __init__(self, fail_on=None):
        self.pos, self.goals, self.fail_on = 0, [], fail_on

    def enable(self, profile_velocity=30): pass
    def disable(self): pass

    def goal(self, ticks):
        if len(self.goals) == self.fail_on:
            self.fail_on = None                                         # a one-off bus error
            raise RuntimeError(f"write goal_position {ticks}: comm=-3001")
        if not 0 <= ticks <= 4095:
            raise RuntimeError(f"write goal_position {ticks}: err=1")
        self.pos = int(ticks); self.goals.append(int(ticks))

    def position(self):
        return self.pos

    def current_ma(self):
        return 12.0


def _write_commands(path, rows):
    n = len(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"dl{i + 1}" for i in range(n)] + ["label"])
        for r in rows:
            w.writerow([f"{v:.6f}" for v in r] + [""])


def _write_zeros(path, names, directions=None, base=1000):
    z = {name: base + 100 * i for i, name in enumerate(names)}
    d = directions or {k: 1 for k in names}
    json.dump({**z, "threshold_ma": 60.0, "pulley_r_mm": 5.0, "time": "x", "directions": d}, open(path, "w"))
    return z


def test_ticks_round_trip():
    for dl in np.linspace(-15.0, 15.0, 301):
        assert abs(ticks_to_mm(mm_to_ticks(dl)) - dl) <= HALF_TICK_MM + 1e-12


def test_dry_run_log(tmp_path):
    names = TENDON_ORDER[:6]
    rows = [[1.0, -0.5, -0.5, 0.0, 0.0, 0.0], [0.0, 2.0, -1.0, 1.5, -0.75, -0.75], [-3.0, 1.0, 2.0, 0.2, 0.1, -0.3]]
    _write_commands(tmp_path / "cmd.csv", rows)
    zeros = _write_zeros(tmp_path / "zeros.json", names)
    motors = {n: FakeMotor() for n in names}
    log = run(tmp_path / "cmd.csv", tmp_path / "zeros.json", motors, tmp_path / "out", settle_s=0.0, hold_s=0.0)
    lines = list(csv.DictReader(open(log)))
    assert len(lines) == 3 * 6
    assert {"row", "t_start", "t_end", "tendon", "goal_ticks", "present_ticks", "current_ma"} <= set(lines[0])
    for ln in lines:
        dl = rows[int(ln["row"])][names.index(ln["tendon"])]
        assert int(ln["goal_ticks"]) == zeros[ln["tendon"]] + mm_to_ticks(dl)
        assert float(ln["t_end"]) >= float(ln["t_start"])
        assert abs(float(ln["dl_cmd_mm"]) - dl) < 1e-6                  # commanded, and reached (fake servo = goal)
        assert abs(float(ln["dl_meas_mm"]) - ticks_to_mm(int(ln["present_ticks"]) - zeros[ln["tendon"]])) < 1e-6


def test_rejects_wrong_width(tmp_path):
    _write_commands(tmp_path / "cmd.csv", [[0.0] * 6])
    _write_zeros(tmp_path / "zeros.json", TENDON_ORDER[:12])
    motors = {n: FakeMotor() for n in TENDON_ORDER[:12]}
    with pytest.raises(ValueError, match=r"6 .*12|12 .*6"):
        run(tmp_path / "cmd.csv", tmp_path / "zeros.json", motors, tmp_path / "out", settle_s=0.0, hold_s=0.0)
    assert all(not m.goals for m in motors.values())                      # nothing moved


def test_rejects_out_of_range_before_moving(tmp_path):
    _write_commands(tmp_path / "cmd.csv", [[0.0] * 3, [20.0, 0.0, 0.0]])
    _write_zeros(tmp_path / "zeros.json", TENDON_ORDER[:3])
    motors = {n: FakeMotor() for n in TENDON_ORDER[:3]}
    with pytest.raises(ValueError, match="15"):
        run(tmp_path / "cmd.csv", tmp_path / "zeros.json", motors, tmp_path / "out", settle_s=0.0, hold_s=0.0)
    assert all(not m.goals for m in motors.values())


def test_reversed_servo_winds_in(tmp_path):
    """direction -1 (ticks decrease winding in): + dl lowers the goal, and the log reads it back as + (code review H1)."""
    names = TENDON_ORDER[:3]
    _write_commands(tmp_path / "cmd.csv", [[3.0, -1.0, 0.5]])
    zeros = _write_zeros(tmp_path / "zeros.json", names, {"s1t1": -1, "s1t2": 1, "s1t3": -1})
    motors = {n: FakeMotor() for n in names}
    log = run(tmp_path / "cmd.csv", tmp_path / "zeros.json", motors, tmp_path / "out", settle_s=0.0, hold_s=0.0)
    assert motors["s1t1"].goals == [zeros["s1t1"] - mm_to_ticks(3.0)]
    assert motors["s1t2"].goals == [zeros["s1t2"] - mm_to_ticks(1.0)]
    assert motors["s1t3"].goals == [zeros["s1t3"] - mm_to_ticks(0.5)]
    for ln in csv.DictReader(open(log)):
        assert abs(float(ln["dl_meas_mm"]) - float(ln["dl_cmd_mm"])) <= HALF_TICK_MM + 1e-9


def test_goal_outside_the_servo_range_rejected_before_moving(tmp_path):
    """A zero near the end of the turn leaves no room for the pull: refuse before anything moves (code review H2)."""
    names = TENDON_ORDER[:3]
    _write_commands(tmp_path / "cmd.csv", [[0.0, 0.0, 0.0], [5.0, -2.0, 1.0]])
    _write_zeros(tmp_path / "zeros.json", names, base=3700)                 # s1t1 at 3700: 5 mm needs 652 ticks
    motors = {n: FakeMotor() for n in names}
    with pytest.raises(ValueError, match=r"servo range.*s1t1 \(zero 3700.*zero must be in 0\.\.3443"):
        run(tmp_path / "cmd.csv", tmp_path / "zeros.json", motors, tmp_path / "out", settle_s=0.0, hold_s=0.0)
    assert all(not m.goals for m in motors.values())


def test_abort_sends_every_servo_back_to_zero(tmp_path):
    names = TENDON_ORDER[:3]
    _write_commands(tmp_path / "cmd.csv", [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    zeros = _write_zeros(tmp_path / "zeros.json", names)
    motors = {n: FakeMotor() for n in names}
    motors["s1t2"].fail_on = 1                                              # the second row's goal fails
    with pytest.raises(RuntimeError):
        run(tmp_path / "cmd.csv", tmp_path / "zeros.json", motors, tmp_path / "out", settle_s=0.0, hold_s=0.0)
    for n in names:
        assert motors[n].goals[-1] == zeros[n]


def test_bad_direction_rejected(tmp_path):
    names = TENDON_ORDER[:3]
    _write_commands(tmp_path / "cmd.csv", [[0.0, 0.0, 0.0]])
    _write_zeros(tmp_path / "zeros.json", names, {"s1t1": 2, "s1t2": 1, "s1t3": 1})
    with pytest.raises(ValueError, match="directions"):
        run(tmp_path / "cmd.csv", tmp_path / "zeros.json", {n: FakeMotor() for n in names}, tmp_path / "out",
            settle_s=0.0, hold_s=0.0)
