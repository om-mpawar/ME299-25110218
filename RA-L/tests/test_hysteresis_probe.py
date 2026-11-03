"""Hysteresis probe (hardware/hysteresis_probe.py): the ramp runs from the tendon's zero, never from tick 0."""
import csv

import pytest

from tdcr.hardware.hysteresis_probe import SimulatedMotor, mm_to_ticks, run_probe


class FakeServo:
    """Position-mode servo that starts wherever it was left (here mid-turn)."""

    def __init__(self, pos=3000):
        self.pos, self.goals, self.closed = pos, [], False

    def enable(self, profile_velocity=30): pass
    def close(self): self.closed = True

    def goal(self, ticks):
        if not 0 <= ticks <= 4095:
            raise RuntimeError(f"write goal_position {ticks}: err=1")
        self.pos = int(ticks); self.goals.append(int(ticks))

    def position(self):
        return self.pos

    def current_ma(self):
        return 10.0


def _probe(motor, tmp_path, **kw):
    return run_probe(motor, 2.0, 1, 5, 0.0, 900.0, str(tmp_path / "h.csv"), **kw)


def test_ramp_starts_and_ends_at_the_start_position(tmp_path, monkeypatch):
    """Without a zero file the zero is where the servo is (code review H3: it used to jump to tick 0)."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    m = FakeServo(3000)
    rows = _probe(m, tmp_path)
    assert min(m.goals) == 3000 and max(m.goals) == 3000 + mm_to_ticks(2.0) and m.goals[-1] == 3000
    assert rows[0][4] == 0 and max(r[4] for r in rows) == mm_to_ticks(2.0)     # logged relative to the zero
    assert m.closed


def test_ramp_follows_the_homed_zero_and_direction(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    m = FakeServo(2500)
    rows = _probe(m, tmp_path, zero=2400, direction=-1)
    assert max(m.goals) == 2400 and min(m.goals) == 2400 - mm_to_ticks(2.0) and m.goals[-1] == 2400
    assert max(r[4] for r in rows) == mm_to_ticks(2.0)                           # + = wound in


def test_range_checked_before_moving(tmp_path):
    m = FakeServo(4000)
    with pytest.raises(ValueError, match="servo range"):
        _probe(m, tmp_path)
    assert not m.goals and m.closed


def test_simulated_run_writes_a_log(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    _probe(SimulatedMotor(), tmp_path)
    lines = list(csv.DictReader(open(tmp_path / "h.csv")))
    assert len(lines) == 10 and {"cmd_mm", "pos_mm", "current_ma"} <= set(lines[0])
