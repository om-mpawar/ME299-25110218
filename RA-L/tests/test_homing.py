"""Servo-current homing of the tendons (plan Task 4), against a fake motor: slack until a hidden taut point, then the
current rises steeply with further winding."""
import json

import numpy as np
import pytest

from tdcr.hardware.homing import HomingError, home_all, home_tendon, homing_repeatability
from tdcr.hardware.hysteresis_probe import mm_to_ticks, ticks_to_mm


class FakeTendonMotor:
    """Position-controlled servo on a tendon: no load until `taut_ticks` (in the winding direction), then the current
    rises by `ma_per_tick` per tick of further winding. Optional position read noise (ticks)."""

    def __init__(self, start=2000, taut_ticks=2500, ma_per_tick=60.0, noise_ticks=0, seed=0):
        self.pos, self.taut, self.k = start, taut_ticks, ma_per_tick
        self.noise, self.rng = noise_ticks, np.random.default_rng(seed)
        self.goals, self.enabled = [], False

    def enable(self, profile_velocity=30):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def goal(self, ticks):
        self.pos = int(ticks); self.goals.append(int(ticks))

    def position(self):
        return self.pos + (int(self.rng.integers(-self.noise, self.noise + 1)) if self.noise else 0)

    def current_ma(self):
        return 3.0 + max(0, self.pos - self.taut) * self.k     # 3 mA idle current below the taut point


def test_finds_taut_point():
    m = FakeTendonMotor(start=2000, taut_ticks=2500)
    zero = home_tendon(m, threshold_ma=50.0, settle_s=0.0)
    assert abs(zero - m.taut) <= mm_to_ticks(0.05)


def test_raises_if_never_taut():
    m = FakeTendonMotor(start=2000, taut_ticks=2000 + mm_to_ticks(20.0))    # beyond the 15 mm travel limit
    with pytest.raises(HomingError):
        home_tendon(m, threshold_ma=50.0, max_travel_mm=15.0, settle_s=0.0)
    assert max(m.goals) - 2000 <= mm_to_ticks(15.0)                          # never past the travel limit
    assert m.pos == 2000                                                     # left where it started


def test_repeatability_reports_std():
    m = FakeTendonMotor(start=2000, taut_ticks=2400, noise_ticks=1)
    r = homing_repeatability(m, threshold_ma=50.0, n=5, back_off_mm=2.0, settle_s=0.0)
    assert r["std_mm"] < 0.05
    assert abs(r["mean_ticks"] - 2400) <= mm_to_ticks(0.05)


def test_writes_zero_file(tmp_path):
    motors = {f"s1t{k}": FakeTendonMotor(start=1000 * k, taut_ticks=1000 * k + 300) for k in (1, 2, 3)}
    zeros = home_all(motors, threshold_ma=50.0, out_dir=tmp_path, settle_s=0.0)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    saved = json.load(open(files[0]))
    assert set(motors) <= set(saved) and saved["threshold_ma"] == 50.0 and saved["pulley_r_mm"] == 5.0
    assert all(saved[k] == zeros[k] for k in motors)
    assert all(m.enabled for m in motors.values())


def test_direction_minus_winds_down():
    m = FakeTendonMotor(start=2000, taut_ticks=2400)
    m.current_ma = lambda: 3.0 + max(0, 1600 - m.pos) * 60.0                # this servo winds in with decreasing ticks
    zero = home_tendon(m, threshold_ma=50.0, settle_s=0.0, direction=-1)
    assert abs(zero - 1600) <= mm_to_ticks(0.05) and ticks_to_mm(abs(zero - 1600)) <= 0.05


def test_homing_never_leaves_the_servo_range():
    """A start near the end of the turn: stop at the range limit and go back to the start (code review H2)."""
    m = FakeTendonMotor(start=4050, taut_ticks=4500)
    with pytest.raises(HomingError, match="servo range"):
        home_tendon(m, threshold_ma=50.0, settle_s=0.0)
    assert max(m.goals) <= 4095 and m.goals[-1] == 4050
