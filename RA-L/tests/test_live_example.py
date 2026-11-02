"""Live twin example (plan Task 12): servo positions -> tendon displacements -> twin, with fake servos."""
import numpy as np

from tdcr.hardware.live_twin import dl_from_ticks, main


def test_example_runs_fake():
    stats = main(["--fake", "--modules", "2", "--mounting", "upright", "--seconds", "1"])
    assert stats["updates"] >= 100
    assert np.all(np.isfinite(stats["last_positions"])) and stats["last_positions"].shape[1] == 3


def test_dl_from_ticks_uses_zero_and_direction():
    from tdcr.hardware.hysteresis_probe import mm_to_ticks
    zeros = {"s1t1": 1000, "s1t2": 2000, "s1t3": 3000}
    dirs = {"s1t1": 1, "s1t2": -1, "s1t3": 1}
    present = {"s1t1": 1000 + mm_to_ticks(2.0), "s1t2": 2000 - mm_to_ticks(1.0), "s1t3": 3000 - mm_to_ticks(0.5)}
    dl = dl_from_ticks(present, zeros, dirs, ["s1t1", "s1t2", "s1t3"])
    from tdcr.hardware.hysteresis_probe import ticks_to_mm
    assert np.allclose(dl, [2.0, 1.0, -0.5], atol=ticks_to_mm(0.5))      # + wound in, - paid out; half a tick
