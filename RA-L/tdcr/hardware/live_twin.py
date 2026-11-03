"""
Live digital twin: read the servos, compute the backbone shape, as fast as the bus allows.

    # on the robot (after homing, tdcr/hardware/homing.py, and calibration, tdcr/calibrate.py)
    .venv/bin/python -m tdcr.hardware.live_twin --zeros data/hardware/zeros/<date>.json --calib results/calib_<s>.json \
        --modules 4 --mounting hanging --port /dev/ttyUSB0
    # without hardware
    .venv/bin/python -m tdcr.hardware.live_twin --fake --modules 2 --mounting upright --seconds 5

Each update reads every servo's present position, converts it to the tendon displacement dl = direction x (present -
zero) in mm (+ wound in), and calls Twin.shape(dl). It prints the tip pose and the update rate twice a second.
In your own code use tdcr.twin.Twin directly; this script is the minimal loop around it.
"""
import argparse
import math
import time

import numpy as np

from tdcr.hardware.hysteresis_probe import mm_to_ticks
from tdcr.hardware.run_poses import TENDON_ORDER, dl_from_ticks, open_motors, read_zeros
from tdcr.twin import MOUNTINGS, Twin


class FakeServo:
    """A servo whose position follows a slow sinusoid (a few mm of tendon travel): exercises the loop without
    hardware."""

    def __init__(self, k, zero=2048):
        self.k, self.zero, self.t0 = k, zero, time.perf_counter()

    def position(self):
        t = time.perf_counter() - self.t0
        return self.zero + mm_to_ticks(1.5 * math.sin(0.5 * t + 2.1 * self.k))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modules", type=int, required=True)
    ap.add_argument("--mounting", default="upright", help=f"one of {sorted(MOUNTINGS)}")
    ap.add_argument("--zeros", help="homing zero file (tdcr/hardware/homing.py); not needed with --fake")
    ap.add_argument("--calib", help="calibration file (tdcr/calibrate.py); default: the SOFA-identified module")
    ap.add_argument("--gap-mm", type=float, default=2.0, help="joint gap without a calibration (as the bench CLIs)")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--ids", type=int, nargs="+", help="servo ids in lab tendon order (default 1..3n)")
    ap.add_argument("--fake", action="store_true", help="fake servos (no hardware)")
    ap.add_argument("--seconds", type=float, default=float("inf"), help="stop after this long (default: Ctrl-C)")
    a = ap.parse_args(argv)
    names = TENDON_ORDER[:3 * a.modules]
    if a.fake:
        motors = {n: FakeServo(k) for k, n in enumerate(names)}
        zeros, directions = {n: 2048 for n in names}, {}
    else:                                                               # pragma: no cover (hardware)
        zeros, directions = read_zeros(a.zeros, names)
        motors = open_motors(a.port, a.baud, a.ids or list(range(1, len(names) + 1)), names)
    tw = Twin(a.modules, a.mounting, calib_json=a.calib, gap_mm=a.gap_mm)
    t_start = t_print = time.perf_counter()
    updates, unconverged, solve_ms, s = 0, 0, [], None
    try:
        while time.perf_counter() - t_start < a.seconds:
            dl = dl_from_ticks({n: m.position() for n, m in motors.items()}, zeros, directions, names)
            s = tw.shape(dl)
            updates += 1
            unconverged += not s.converged
            solve_ms.append(s.solve_ms)
            if time.perf_counter() - t_print > 0.5:
                t_print = time.perf_counter()
                rate = updates / (t_print - t_start)
                tip = s.positions[-1]
                print(f"tip [{tip[0]:7.2f} {tip[1]:7.2f} {tip[2]:7.2f}] mm | {rate:5.0f} Hz | solve "
                      f"{np.median(solve_ms[-200:]):.2f} ms | {'ok' if s.converged else 'NOT CONVERGED'}", flush=True)
    except KeyboardInterrupt:                                           # pragma: no cover (interactive)
        pass
    elapsed = time.perf_counter() - t_start
    return dict(updates=updates, unconverged=unconverged, rate_hz=updates / elapsed, median_solve_ms=float(np.median(solve_ms)),
                last_positions=s.positions if s is not None else None)


if __name__ == "__main__":
    main()
