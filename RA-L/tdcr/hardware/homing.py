"""
Tendon homing by servo current: each tendon's zero is the servo position where it just becomes taut.

The robot is sensor-free; the zero offsets of the 12 tendons are set with the actuators' own current reading. A
tendon is wound in, a small step at a time, until the Dynamixel present current reaches a threshold (a defined small
tension); that position is the tendon's zero, i.e. dl = 0 means "taut at the threshold tension". A 0.2 mm zero error
tilts a module by ~0.8 deg (~8 mm at a four-module tip), so re-home after every re-tensioning or reassembly and report
the repeatability (homing_repeatability).

    motors = {"s1t1": Dynamixel(port, baud, 1), ...}           # lab tendon names, see tdcr/hardware/run_poses.TENDON_ORDER
    zeros = home_all(motors, threshold_ma=60.0)                 # -> data/hardware/zeros/<date>.json

Motors must be in position control (Dynamixel.enable() does that); home_all enables them. `direction` is the sign of
the tick change that winds the tendon in (it depends on how the servo is mounted).
"""
import datetime
import json
import math
import os
import time

import numpy as np

from tdcr.hardware.hysteresis_probe import GOAL_RANGE, PULLEY_R_MM, mm_to_ticks, ticks_to_mm
from tdcr.paths import HARDWARE

ZERO_DIR = os.path.join(HARDWARE, "zeros")


class HomingError(RuntimeError):
    """A tendon never reached the threshold current within the allowed travel (unhooked, broken or far too slack)."""


def home_tendon(motor, threshold_ma, step_mm=0.05, max_travel_mm=15.0, settle_s=0.05, direction=1,
                goal_range=GOAL_RANGE):
    """Wind the tendon in by step_mm at a time until |present current| >= threshold_ma -> the present position (ticks).
    Never moves more than max_travel_mm from the start, nor outside goal_range; on failure the motor is sent back to
    the start and HomingError is raised."""
    start = int(motor.position())
    step = max(1, mm_to_ticks(step_mm))
    n_steps = int(math.floor(mm_to_ticks(max_travel_mm) / step))
    for k in range(1, n_steps + 1):
        g = start + direction * k * step
        if not goal_range[0] <= g <= goal_range[1]:
            motor.goal(start)
            raise HomingError(f"homing would leave the servo range {goal_range[0]}..{goal_range[1]} (start {start}, "
                              f"direction {direction:+d}): re-seat the pulley horn")
        motor.goal(g)
        if settle_s:
            time.sleep(settle_s)
        if abs(motor.current_ma()) >= threshold_ma:
            return int(motor.position())
    motor.goal(start)
    raise HomingError(f"current never reached {threshold_ma} mA within {max_travel_mm} mm of travel")


def homing_repeatability(motor, threshold_ma, n=5, back_off_mm=2.0, **kw):
    """Home n times, paying out back_off_mm before each -> {"zeros": ticks, "mean_ticks", "std_mm"}. kw go to
    home_tendon (step_mm, max_travel_mm, settle_s, direction)."""
    direction = kw.get("direction", 1)
    zeros = []
    for _ in range(n):
        zeros.append(home_tendon(motor, threshold_ma, **kw))
        motor.goal(zeros[-1] - direction * mm_to_ticks(back_off_mm))
        if kw.get("settle_s", 0.05):
            time.sleep(kw.get("settle_s", 0.05))
    z = np.array(zeros, float)
    return {"zeros": [int(v) for v in zeros], "mean_ticks": float(z.mean()), "std_mm": ticks_to_mm(z.std())}


def home_all(motors, threshold_ma, out_dir=ZERO_DIR, directions=None, **kw):
    """Enable and home every motor (dict name -> motor) in turn -> {name: zero ticks}, also written to
    <out_dir>/<date>.json with the threshold and pulley radius. directions: optional {name: +-1}."""
    zeros = {}
    for name, motor in motors.items():
        motor.enable()
        zeros[name] = home_tendon(motor, threshold_ma, direction=(directions or {}).get(name, 1), **kw)
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    record = dict(zeros, threshold_ma=float(threshold_ma), pulley_r_mm=PULLEY_R_MM, time=stamp,
                  directions={k: (directions or {}).get(k, 1) for k in motors})
    with open(os.path.join(out_dir, f"{stamp}.json"), "w") as f:
        json.dump(record, f, indent=1)
    return zeros
