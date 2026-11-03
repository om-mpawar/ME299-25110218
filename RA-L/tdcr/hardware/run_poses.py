"""
Pose runs on the robot: drive the homed tendons through a command list and log every hold.

Each row of the command csv (columns dl1..dlN, mm, + wound in, - paid out, lab tendon order; e.g. the designs from
tdcr/doe.py) is sent as goal = zero + direction x mm_to_ticks(dl) to every tendon's servo (zero and direction from the
homing file, tdcr/hardware/homing.py). The run waits settle_s for the robot to come to rest, then holds for hold_s (the
window the motion capture averages over, tdcr/hardware/mocap.py) and logs, per tendon, the hold's wall-clock start and
end, goal and present ticks, present current, and the commanded and reached displacement (dl_cmd_mm, dl_meas_mm =
direction x (present - zero); the model's input is the reached one).

    .venv/bin/python -m tdcr.hardware.run_poses --commands results/configs_std6_2s_train.csv \
        --zeros data/hardware/zeros/<date>.json --port /dev/ttyUSB0 --out data/hw/<session>

All commands are checked before anything moves: width against the motors, |dl| <= max_dl_mm, and every goal inside
the servo's range (GOAL_RANGE, one pulley turn in position mode 3: the homed zero must leave room for the list's
largest pull and pay-out). If the run stops on an error or Ctrl-C, every servo is sent back to its zero. Clocks: the log
uses time.time() of this PC; the motion-capture PC's clock must be synchronised to it (or offset measured) for mocap.py.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from tdcr.hardware.hysteresis_probe import GOAL_RANGE, Dynamixel, mm_to_ticks, ticks_to_mm

TENDON_ORDER = [f"s{s}t{k}" for s in range(1, 5) for k in range(1, 4)]       # lab order, as the cable files
LOG_COLUMNS = ["row", "rep", "t_start", "t_end", "tendon", "goal_ticks", "present_ticks", "current_ma", "dl_cmd_mm",
               "dl_meas_mm"]


def read_commands(path):
    """-> (P, N) tendon displacements (mm) from the dl1..dlN columns of a design csv."""
    with open(path) as f:
        rd = csv.DictReader(f)
        cols = sorted((c for c in rd.fieldnames if c.startswith("dl") and c[2:].isdigit()), key=lambda c: int(c[2:]))
        return np.array([[float(r[c]) for c in cols] for r in rd])


def read_zeros(path, names):
    """Zero ticks and winding directions (+1: ticks increase winding in; -1: they decrease) of the named tendons from a
    tdcr/hardware/homing.py zero file -> ({name: zero}, {name: direction}); a file without directions means +1."""
    z = json.load(open(path))
    missing = [n for n in names if n not in z]
    if missing:
        raise ValueError(f"zero file {path} has no zero for {missing}")
    dirs = {n: int(z.get("directions", {}).get(n, 1)) for n in names}
    bad = {n: d for n, d in dirs.items() if d not in (1, -1)}
    if bad:
        raise ValueError(f"zero file {path}: directions must be +1 or -1, got {bad}")
    return {n: int(z[n]) for n in names}, dirs


def goal_ticks(zero, direction, dl_mm):
    """Servo goal (ticks) for a tendon displacement (mm, + wound in)."""
    return zero + direction * mm_to_ticks(dl_mm)


def dl_from_ticks(present, zeros, directions, names):
    """Present ticks -> tendon displacements (mm, + wound in) in the order of names."""
    return np.array([directions.get(n, 1) * ticks_to_mm(present[n] - zeros[n]) for n in names])


def run(commands_csv, zeros_json, motors, out_dir, settle_s=3.0, repeat=1, hold_s=1.0, max_dl_mm=15.0,
        goal_range=GOAL_RANGE):
    """Drive every command row (repeated `repeat` times) and log each hold -> path of <out_dir>/servo_log.csv.
    motors: {tendon name: motor} for the first 3n names of TENDON_ORDER. On any error or Ctrl-C every servo is sent
    back to its zero and the error is raised again."""
    dl = read_commands(commands_csv)
    names = TENDON_ORDER[:dl.shape[1]]
    if dl.shape[1] != len(motors) or set(names) != set(motors):
        raise ValueError(f"the commands have {dl.shape[1]} tendon columns but {len(motors)} motors were given "
                         f"({sorted(motors)})")
    if np.abs(dl).max() > max_dl_mm:
        bad = int(np.argmax(np.abs(dl).max(1)))
        raise ValueError(f"row {bad} asks for {np.abs(dl[bad]).max():.2f} mm, beyond the {max_dl_mm:g} mm limit")
    zeros, dirs = read_zeros(zeros_json, names)
    goals = np.array([[goal_ticks(zeros[n], dirs[n], v) for n, v in zip(names, row)] for row in dl])
    lo, hi = goal_range
    out = (goals < lo) | (goals > hi)
    if out.any():
        msg = []
        for k in np.flatnonzero(out.any(0)):
            n, need = names[k], dirs[names[k]] * np.array([mm_to_ticks(v) for v in dl[:, k]])
            msg.append(f"{n} (zero {zeros[n]}, direction {dirs[n]:+d}): goals {goals[:, k].min()}..{goals[:, k].max()}, "
                       f"zero must be in {lo - need.min()}..{hi - need.max()}")
        raise ValueError(f"goals outside the servo range {lo}..{hi}; nothing moved. Re-seat the pulley horn or re-home: "
                         + "; ".join(msg))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "servo_log.csv")
    try:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(LOG_COLUMNS)
            for rep in range(repeat):
                for i, row in enumerate(dl):
                    for k, n in enumerate(names):
                        motors[n].goal(int(goals[i, k]))
                    if settle_s:
                        time.sleep(settle_s)
                    t_start = time.time()
                    if hold_s:
                        time.sleep(hold_s)
                    reads = {n: (int(motors[n].position()), float(motors[n].current_ma())) for n in names}
                    t_end = time.time()
                    meas = dl_from_ticks({n: r[0] for n, r in reads.items()}, zeros, dirs, names)
                    for k, (n, v) in enumerate(zip(names, row)):
                        w.writerow([i, rep, f"{t_start:.4f}", f"{t_end:.4f}", n, int(goals[i, k]), reads[n][0],
                                    f"{reads[n][1]:.1f}", f"{v:.6f}", f"{meas[k]:.6f}"])
                    f.flush()
    except BaseException:
        for n in names:
            try:
                motors[n].goal(zeros[n])
            except Exception:
                pass
        print(f"run_poses: aborted, every servo sent back to its zero (log so far: {path})")
        raise
    return path


class SharedBusDynamixel(Dynamixel):
    """A Dynamixel on a port shared with the other servos (the base class opens one port per motor, which fails
    for twelve servos on one bus)."""

    def __init__(self, port_handler, packet_handler, motor_id):          # pragma: no cover (hardware)
        self.ph, self.pk, self.id = port_handler, packet_handler, motor_id


def open_motors(port, baud, ids, names):                                # pragma: no cover (hardware)
    """One serial port for all servos -> {name: motor}, enabled in position control."""
    from dynamixel_sdk import PacketHandler, PortHandler
    ph, pk = PortHandler(port), PacketHandler(2.0)
    if not ph.openPort() or not ph.setBaudRate(baud):
        raise RuntimeError(f"could not open {port} at {baud}")
    motors = {n: SharedBusDynamixel(ph, pk, i) for n, i in zip(names, ids)}
    for m in motors.values():
        m.enable()
    return motors


def main(argv=None):                                                    # pragma: no cover (hardware)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--commands", required=True)
    ap.add_argument("--zeros", required=True)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--ids", type=int, nargs="+", help="servo ids in lab tendon order (default 1..N)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args(argv)
    n = read_commands(a.commands).shape[1]
    names = TENDON_ORDER[:n]
    motors = open_motors(a.port, a.baud, a.ids or list(range(1, n + 1)), names)
    print(run(a.commands, a.zeros, motors, a.out, settle_s=a.settle, repeat=a.repeat, hold_s=a.hold))


if __name__ == "__main__":
    main()
