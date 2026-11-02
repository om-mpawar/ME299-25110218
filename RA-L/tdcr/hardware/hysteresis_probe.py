"""
Measure tendon hysteresis on the real robot, with no motion capture.

Why this is worth doing before mocap
------------------------------------
Every estimator in this project so far is a STATIC map: six motor readings
in, one pose out. That is a function, so it returns one output per input.

Tendon friction makes the real robot path-dependent: the same six motor
readings give a different tip pose depending on whether you arrived from
above or below. A static map cannot represent that, so its accuracy has a
FLOOR roughly equal to the hysteresis width, no matter how much mocap data
is eventually collected.

That floor is the number this script measures, and it decides a research
direction:

  * If hysteresis is small, the static GP / closed-form models are the right
    answer and the filter work is not worth doing.
  * If hysteresis is large, a state-based estimator (a UKF carrying a
    friction state) can go below a floor no regressor can reach -- which is
    a genuine contribution, and one that no amount of FEM can demonstrate
    because a frictionless simulation has zero hysteresis by construction.

Crucially this needs NO external sensor. Commanding a motor position and
reading back the achieved position and current measures the actuation-side
friction directly, which is the dominant source. A loading/unloading ramp
traces the loop.

What it does
------------
Slowly ramps one tendon 0 -> max -> 0 for several cycles, logging commanded
position, present position, and present current at every step. Then reports
the loop width -- the gap between the ascending and descending branches at
the same command.

    python3 hardware/hysteresis_probe.py --sim                       # no hardware
    python3 hardware/hysteresis_probe.py --port /dev/ttyUSB0 --ids 1 --max-pull 5
    python3 hardware/hysteresis_probe.py --analyse results/hysteresis_id1.csv

SAFETY. This moves the robot. Read before running:
  * --max-pull is in MILLIMETRES of tendon travel and defaults to a
    deliberately small 5 mm. Raise it only once you have watched a run.
  * The motor stops and unwinds if present current exceeds --stall-current,
    which catches a jammed tendon or a mechanical end stop.
  * Torque is always disabled and the tendon returned to zero on exit, even
    on Ctrl-C or an exception.
  * It refuses to start without an explicit confirmation unless --yes.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tdcr`
import argparse
import csv
import math
import os
import sys
import time

import numpy as np

from tdcr.paths import ROOT as _DIR  # paths below are relative to the repo root
RESULTS_DIR = os.path.join(_DIR, "results")

# ── Actuator geometry ─────────────────────────────────────────────────────
# Pulley radius converts motor angle to tendon length. PULLEY_DIA_MM = 10 in
# cylinder_scene.py, so r = 5 mm and 1 rad of motor rotation pays out 5 mm.
PULLEY_R_MM = 5.0
TICKS_PER_REV = 4096          # XM/XL series, 0..4095 over 360 deg

# ── Dynamixel control table (Protocol 2.0, XM430 / XL430 family) ──────────
# CHECK THESE AGAINST YOUR MODEL. They differ between series, and writing a
# goal position to the wrong address on the wrong model does something
# unpredictable rather than failing cleanly.
ADDR = {
    "torque_enable":    (64, 1),
    "goal_position":    (116, 4),
    "present_position": (132, 4),
    "present_current":  (126, 2),
    "profile_velocity": (112, 4),
    "operating_mode":   (11, 1),
}
CURRENT_UNIT_MA = 2.69        # XM430: 2.69 mA per unit


def mm_to_ticks(mm):
    """Tendon millimetres -> motor ticks, via the pulley."""
    return int(round(mm / PULLEY_R_MM * TICKS_PER_REV / (2.0 * math.pi)))


def ticks_to_mm(ticks):
    return float(ticks) * 2.0 * math.pi / TICKS_PER_REV * PULLEY_R_MM


# ── Hardware backend ──────────────────────────────────────────────────────

class Dynamixel:
    """Thin wrapper over dynamixel_sdk. Only what this probe needs."""

    def __init__(self, port, baud, motor_id, protocol=2.0):
        from dynamixel_sdk import PortHandler, PacketHandler
        self.ph = PortHandler(port)
        self.pk = PacketHandler(protocol)
        if not self.ph.openPort():
            raise RuntimeError(f"could not open {port}")
        if not self.ph.setBaudRate(baud):
            raise RuntimeError(f"could not set baud {baud}")
        self.id = motor_id

    def _write(self, name, value):
        addr, size = ADDR[name]
        fn = {1: self.pk.write1ByteTxRx, 2: self.pk.write2ByteTxRx,
              4: self.pk.write4ByteTxRx}[size]
        rc, err = fn(self.ph, self.id, addr, int(value))
        if rc != 0 or err != 0:
            raise RuntimeError(f"write {name}: comm={rc} err={err}")

    def _read(self, name):
        addr, size = ADDR[name]
        fn = {1: self.pk.read1ByteTxRx, 2: self.pk.read2ByteTxRx,
              4: self.pk.read4ByteTxRx}[size]
        val, rc, err = fn(self.ph, self.id, addr)
        if rc != 0 or err != 0:
            raise RuntimeError(f"read {name}: comm={rc} err={err}")
        return val

    def enable(self, profile_velocity=30):
        self._write("torque_enable", 0)          # must be off to change mode
        self._write("operating_mode", 3)         # position control
        self._write("profile_velocity", profile_velocity)
        self._write("torque_enable", 1)

    def disable(self):
        try:
            self._write("torque_enable", 0)
        except Exception:
            pass

    def goal(self, ticks):
        self._write("goal_position", ticks)

    def position(self):
        return self._read("present_position")

    def current_ma(self):
        raw = self._read("present_current")
        if raw > 0x7FFF:                          # signed 16-bit
            raw -= 0x10000
        return raw * CURRENT_UNIT_MA

    def close(self):
        self.disable()
        self.ph.closePort()


class SimulatedMotor:
    """Stand-in with a deliberately simple Dahl-style friction state.

    Exists so the ramp logic, the logging and the analysis can all be
    exercised without touching the robot. The numbers it produces are
    FABRICATED -- its only job is to make the loop non-degenerate so the
    analysis has something to chew on.
    """

    def __init__(self, width_mm=1.2, seed=0):
        self.rng = np.random.default_rng(seed)
        self.width = mm_to_ticks(width_mm)
        self.pos = 0.0
        self.last_goal = 0.0

    def enable(self, profile_velocity=30): pass
    def disable(self): pass
    def close(self): pass

    def goal(self, ticks):
        # Lag the achieved position behind the command by up to half the loop
        # width, in whichever direction the command is moving.
        direction = np.sign(ticks - self.last_goal)
        target = ticks - direction * self.width / 2.0
        self.pos += 0.6 * (target - self.pos)
        self.last_goal = ticks

    def position(self):
        return int(self.pos + self.rng.normal(0, 1.5))

    def current_ma(self):
        return abs(self.pos) * 0.09 + self.rng.normal(0, 4.0)


# ── The ramp ──────────────────────────────────────────────────────────────

def run_probe(motor, max_pull_mm, cycles, steps, dwell, stall_ma, out_csv,
              label=""):
    """Ramp 0 -> max -> 0 for `cycles` cycles, logging every step."""
    peak = mm_to_ticks(max_pull_mm)
    rows = []
    t0 = time.time()

    try:
        motor.enable()
        for cyc in range(cycles):
            up = list(np.linspace(0, peak, steps))
            down = list(np.linspace(peak, 0, steps))
            for branch, seq in (("up", up), ("down", down)):
                for cmd in seq:
                    motor.goal(int(round(cmd)))
                    time.sleep(dwell)
                    pos = motor.position()
                    cur = motor.current_ma()

                    if abs(cur) > stall_ma:
                        print(f"\nSTALL: {cur:.0f} mA exceeds {stall_ma} mA. "
                              "Unwinding and stopping.")
                        motor.goal(0)
                        time.sleep(0.5)
                        raise SystemExit(1)

                    rows.append([cyc, branch, float(cmd), ticks_to_mm(cmd),
                                 pos, ticks_to_mm(pos), cur, time.time() - t0])
                print(f"  cycle {cyc+1}/{cycles} {branch:4s}  "
                      f"cmd {ticks_to_mm(cmd):5.2f} mm  "
                      f"pos {ticks_to_mm(pos):5.2f} mm  {cur:7.1f} mA")
    finally:
        try:
            motor.goal(0)
            time.sleep(0.6)
        except Exception:
            pass
        motor.close()

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "branch", "cmd_ticks", "cmd_mm", "pos_ticks",
                    "pos_mm", "current_ma", "t_s"])
        w.writerows(rows)
    print(f"\nwrote {len(rows)} samples to {out_csv}")
    return rows


# ── Analysis ──────────────────────────────────────────────────────────────

def analyse(path, n_bins=20):
    """Report the hysteresis loop width, in tendon millimetres.

    The width at a given command is the gap between the achieved position on
    the ascending branch and on the descending branch. That gap IS the error
    floor of any static model: two different states, one input.
    """
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path} is empty")

    cmd = np.array([float(r["cmd_mm"]) for r in rows])
    pos = np.array([float(r["pos_mm"]) for r in rows])
    cur = np.array([float(r["current_ma"]) for r in rows])
    branch = np.array([r["branch"] for r in rows])

    lo, hi = cmd.min(), cmd.max()
    edges = np.linspace(lo, hi, n_bins + 1)
    widths, centres, dcur = [], [], []

    for i in range(n_bins):
        m = (cmd >= edges[i]) & (cmd < edges[i + 1])
        u, d = m & (branch == "up"), m & (branch == "down")
        if u.sum() < 2 or d.sum() < 2:
            continue
        centres.append(0.5 * (edges[i] + edges[i + 1]))
        widths.append(float(pos[d].mean() - pos[u].mean()))
        dcur.append(float(cur[u].mean() - cur[d].mean()))

    widths = np.array(widths); centres = np.array(centres); dcur = np.array(dcur)

    print(f"Hysteresis loop from {os.path.basename(path)}")
    print(f"  {len(rows)} samples, command range {lo:.2f} .. {hi:.2f} mm\n")
    print(f"  {'command mm':>11} {'loop width mm':>14} {'current gap mA':>15}")
    for c, w, dc in zip(centres, widths, dcur):
        print(f"  {c:11.2f} {w:14.3f} {dc:15.1f}")

    peak = float(np.max(np.abs(widths))) if len(widths) else 0.0
    print(f"\n  peak loop width      : {peak:.3f} mm of tendon travel")
    print(f"  mean |loop width|    : {np.mean(np.abs(widths)):.3f} mm")

    # Translate the actuation-side loop width into a tip-position floor by
    # pushing it through the ACTUAL kinematics rather than a back-of-envelope
    # formula: perturb one tendon by the loop width at a representative
    # mid-range bend and measure how far the tip moves. This is a headline
    # number, so it should come from the model the rest of the project uses.
    from tdcr.multisection import theta_alpha_to_deltas, theta_alpha_to_pose6, SECTION_LENGTHS
    from tdcr.digitaltuning import THETA_MAX

    th = [THETA_MAX / 2.0, THETA_MAX / 2.0]
    al = [0.0, 0.0]
    dl0 = np.asarray(theta_alpha_to_deltas(th, al), dtype=float)
    p_ref = np.asarray(theta_alpha_to_pose6(th, al))[:3]

    from tdcr.multisection import deltas_to_sections
    dl1 = dl0.copy()
    dl1[0] += peak                                  # one tendon, one loop width
    secs = deltas_to_sections(dl1, SECTION_LENGTHS)
    from tdcr.multisection import tip_position
    p_pert = np.asarray(tip_position(secs))
    tip_floor = float(np.linalg.norm(p_pert - p_ref))

    print(f"\n  implied tip-position floor for ANY static model: ~{tip_floor:.1f} mm")
    print( "  (loop width applied to one tendon at half-range bend, propagated")
    print( "   through the two-section kinematics -- not a hand formula.)")
    if tip_floor > 2.0:
        print("\n  -> Larger than the GP's current held-out error. A static map")
        print("     cannot reach this accuracy on hardware however much data")
        print("     it gets. The state-based estimator is worth building.")
    else:
        print("\n  -> Small. The static models are probably the right answer;")
        print("     do not spend the time on a filter.")
    return centres, widths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyse", metavar="CSV", default=None,
                    help="analyse an existing log instead of running the robot")
    ap.add_argument("--sim", action="store_true",
                    help="simulated motor, no hardware, FABRICATED numbers")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--ids", type=int, nargs="+", default=[1],
                    help="motor id(s); each is probed in turn")
    ap.add_argument("--max-pull", type=float, default=5.0,
                    help="MILLIMETRES of tendon travel (default deliberately small)")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--steps", type=int, default=25, help="steps per branch")
    ap.add_argument("--dwell", type=float, default=0.35,
                    help="seconds to settle at each step -- keep it slow, this")
    ap.add_argument("--stall-current", type=float, default=900.0, help="mA")
    ap.add_argument("--out", default=None)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation")
    args = ap.parse_args()

    if args.analyse:
        analyse(args.analyse)
        sys.exit(0)

    for mid in args.ids:
        out = args.out or os.path.join(RESULTS_DIR, f"hysteresis_id{mid}.csv")
        print(f"\n=== tendon on motor id {mid} ===")
        print(f"  travel {args.max_pull} mm, {args.cycles} cycles, "
              f"{2*args.steps*args.cycles} points, "
              f"~{2*args.steps*args.cycles*args.dwell/60:.1f} min")

        if args.sim:
            motor = SimulatedMotor()
            print("  SIMULATED MOTOR -- numbers are fabricated")
        else:
            if not args.yes:
                resp = input(f"  This will MOVE the robot on {args.port}. "
                             "Type 'yes' to continue: ")
                if resp.strip().lower() != "yes":
                    print("  aborted")
                    sys.exit(0)
            motor = Dynamixel(args.port, args.baud, mid)

        run_probe(motor, args.max_pull, args.cycles, args.steps, args.dwell,
                  args.stall_current, out)
        analyse(out)
