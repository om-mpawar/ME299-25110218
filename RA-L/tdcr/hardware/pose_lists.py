"""
Command lists for the first hardware sessions (the csv format of hardware/run_poses.py: dl1..dlN in mm, + wound in,
lab tendon order, dl = 0 = every tendon just taut at the homing current).

Pretension p (mm) = the pull of the section-1 tendons; the tendons of section s are pulled p * pretension_pattern[s]
(1 : 1.5 for two modules, 1 : 1.75 : 2.25 : 2.5 for four) so that every tendon carries the same tension in the
straight robot. Equal pulls would leave the distal tendons slack: the proximal tendons already compress the modules
the distal ones run through. The ratio is the number of tendons through each module, so it hardly depends on the
model (tendon stretch, ~0.005 mm for steel wire here, is neglected).

    standup_4s       pretension on all 12 tendons, 0 -> standup_max -> 0 in 0.25 mm steps, robot UPRIGHT: note the
                     first row at which the stack stands by itself (and the row where it falls on the way down)
    calib_2s         ~50 poses, 2 modules: at two pretensions (pretension_mm / 2 and pretension_mm), the same six
                     random targets, each run as an up-and-down pass  straight -> 1/2 -> full -> 1/2 -> straight
                     (the two 1/2 poses are the same command reached winding in and paying out: hysteresis, and
                     the friction direction for tdcr/calibrate.py), mount once upright and once horizontal_x
    heldout_2s       other targets (another seed), at pretension_mm: full -> 1/2 -> straight each
    test_4s_upright  4 modules, six targets as up-and-down passes at pretension_mm (>= the stand-up minimum)
    test_4s_hanging  the same commands hanging (calibrate small, deploy large: never used for calibration)

Bends come from the lab's tendon mapping and are screened with the identified physics core (tools/make_cmd_doe.py:
predicted bend per section, axial load per module); every |dl| <= MAX_DL_MM, the limit
run_poses checks. pretension_mm must be the stand-up minimum plus a margin: run standup_4s first and regenerate.

    .venv/bin/python -m hardware.pose_lists --pretension 2.0 --out hardware/poses
"""
import argparse
import csv
import functools
import json
import os

import numpy as np

from tdcr.paths import ROOT
from tdcr.eb_prior import geometry_from_cables
from tools.make_cmd_doe import HOLES6, lab_offsets, mapped, screen

MAX_DL_MM = 15.0                       # hardware/run_poses.run refuses larger commands
MAX_LOAD = 1.83                        # axial load per module, tension / E (mm^2): the stable 4-section FEM set
R_D = 9.5                              # mapping radius (channel radius, mm)
CABLES = {2: "std6_two_section_sym.cables.json", 4: "std6_four_section_sym.cables.json"}
PASS = ((0.5, "up"), (1.0, "up"), (0.5, "down"), (0.0, "down"))     # one target's up-and-down pass (from straight)


def _cables(n):
    return os.path.join(ROOT, "data", "cables", CABLES[n])


@functools.lru_cache(maxsize=None)
def screen_model(n):
    """The physics core tools/make_cmd_doe.screen uses (identified module, real-robot routing)."""
    from analysis.module_compliance import ModuleModel
    res = json.load(open(os.path.join(ROOT, "results", "mc_grav.json")))
    return ModuleModel(res["knot"]).build(np.array(res["theta"]), n, az=geometry_from_cables(_cables(n))[1],
                                          holes=HOLES6)


def pretension_pattern(n):
    """(3n,) pull per tendon for 1 mm of pretension: equal tensions straight, normalised to 1 on section 1."""
    v = screen_model(n).G @ np.ones(3 * n)
    return v / v[0]


def _row(n, dl, pretension, phase, target=-1, frac=0.0, theta=None, alpha=None):
    return dict(dl=np.asarray(dl, float), pretension=float(pretension), phase=phase, target=int(target),
                frac=float(frac), theta=np.zeros(n) if theta is None else np.asarray(theta, float),
                alpha=np.zeros(n) if alpha is None else np.asarray(alpha, float))


def standup(n, p_max=3.0, step=0.25):
    """Pretension ramp 0 -> p_max -> 0 (rows)."""
    up = np.arange(0.0, p_max + 1e-9, step)
    p = np.concatenate([up, up[-2::-1]])
    return [_row(n, v * pretension_pattern(n), v, "ramp_up" if k < len(up) else "ramp_down") for k, v in enumerate(p)]


def targets(n, k, max_theta, seed, pretension):
    """k random target bends (theta, alpha per section, deg, uniform over the disk of radius max_theta) whose full
    command, at `pretension`, passes the screen -> theta (k, n), alpha (k, n)."""
    rng = np.random.default_rng(seed)
    off = lab_offsets(_cables(n))
    th, al = [], []
    while len(th) < k:
        t = max_theta * np.sqrt(rng.uniform(0, 1, (4 * k, n)))
        a = rng.uniform(0, 360.0, (4 * k, n))
        dl = mapped(t, a, off, R_D) + pretension * pretension_pattern(n)
        keep = screen(dl, _cables(n), n, max_theta * 1.5, MAX_LOAD, False)[0] & (np.abs(dl).max(1) <= MAX_DL_MM)
        for i in np.flatnonzero(keep)[:k - len(th)]:
            th.append(t[i]), al.append(a[i])
    return np.array(th), np.array(al)


def passes(n, theta, alpha, pretension, pattern=PASS):
    """Straight at `pretension`, then every target run through `pattern` (fraction of the bend, phase)."""
    off = lab_offsets(_cables(n))
    pre = pretension * pretension_pattern(n)
    rows = [_row(n, pre, pretension, "straight")]
    for j, (t, a) in enumerate(zip(theta, alpha)):
        for f, ph in pattern:
            dl = mapped((f * t)[None], a[None], off, R_D)[0] + pre
            rows.append(_row(n, dl, pretension, "straight" if f == 0 else ph, j, f, f * t, a))
    return rows


def write(path, n, rows, max_bend):
    """Write the run_poses csv -> meta of the list (predicted bend and axial load of every row, from the screen)."""
    dl = np.array([r["dl"] for r in rows])
    keep, bend, load, slack, its = screen(dl, _cables(n), n, max_bend, MAX_LOAD, False)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "phase", "pretension_mm", "target", "frac"] + [f"theta{s + 1}_deg" for s in range(n)]
                   + [f"alpha{s + 1}_deg" for s in range(n)] + [f"dl{i + 1}" for i in range(3 * n)]
                   + ["pred_bend_deg", "pred_load", "pred_slack"])
        for i, r in enumerate(rows):
            w.writerow([i, r["phase"], f"{r['pretension']:.3f}", r["target"], f"{r['frac']:.2f}"]
                       + [f"{v:.3f}" for v in r["theta"]] + [f"{v:.3f}" for v in r["alpha"]]
                       + [f"{v:.6f}" for v in r["dl"]] + [f"{bend[i]:.2f}", f"{load[i]:.4f}", int(slack[i])])
    return dict(csv=path, n=n, rows=len(rows), pretensions=sorted({r["pretension"] for r in rows}),
                max_bend_deg=float(max_bend), pred_bend_max_deg=float(bend.max()), pred_load_max=float(load.max()),
                all_pass_screen=bool(keep.all()), dl_range_mm=[float(dl.min()), float(dl.max())])


def make_all(out, pretension_mm=2.0, seed=0, standup_max=5.0, theta_2s=35.0, theta_4s=25.0, k=6):
    """Write every list to `out` (plus manifest.json) -> {name: meta}."""
    os.makedirs(out, exist_ok=True)
    p, made = pretension_mm, {}
    made["standup_4s"] = write(os.path.join(out, "standup_4s.csv"), 4, standup(4, standup_max), 0.5)
    th, al = targets(2, k, theta_2s, seed, p)
    rows = passes(2, th, al, 0.5 * p) + passes(2, th, al, p)
    made["calib_2s"] = write(os.path.join(out, "calib_2s.csv"), 2, rows, theta_2s * 1.5)
    th, al = targets(2, 2 * k, theta_2s, seed + 1000, p)
    rows = passes(2, th, al, p, pattern=((1.0, "up"), (0.5, "down"), (0.0, "down")))
    made["heldout_2s"] = write(os.path.join(out, "heldout_2s.csv"), 2, rows, theta_2s * 1.5)
    th, al = targets(4, k, theta_4s, seed + 2000, p)
    rows = passes(4, th, al, p)
    for m in ("upright", "hanging"):
        made[f"test_4s_{m}"] = write(os.path.join(out, f"test_4s_{m}.csv"), 4, rows, theta_4s * 1.5)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(dict(pretension_mm=p, seed=seed, standup_max_mm=standup_max, theta_2s_deg=theta_2s,
                       theta_4s_deg=theta_4s, mapping="tools/make_cmd_doe.mapped (lab, pure pay-out) + pretension",
                       order=["standup_4s", "calib_2s (upright, then horizontal_x)", "heldout_2s",
                              "test_4s_upright", "test_4s_hanging"],
                       lists={k_: {**v, "csv": os.path.basename(v["csv"])} for k_, v in made.items()}), f, indent=1)
    return made


def main(argv=None):                                                    # pragma: no cover (CLI)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretension", type=float, required=True, help="common pull (mm): stand-up minimum + margin")
    ap.add_argument("--standup-max", type=float, default=5.0, help="top of the stand-up ramp, section-1 pull (mm)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "hardware", "poses"))
    a = ap.parse_args(argv)
    for name, m in make_all(a.out, a.pretension, a.seed, a.standup_max).items():
        print(f"{name:16s} {m['rows']:3d} rows, n={m['n']}, pretension {m['pretensions']} mm, dl "
              f"{m['dl_range_mm'][0]:.1f}..{m['dl_range_mm'][1]:.1f} mm, predicted bend <= {m['pred_bend_max_deg']:.0f}"
              f" deg, module load <= {m['pred_load_max']:.2f} mm^2")


if __name__ == "__main__":
    main()
