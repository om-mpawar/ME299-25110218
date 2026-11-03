"""
Regenerate every simulation-derived number and figure of paper/main.tex, then check them against the paper.

    .venv/bin/python scripts/reproduce.py              # fast mode (default): everything but the multi-hour jobs, ~5 min
    .venv/bin/python scripts/reproduce.py --full       # also n_transfer (~2 h, 8 threads) and the PINN baseline (hours)
    .venv/bin/python scripts/reproduce.py --list       # the steps, their commands and what they write
    .venv/bin/python scripts/reproduce.py --only runtime claims     # selected steps

Every step is an ordinary command (printed by --list) that can also be run by hand; its output goes to
results/logs/<step>.log. Producers write their result files in place under results/, so `git diff results/` shows what
moved against the committed files. Figures go to results/figures/ and are compared pixel by pixel with
paper/figures/ (needs pdftoppm; the paper's own figure files are never overwritten). The last step,
scripts/claims.py, compares each number printed in paper/main.tex with the regenerated value and writes
results/claims.json and results/CLAIMS.md.

In fast mode the long jobs are not re-run, and their claims are checked against the stored result files:
results/n_transfer.json (Table I learned column, Fig. 2, coverage; a full run of the packaged code, 2 Nov 2026) and
results/pinn/*.json (Table II; produced by the old repository at d363e90). The fast mode needs no SOFA, no hardware
and no network; exact_global3 needs the `pinn` extra (torch) and is skipped with a note without it.

Determinism: module_compliance runs with OMP_NUM_THREADS=1 (the BLAS thread count changes the last digits of
mc_std6.json, 1e-10); timings (runtime, study_guide, exact_global3's fit time) depend on the machine and its load, so
the steps run one at a time and the computer should otherwise be idle.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
TRAIN = ["std6_1s_sym_train_g0", "std6_2s_sym_train_g0", "std6_1s_sym_train", "std6_2s_sym_train"]


def step(name, cmd, writes, env=None, long=False, needs=None, about=""):
    return dict(name=name, cmd=cmd, writes=writes, env=env or {}, long=long, needs=needs, about=about)


STEPS = [
    step("torsion_identify", [PY, "scripts/torsion_identify.py"], ["results/torsion_std6.json"],
         about="torsion factor x2.45 of plane section, from the SOFA twist test (results/twist_test_std6.json)"),
    step("module_compliance", [PY, "scripts/module_compliance.py", "--train", *TRAIN, "--torsion-json",
                               "results/torsion_std6.json", "--out", "results/mc_std6.json"],
         ["results/mc_std6.json"], env=dict(OMP_NUM_THREADS="1"),
         about="Table I: identified physics, plane-section column; x20.2/5.9/5.3 (~2 min)"),
    step("mc_reeval", [PY, "scripts/mc_reeval.py"], ["results/mc_std6_reeval.json"],
         about="Table I re-evaluated after the convergence fix (comment under Table I)"),
    step("tol_reeval", [PY, "scripts/tol_reeval.py"], ["results/tol_reeval.json"],
         about="Table I at a tight gravity tolerance (physics audit R2)"),
    step("n_transfer", [PY, "scripts/n_transfer.py", "--mc", "results/mc_std6.json"], ["results/n_transfer.json"],
         env=dict(OMP_NUM_THREADS="8"), long=True,
         about="Table I learned column, Fig. 2 points, residual variants, coverage (~2 h)"),
    step("pinn_stage1", ["bash", "scripts/pinn_stage1.sh"], ["results/pinn/2s_*.json"], long=True, needs="torch",
         about="Table II PINN rows, 2-module training (hours)"),
    step("pinn_stage2", ["bash", "scripts/pinn_stage2.sh"], ["results/pinn/4s_*.json"], long=True, needs="torch",
         about="Table II four-module retraining (hours)"),
    step("exact_global3", [PY, "scripts/exact_global3.py"],
         ["results/pinn/exact_global3_2s.json", "results/pinn/pinn_laws_exact.json"],
         env=dict(OMP_NUM_THREADS="2", NUMBA_NUM_THREADS="2"), needs="torch",
         about="Table II first and last rows; x20.2/5.8/5.3 in 80 s (~3 min)"),
    step("physics_props", [PY, "scripts/physics_props.py"], ["results/physics_props.json"],
         about="Prop. 1 example, Prop. 3 ratios, tendon stretch, Remark 1 hysteresis, sag, mounting error"),
    step("study_guide", [PY, "scripts/study_guide_figs.py"], ["results/study_guide/numbers.json"],
         about="Prop. 1 (4.5 mm), Prop. 2 Jacobian (2000x), run-time medians and p95"),
    step("runtime", [PY, "scripts/runtime.py"], ["results/runtime.json"],
         about="slowest twin call (25 ms), Cosserat 280 ms per pose"),
    step("geometry_sensitivity", [PY, "scripts/geometry_sensitivity.py"], ["results/geometry_sensitivity.json"],
         about="D30 vs D40: 2.9 / 1.9 mm, 3 %; 6.5 vs 1.4 mm between notches"),
    step("calib_synthetic", [PY, "scripts/calib_synthetic.py"], ["results/synthetic_calib.json"],
         about="calibration on synthetic data: bending scale 0.003, mu 0.244, mass drift (60 % in the paper)"),
    step("arm_calibration_fem", [PY, "scripts/arm_calibration_fem.py"], ["results/arm_calibration_fem.json"],
         about="moment arm as a 5th calibration parameter: SOFA as the robot, prior arm 10.15 vs 10.55 mm"),
    step("mc_ablation", [PY, "scripts/mc_ablation.py"], ["results/mc_ablation.json"],
         about="footnote: one module only 6.5 mm, +10 two-module poses 0.73 mm (older robot)"),
    step("fem_mass", [PY, "scripts/fem_mass.py"], ["results/fem_mass.json"],
         about="FEM nodal mass vs the part: '2 % too much weight' (needs data/meshes/std6_*.vtk)"),
    step("mesh_error", [PY, "scripts/mesh_error.py"], ["results/mesh_error.json"],
         about="0.6 mm FEM mesh error: stored old-robot values only (needs SOFA)"),
    step("fig_profile", [PY, "scripts/fig_profile.py", "--out-dir", "results/figures"],
         ["results/figures/fig_profile.pdf"], about="Fig. 1"),
    step("fig_transfer", [PY, "scripts/fig_transfer.py", "--out-dir", "results/figures"],
         ["results/figures/fig_transfer.pdf"], about="Fig. 2"),
    step("figures_compare", [PY, "scripts/claims.py", "--compare-figures"], ["results/figures/compare.json"],
         about="Fig. 1 and 2 against paper/figures/ (pixel comparison at 150 dpi)"),
    step("claims", [PY, "scripts/claims.py"], ["results/claims.json", "results/CLAIMS.md"],
         about="every number of paper/main.tex: printed vs regenerated"),
]


def have(module):
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="also the multi-hour jobs (n_transfer, PINN)")
    ap.add_argument("--only", nargs="+", choices=[s["name"] for s in STEPS], help="run only these steps")
    ap.add_argument("--skip", nargs="+", default=[], choices=[s["name"] for s in STEPS])
    ap.add_argument("--list", action="store_true", help="print the steps and exit")
    a = ap.parse_args()
    if a.list:
        for s in STEPS:
            env = " ".join(f"{k}={v}" for k, v in s["env"].items())
            print(f"{s['name']:21s} {'[long] ' if s['long'] else ''}{s['about']}\n    {env + ' ' if env else ''}"
                  f"{' '.join(os.path.relpath(c, ROOT) if c == PY else c for c in s['cmd'])}\n    -> {', '.join(s['writes'])}")
        return
    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
    mode = "full" if a.full else "fast"
    todo = [s for s in STEPS if (a.only is None or s["name"] in a.only) and s["name"] not in a.skip]
    if a.only is None and not a.full:
        todo = [s for s in todo if not s["long"]]
    summary, failed = [], []
    t_all = time.time()
    manifest_path = os.path.join(ROOT, "results", "logs", "last_run.json")
    try:                                    # steps of earlier runs stay listed, with their own time and mode
        manifest = json.load(open(manifest_path))
    except (OSError, ValueError):
        manifest = {}
    manifest.setdefault("steps", {})
    for s in todo:
        if s["needs"] and not have(s["needs"]):
            print(f"[skip] {s['name']}: needs {s['needs']} (pip install -e '.[pinn]')", flush=True)
            summary.append((s["name"], "skipped", 0.0))
            continue
        env = dict(os.environ, **s["env"], PYTHONUNBUFFERED="1", REPRODUCE_MODE=mode)
        log = os.path.join(ROOT, "results", "logs", f"{s['name']}.log")
        print(f"[run ] {s['name']:21s} {s['about']}", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            r = subprocess.run(s["cmd"], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        ok = r.returncode == 0
        summary.append((s["name"], "ok" if ok else f"FAILED ({r.returncode})", dt))
        print(f"       {'ok' if ok else 'FAILED, see ' + os.path.relpath(log, ROOT)} in {dt:.0f} s", flush=True)
        if not ok:
            failed.append(s["name"])
        manifest["steps"][s["name"]] = dict(ok=ok, seconds=round(dt, 1), mode=mode,
                                            finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(manifest, open(manifest_path, "w"), indent=1)
    print(f"\n{mode} mode, {time.time() - t_all:.0f} s in total")
    for name, st, dt in summary:
        print(f"  {name:21s} {st:12s} {dt:7.0f} s")
    if any(s["name"] == "claims" for s in todo) and "claims" not in failed:
        print("\nresults/CLAIMS.md: the paper's numbers, printed vs regenerated; `git diff --stat results/` shows "
              "which result files moved")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
