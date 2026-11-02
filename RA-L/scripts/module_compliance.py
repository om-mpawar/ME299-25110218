"""
Learn the MODULE, compose by statics: a compliance profile along the module, shared by every module, identified
inside the variable-section Euler-Bernoulli prior (tdcr/eb_prior.py) from 1- and 2-section FEM data, then used
unchanged for any number of sections. The model and the fit live in tdcr/compliance.py; this script runs the
identification, reports the test errors (paper Table I) and writes the result file.

The plane-section compliances per cell (integrals of 1/A, 1/Jx, 1/Jz) are multiplied by exp(s c(y) + theta(y)): c(y)
is 1 in the cells a notch cuts and 0 in the plain tube (where plane-section theory is accurate), s a global log-scale and theta a piecewise-linear field over the height y within the module (knots every `knot` mm),
one pair each for axial, bending towards x and bending towards z. theta is penalised (ridge and first differences,
so it stays smooth and near the CAD); s is free except for a tiny ridge that fixes the overall scale when no gravity
data are used. The multipliers are positive, so the complementary energy stays a positive
quadratic: the pull-only problem stays a strictly convex QP and the model composes to N modules exactly as the
physics does (Camarillo 2009 coupling, slack, gravity). theta = 0 is the CAD / plane-section model; a small ridge
pulls theta towards it. Without gravity the overall scale of the three fields cancels (only ratios are identifiable).

    .venv/bin/python scripts/module_compliance.py --train std_1s_sym_g0 std_2s_sym_g0

Real robot (std6, 6 channels, sections 3-4 at 60/180/300 deg), trained on 1-2 modules with and without gravity:

    .venv/bin/python scripts/module_compliance.py --train std6_1s_sym_train_g0 std6_2s_sym_train_g0 \
        std6_1s_sym_train std6_2s_sym_train --out results/mc_std6.json
"""
import argparse
import json
import os
import warnings
warnings.simplefilter("ignore")

import numpy as np

from tdcr.backbones import evaluate
from tdcr.compliance import SHEAR_MODES, TESTS, TESTS_STD6, ModuleModel, fit, load_set
from tdcr.paths import RESULTS


def report(model, theta, label, tests=TESTS):
    out = {}
    for tag in tests:
        if not os.path.exists(os.path.join(RESULTS, f"backbone_{tag}.npz")):
            print(f"   {label:38s} {tag}: no backbone file, skipped")
            continue
        d = load_set(tag); n = d["n"]
        C, R, lam = model.predict(theta, d)
        s = evaluate(C, R, d["C"], d["R"])
        out[tag] = dict(tip=float(s["tip"].mean()), tip95=float(np.percentile(s["tip"], 95)), rot=float(s["tip_rot"].mean()),
                        along=float(s["mean_along"].mean()))
        print(f"   {label:38s} {tag:26s} ({len(d['C']):3d}): tip {s['tip'].mean():5.2f} mm (95% {np.percentile(s['tip'], 95):5.2f}) | "
              f"orient. {s['tip_rot'].mean():4.2f} deg | along {s['mean_along'].mean():4.2f} mm")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["std_1s_sym_g0", "std_2s_sym_g0"])
    ap.add_argument("--knot", type=float, default=10.0)
    ap.add_argument("--ridge", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--init", type=float, nargs=3, help="initial global log-scales (axial, bend x, bend z)")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--shear", choices=[m for m in SHEAR_MODES if m], help="Timoshenko shear under transverse gravity")
    ap.add_argument("--init-from", help="start from the parameters of an earlier result (json); missing entries 0")
    ap.add_argument("--only-shear", action="store_true", help="--shear fit: fit only the shear factor, keep the rest")
    ap.add_argument("--tests", nargs="+", help="test sets (default: the std6 tests when training on std6_* sets)")
    ap.add_argument("--out", default="results/module_compliance.json")
    a = ap.parse_args()
    tests = a.tests or (TESTS_STD6 if all(t.startswith("std6_") for t in a.train) else TESTS)
    model = ModuleModel(a.knot, a.shear)
    train = []
    for tag in a.train:
        train.append(load_set(tag))
    print(f"training on {', '.join(f'{t} ({len(d['C'])})' for t, d in zip(a.train, train))}; "
          f"{model.n_theta} parameters (knots every {a.knot:g} mm), ridge {a.ridge}, shear {a.shear}")
    res = dict(train=a.train, knot=a.knot, ridge=a.ridge, shear=a.shear)
    if not a.skip_baseline:
        res["plane_section"] = report(model, np.zeros(model.n_theta), "EB plane-section (theta = 0)", tests)
    th0 = np.zeros(model.n_theta)
    if a.init_from:
        prev = np.array(json.load(open(a.init_from))["theta"])
        th0[:min(len(prev), 3 + 3 * model.K)] = prev[:3 + 3 * model.K]
        res["init_from"] = a.init_from
    if a.init:
        th0[:3] = a.init
    free = None
    if a.only_shear:
        if a.shear != "fit":
            raise SystemExit("--only-shear needs --shear fit")
        free = np.zeros(model.n_theta, bool); free[-1] = True
    theta = fit(model, train, ridge=a.ridge, iters=a.iters, smooth=a.smooth, theta0=th0, free=free)
    res["learned"] = report(model, theta, "EB + learned module compliance", tests)
    sc, f = model.split(theta)
    print(f"   global compliance scales (axial / bend x / bend z): x{np.exp(sc[0]):.3f} x{np.exp(sc[1]):.3f} x{np.exp(sc[2]):.3f}")
    if model.shear == "fit":
        print(f"   notch shear compliance: x{np.exp(theta[-1]):.3f} of Timoshenko on the CAD sections")
    print("   field multipliers at the knots (axial / bend x / bend z):")
    for k, y in enumerate(model.knots):
        print(f"      y {y:5.0f} mm: x{np.exp(f[0, k]):5.2f} x{np.exp(f[1, k]):5.2f} x{np.exp(f[2, k]):5.2f}")
    res.update(knots=model.knots.tolist(), theta=theta.tolist(), smooth=a.smooth)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"   -> {a.out}")


if __name__ == "__main__":
    main()
