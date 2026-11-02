"""
Data ablation of the module identification (paper Sec. IV-B and its footnote): trained on one module only, the model
errs by 6.5 mm on four modules; with ten two-module poses added, by 0.73 mm. On the EARLIER robot variant with three
shared channels (std_* sets: 144 one-module and 200 two-module gravity-free training poses; tests std_1s_sym_test_g0,
std_2s_sym_test_g0, std_4s_sym_stabletest_g0), not the std6 robot of the rest of the paper.

Recovered from the old repository's session of 7 Oct 2026, which wrote results/mc_ablation.json with this computation
(knots every 10 mm, smooth 0.05, 40 iterations; random two-module subsets with numpy's default_rng(0)). Each value is
[mean tip error (mm), mean tip orientation error (deg)] on the 1-, 2- and 4-module test sets. The code has changed
since (std6 work, gravity solver); regenerated, the file matches the stored one to 3 decimals except
"1s (144) + 2s (25)" on four modules (0.62 -> 1.87 mm), which the paper does not print.

    .venv/bin/python scripts/mc_ablation.py              # ~30 s (forks up to 20 processes) -> results/mc_ablation.json
"""
import argparse
import json
import os
import warnings
warnings.simplefilter("ignore")

import numpy as np

from tdcr.backbones import evaluate
from tdcr.compliance import TESTS, ModuleModel, fit, load_set
from tdcr.paths import RESULTS, ROOT


def subset(d, idx):
    P = len(d["C"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and k != "stations" and v.ndim and len(v) == P else v)
            for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(RESULTS, "mc_ablation.json"))
    a = ap.parse_args()
    tests = [load_set(t) for t in TESTS[:3]]
    d1, d2 = load_set("std_1s_sym_g0"), load_set("std_2s_sym_g0")
    rng = np.random.default_rng(0)
    res = {}

    def run(name, train):
        m = ModuleModel(10.0)
        th = fit(m, train, smooth=0.05, iters=40)
        sc = []
        for d in tests:
            C, R, _ = m.predict(th, d)
            s = evaluate(C, R, d["C"], d["R"])
            sc.append((float(s["tip"].mean()), float(s["tip_rot"].mean())))
        res[name] = sc
        print(f"{name:28s} " + " | ".join(f"{n}s {t:5.2f} mm {r:4.2f} deg" for (t, r), n in zip(sc, (1, 2, 4))),
              flush=True)

    run(f"1s only ({len(d1['C'])})", [d1])
    run(f"2s only ({len(d2['C'])})", [d2])
    for k in (5, 10, 25, 50):
        idx = rng.choice(len(d2["C"]), k, replace=False)
        run(f"1s ({len(d1['C'])}) + 2s ({k})", [d1, subset(d2, idx)])
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"-> {os.path.relpath(os.path.join(ROOT, a.out), ROOT)}")


if __name__ == "__main__":
    main()
