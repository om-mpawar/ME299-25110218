"""
Does the gravity fixed-point tolerance change Table I? Re-evaluates every std6 test set with gravity of
results/mc_std6.json with the tolerance the paper's numbers used (predict_gravity default tol = 1e-4, iters = 60, via
ModuleModel.predict) and with a tight one (1e-11, 4000 iterations), against the FEM backbones.

Ported from the physics audit (reviews/physics_audit/tol_reeval.py, W3, 2 Nov 2026, finding R2); only the imports and
the output path changed.

    .venv/bin/python scripts/tol_reeval.py              # ~1 min -> results/tol_reeval.json
"""
import argparse
import json
import os
import warnings
warnings.simplefilter("ignore")

import numpy as np

from tdcr.backbones import evaluate
from tdcr.compliance import E_FEM, TESTS_STD6, ModuleModel, load_set
from tdcr.eb_prior import predict_gravity
from tdcr.paths import RESULTS, ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(RESULTS, "tol_reeval.json"))
    a = ap.parse_args()
    model, theta = ModuleModel.from_result(json.load(open(os.path.join(RESULTS, "mc_std6.json"))))
    out = {}
    for tag in TESTS_STD6:
        d = load_set(tag)
        if d["g"] is None:
            continue
        eb = model.build(theta, d["n"], d["routing"], d["az"], d["holes"])
        row = {}
        for name, tol, iters in (("paper_tol1e-4", 1e-4, 60), ("tight_tol1e-11", 1e-11, 4000)):
            C, R, lam, info = predict_gravity(eb, d["dl"], d["stations"], g_vec=d["g"], E=E_FEM, mass=d["mass"],
                                              tol=tol, iters=iters)
            s = evaluate(C, R, d["C"], d["R"])
            row[name] = dict(tip=float(s["tip"].mean()), tip95=float(np.percentile(s["tip"], 95)),
                             rot=float(s["tip_rot"].mean()), converged=bool(info["converged"]),
                             iterations=info["iterations"])
            row[name + "_C"] = C
        dC = np.linalg.norm(row["paper_tol1e-4_C"] - row["tight_tol1e-11_C"], axis=2)
        row["shape_change_max_mm"] = float(dC.max())
        row["tip_change_mean_mm"] = float(dC[:, -1].mean())
        del row["paper_tol1e-4_C"], row["tight_tol1e-11_C"]
        out[tag] = row
        print(tag, json.dumps(row), flush=True)
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"-> {os.path.relpath(os.path.join(ROOT, a.out), ROOT)}")


if __name__ == "__main__":
    main()
