"""
Table I's identified-physics columns, re-evaluated: the identified module (results/mc_std6.json) on every std6 test
set, through ModuleModel.predict (gravity fixed point at its default tolerance, as Table I), after the 29 Oct
convergence fix. The paper's comment under Table I cites this file ("re-evaluated after the convergence fix:
unchanged").

Recovered from the old repository's session of 29 Oct 2026, which wrote results/mc_std6_reeval.json with this
computation and printed no producer; nothing but the imports changed.

    .venv/bin/python scripts/mc_reeval.py               # ~1 min -> results/mc_std6_reeval.json
"""
import argparse
import json
import os
import warnings
warnings.simplefilter("ignore")

import numpy as np

from tdcr.compliance import TESTS_STD6, ModuleModel, load_set
from tdcr.paths import RESULTS, ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc", default=os.path.join(RESULTS, "mc_std6.json"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "mc_std6_reeval.json"))
    a = ap.parse_args()
    res = json.load(open(a.mc))
    m, th = ModuleModel.from_result(res)
    out = {}
    for t in TESTS_STD6:
        d = load_set(t)
        C, R, lam = m.predict(th, d)
        tip = np.linalg.norm(C[:, -1] - d["C"][:, -1], axis=1)
        Rr = np.einsum("pji,pjk->pik", R[:, -1], d["R"][:, -1])
        rot = np.degrees(np.arccos(np.clip((np.trace(Rr, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        old = res["learned"][t]
        out[t] = dict(tip=float(tip.mean()), tip95=float(np.percentile(tip, 95)), rot=float(rot.mean()), poses=len(tip))
        print(f"{t:26s} tip {tip.mean():.3f} (was {old['tip']:.3f})  p95 {np.percentile(tip, 95):.2f} "
              f"(was {old['tip95']:.2f})  rot {rot.mean():.3f} (was {old['rot']:.3f})  n={len(tip)}", flush=True)
    json.dump(dict(source="results/mc_std6.json re-evaluated after ff9fa02/f464bd2 (convergence on axial strain)",
                   tests=out), open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"-> {os.path.relpath(os.path.join(ROOT, a.out), ROOT)}")


if __name__ == "__main__":
    main()
