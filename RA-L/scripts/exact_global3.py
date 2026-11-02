"""
Table II, first row, and the text's "the exact solver ... finds x20.2/5.8/5.3 in 80 s": the three notch log-scales of
the PINN's physics (plane-section CAD x one factor per compliance on the notched cells, tdcr.baselines.pinn), fitted by
least squares through the exact solver on the PINN's own training data (the two 2-module training sets), then tested on
the 2-module sets and, unchanged, on four modules.

Recovered from the old repository's session of 29 Oct 2026, which wrote results/pinn/exact_global3_2s.json with this
computation (run with OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2; 80 s, 10 evaluations); the fit time is now stored too.
Needs the `pinn` extra (torch), because the parametrisation is the PINN module's.

Also solves the PINNs' identified laws exactly (Table II, last row; tdcr.baselines.pinn.exact_same_physics on the
notch factors stored by the PINN runs) -> results/pinn/pinn_laws_exact.json.

    OMP_NUM_THREADS=2 NUMBA_NUM_THREADS=2 .venv/bin/python scripts/exact_global3.py   # ~3 min
    -> results/pinn/exact_global3_2s.json, results/pinn/pinn_laws_exact.json
"""
import argparse
import json
import os
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.optimize import least_squares

from tdcr.compliance import E_FEM, load_set
from tdcr.eb_prior import predict_gravity
from tdcr.paths import RESULTS, ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(RESULTS, "pinn", "exact_global3_2s.json"))
    a = ap.parse_args()
    import torch
    from tdcr.baselines import pinn as pb

    geo = pb.Geometry(2, cell=0.5)
    eb = geo.eb
    base = (eb.cA.copy(), eb.cX.copy(), eb.cZ.copy())
    train = [load_set(t) for t in pb.TRAIN_2S]

    def set_scales(s):
        f = pb.compliance_factors(geo, torch.tensor(s)).numpy()
        eb.cA, eb.cX, eb.cZ = base[0] * f[0], base[1] * f[1], base[2] * f[2]
        eb.assemble()

    def pred(d):
        if d["g"] is None:
            return eb.predict(d["dl"], geo.stations)[0]
        return predict_gravity(eb, d["dl"], geo.stations, g_vec=d["g"], E=E_FEM, mass=d["mass"], tol=1e-8, iters=400)[0]

    def resid(s):
        set_scales(s)
        return np.concatenate([(pred(d) - d["C"]).ravel() for d in train])

    t = time.time()
    r = least_squares(resid, np.zeros(3), diff_step=1e-3, x_scale=1.0)
    seconds = time.time() - t
    print(f"exact-solver identification of 3 notch scales on 2-module data: factors {np.round(np.exp(r.x), 2).tolist()} "
          f"({seconds:.0f} s, {r.nfev} evals)", flush=True)
    set_scales(r.x)
    out = {}
    for tag in pb.TESTS_2S:
        d = load_set(tag)
        out[tag] = pb.errors(pred(d), d["C"])
        print(f"  {tag:26s} tip {out[tag]['tip']:.2f} mm", flush=True)
    tests_4s = pb.exact_same_physics(4, [pb.TEST_4S], r.x)
    print(f"  {pb.TEST_4S:26s} tip {tests_4s[pb.TEST_4S]['tip']:.2f} mm (law identified on 2 modules)")
    res = dict(factors=np.exp(r.x).tolist(), tests_2s=out, tests_4s=tests_4s, fit_seconds=seconds, nfev=int(r.nfev),
               threads=dict(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS"),
                            NUMBA_NUM_THREADS=os.environ.get("NUMBA_NUM_THREADS")))
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"-> {os.path.relpath(os.path.join(ROOT, a.out), ROOT)}")

    # Table II, last row ("its law, solved exactly"): the inverse PINN's identified notch factors (stored result of the
    # long PINN run) solved by the exact solver, as tdcr.baselines.pinn does at the end of its runs; cheap, so redone
    laws = {}
    for src in ("2s_inverse_w10.json", "2s_forward_w0.1.json"):
        path = os.path.join(RESULTS, "pinn", src)
        if not os.path.exists(path):
            continue
        f = json.load(open(path))["notch_factors"]
        ls = np.log(np.asarray(f))
        laws[src] = dict(notch_factors=f, tests_2s=pb.exact_same_physics(2, pb.TESTS_2S, ls),
                         tests_4s=pb.exact_same_physics(4, [pb.TEST_4S], ls))
        print(f"  law of {src} ({np.round(f, 2).tolist()}) solved exactly: "
              + ", ".join(f"{t.replace('std6_', '')} {v['tip']:.2f}" for t, v in
                          {**laws[src]["tests_2s"], **laws[src]["tests_4s"]}.items()) + " mm", flush=True)
    out_laws = os.path.join(os.path.dirname(os.path.join(ROOT, a.out)), "pinn_laws_exact.json")
    json.dump(laws, open(out_laws, "w"), indent=1)
    print(f"-> {os.path.relpath(out_laws, ROOT)}")


if __name__ == "__main__":
    main()
