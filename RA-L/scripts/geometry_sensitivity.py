"""
Sensitivity to the module geometry (paper Sec. VI-C): the same tendon displacements through the twin built on the
simulated D = 30 mm notch disc and on the printed D = 40 mm disc (data/cad/std6_tdcr_L150_D40.json, same 10 + 9
notches), with the SOFA-identified notch corrections (results/mc_std6.json) kept. Paper: the tip moves 2.9 mm median
on two modules without gravity (3 % of the tip travel) and 1.9 mm hanging; D40 prints with 6.5 mm instead of 1.4 mm of
material between notches.

Recovered from the old repository's session of 26 Oct 2026 (a scratch script, geom_transfer.py, run before the
29 Oct gravity-solver fix); the CAD swap now goes through Twin(cad_json=...) instead of patching
tdcr.eb_prior.CAD_JSON, the D40 module mass is scaled by the CAD volume ratio (the original used the rounded
34.8 / 27.9), and the constant-curvature comparison it also printed is left out (not in the paper).

    .venv/bin/python scripts/geometry_sensitivity.py     # ~1 min -> results/geometry_sensitivity.json
"""
import argparse
import json
import os
import warnings
warnings.simplefilter("ignore")

import numpy as np

from tdcr.compliance import MASS_STD6, load_set
from tdcr.params import STD6
from tdcr.paths import CAD, RESULTS, ROOT
from tdcr.twin import Twin

D30 = os.path.join(CAD, "std6_tdcr_L150.json")             # the simulated module (the twin's profile is std_tdcr_L150,
D40 = os.path.join(CAD, "std6_tdcr_L150_D40.json")         # same geometry; holes cut via HOLES6)


def material_between_notches(path):
    g = json.load(open(path))
    return g["pitch_mm"] - g["notch_height_mm"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", type=int, default=40, help="first test commands of each set")
    ap.add_argument("--out", default=os.path.join(RESULTS, "geometry_sensitivity.json"))
    a = ap.parse_args()
    v30, v40 = (json.load(open(p))["volume_mm3"] for p in (D30, D40))
    m40 = MASS_STD6 * v40 / v30
    res = dict(material_between_notches_mm=dict(D30=material_between_notches(D30), D40=material_between_notches(D40)),
               mass_kg=dict(D30=MASS_STD6, D40=m40), poses=a.poses, runs={})
    for n, name in ((2, "std6_2s_sym_test_g0"), (4, "std6_4s_sym_test_g0")):
        dl = load_set(name)["dl"][:a.poses]
        for mount in ((0.0, 0.0, 0.0), "hanging", "upright"):
            ta, tb = Twin(n, mount, mass_kg=MASS_STD6), Twin(n, mount, mass_kg=m40, cad_json=D40)
            d, travel = [], []
            for x in dl:
                sa, sb = ta.shape(x), tb.shape(x)
                if not (sa.converged and sb.converged):
                    continue
                d.append(np.linalg.norm(sa.positions[-1] - sb.positions[-1]))
                travel.append(np.linalg.norm(sa.positions[-1] - [0, 0, STD6.module_length_mm * n]))
            key = f"{n}s_{'g0' if mount == (0.0, 0.0, 0.0) else mount}"
            res["runs"][key] = dict(d30_vs_d40_tip_median_mm=float(np.median(d)), d30_vs_d40_tip_max_mm=float(np.max(d)),
                                    tip_travel_median_mm=float(np.median(travel)),
                                    ratio_to_travel=float(np.median(d) / np.median(travel)), poses=len(d))
            print(f"{key:12s}: D30 vs D40 tip median {np.median(d):5.2f}  max {np.max(d):5.2f} mm | tip travel median "
                  f"{np.median(travel):5.1f} mm ({len(d)} poses)", flush=True)
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"-> {os.path.relpath(os.path.join(ROOT, a.out), ROOT)}")


if __name__ == "__main__":
    main()
