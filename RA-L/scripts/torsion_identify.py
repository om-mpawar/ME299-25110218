"""
Torsional compliance of the std6 module, identified from the SOFA twist test (sim/twist_test.py ->
results/twist_test_std6.json; physics audit C3).

factor = FEM twist of the module's top face under a pure torque / the plane-section prediction (integral of 1 / (G J_p)
over the module, G = E / 2(1 + nu), J_p = Jx + Jz of the CAD sections with the six channels), at the smaller load
(the test is linear: the larger load agrees to 1e-4). The same test's pure bending moments are compared with the
bending compliance identified from the tendon sweeps (results/mc_std6.json): their ratio checks the test, not the
model. Cells are counted from y = 0, as the twin does (the FEM clamps to y = 4 mm; the ratios from y = 4 are also
written).

    .venv/bin/python scripts/torsion_identify.py          # -> results/torsion_std6.json
"""
import argparse
import json
import os

import numpy as np

from tdcr.compliance import HOLES6, ModuleModel
from tdcr.eb_prior import geometry_from_cables
from tdcr.params import STD6
from tdcr.paths import CABLES, RESULTS, ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", default=os.path.join(RESULTS, "twist_test_std6.json"))
    ap.add_argument("--compliance", default=os.path.join(RESULTS, "mc_std6.json"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "torsion_std6.json"))
    a = ap.parse_args()
    fem = json.load(open(a.test))
    model, theta = ModuleModel.from_result(json.load(open(a.compliance)))
    az = geometry_from_cables(os.path.join(CABLES, "std6_one_section_sym.cables.json"))[1]
    eb = model.build(theta, 1, "paired60", az, HOLES6)
    E = fem["young_kpa"]
    loads = sorted(fem["cases"]["twist"], key=float)
    M = float(loads[0])
    comp = dict(twist=(1, eb.cT_plane), bend_x=(0, eb.cZZ), bend_z=(2, eb.cXX))   # rotation component, compliance
    ratios = {}
    for start in (0.0, 4.0):
        for case, (k, c) in comp.items():
            for level, rot in fem["cases"][case][loads[0]].items():
                if level.startswith("_"):
                    continue
                pred = M / E * c[(eb.yc >= start) & (eb.yc <= float(level))].sum()
                ratios.setdefault(f"from_y{start:g}", {}).setdefault(case, {})[level] = rot[k] / pred
    top = ratios["from_y0"]
    lin = {case: fem["cases"][case][loads[-1]]["160"][k] / fem["cases"][case][loads[0]]["160"][k]
           / (float(loads[-1]) / M) - 1.0 for case, (k, _) in comp.items()}
    res = dict(factor=top["twist"]["160"], bending_check=dict(about_x=top["bend_x"]["160"], about_z=top["bend_z"]["160"]),
               nonlinearity=lin, load_mN_mm=M, poisson=STD6.poisson, ratios=ratios,
               source=os.path.relpath(a.test, ROOT), compliance=os.path.relpath(a.compliance, ROOT))
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"torsion: FEM twist / plane-section = {res['factor']:.3f}; bending check (FEM / identified): about x "
          f"{res['bending_check']['about_x']:.3f}, about z {res['bending_check']['about_z']:.3f}; "
          f"nonlinearity {max(abs(v) for v in lin.values()):.1e} -> {os.path.relpath(a.out, ROOT)}")


if __name__ == "__main__":
    main()
