"""
Mass distribution of the FEM ground truth vs the part (paper Sec. VI-B: "the FEM's uniform nodal mass places 2 % too
much weight in the notches"): the FEM's UniformMass (equal mass per node, every std6 sweep), the true material
distribution (tet volumes) and the EB prior's (CAD area per cell). Compares the centre of mass, the mass in notched
cells and the base moment of a straight horizontal stack.

Needs the std6 FEM meshes (data/meshes/std6_{one,two}_section_sym.vtk, not in the repository: data/README.md); a
mesh that is missing is reported as such and its numbers are left out.

Ported from the physics audit (reviews/physics_audit/mass_check.py, W3, 2 Nov 2026, findings C3/K3); only the imports,
the mesh location and the output path changed.

    .venv/bin/python scripts/fem_mass.py                 # seconds -> results/fem_mass.json
"""
import argparse
import json
import os

import numpy as np

from tdcr.compliance import HOLES6, MASS_STD6, ModuleModel
from tdcr.paths import MESHES, RESULTS, ROOT


def _read_tets(path):
    """Legacy ASCII VTK -> points (N, 3), tetrahedra (M, 4) (as sim/mass.py, which is not importable from scripts/)."""
    tok = open(path).read().split()
    i = tok.index("POINTS")
    n = int(tok[i + 1])
    pts = np.array(tok[i + 3:i + 3 + 3 * n], float).reshape(n, 3)
    j = tok.index("CELLS")
    m, size = int(tok[j + 1]), int(tok[j + 2])
    flat = np.array(tok[j + 3:j + 3 + size], int)
    tets, k = [], 0
    for _ in range(m):
        c = flat[k]
        if c == 4:
            tets.append(flat[k + 1:k + 5])
        k += c + 1
    return pts, np.array(tets)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshes", default=MESHES)
    ap.add_argument("--out", default=os.path.join(RESULTS, "fem_mass.json"))
    a = ap.parse_args()
    model, theta = ModuleModel.from_result(json.load(open(os.path.join(RESULTS, "mc_std6.json"))))
    out = {}
    for n, name in ((1, "one"), (2, "two")):
        mesh = os.path.join(a.meshes, f"std6_{name}_section_sym.vtk")
        if not os.path.exists(mesh):
            out[f"N{n}"] = dict(missing=os.path.relpath(mesh, ROOT))
            print(f"N{n}: {os.path.relpath(mesh, ROOT)} not found (data/README.md), skipped")
            continue
        pts, tets = _read_tets(mesh)
        M = MASS_STD6 * n
        y = pts[:, 1]
        m_uniform = np.full(len(y), M / len(y))
        p0, p1, p2, p3 = (pts[tets[:, i]] for i in range(4))
        vol = np.abs(np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0)) / 6
        yt = (p0[:, 1] + p1[:, 1] + p2[:, 1] + p3[:, 1]) / 4
        m_vol = vol / vol.sum() * M
        eb = model.build(theta, n, "paired60", None, HOLES6)
        area = eb.cell / eb.cA_cad
        per = len(eb.yc) // n
        m_eb = np.concatenate([MASS_STD6 * area[j * per:(j + 1) * per] / area[j * per:(j + 1) * per].sum()
                               for j in range(n)])
        row = {}
        for lab, yy, mm in (("fem_uniform_nodes", y, m_uniform), ("true_volume", yt, m_vol), ("eb_prior", eb.yc, m_eb)):
            row[lab] = dict(com_y_mm=float((yy * mm).sum() / mm.sum()), base_sleeve_frac=float(mm[yy < 10].sum() / mm.sum()),
                            base_moment_mNmm=float((yy * mm).sum() * 9810.0), second_moment=float((yy ** 2 * mm).sum()))
        for lab in ("fem_uniform_nodes", "eb_prior"):
            row[lab]["base_moment_vs_true"] = row[lab]["base_moment_mNmm"] / row["true_volume"]["base_moment_mNmm"] - 1
        row["mesh_nodes"], row["mesh_volume_mm3"] = int(len(y)), float(vol.sum())
        cut = np.tile(eb.cA_cad[:per] > 1.0001 * eb.cA_cad[:per].min(), n)   # cells where a notch removes material
        for lab, yy, mm in (("fem_uniform_nodes", y, m_uniform), ("true_volume", yt, m_vol)):
            k = np.clip(np.searchsorted(eb.edges, yy) - 1, 0, len(cut) - 1)
            row[lab]["notched_cell_mass_frac"] = float(mm[cut[k]].sum() / mm.sum())
        row["eb_prior"]["notched_cell_mass_frac"] = float(m_eb[cut].sum() / m_eb.sum())
        out[f"N{n}"] = row
        print(f"N{n}: FEM base moment {100 * row['fem_uniform_nodes']['base_moment_vs_true']:+.1f} % vs the part; "
              f"mass in notched cells {100 * row['fem_uniform_nodes']['notched_cell_mass_frac']:.1f} % (FEM) vs "
              f"{100 * row['true_volume']['notched_cell_mass_frac']:.1f} % (part)", flush=True)
    json.dump(out, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"-> {os.path.relpath(os.path.join(ROOT, a.out), ROOT)}")


if __name__ == "__main__":
    main()
