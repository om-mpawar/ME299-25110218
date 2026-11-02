"""
Mass model of the SOFA scenes (TDCR_MASS): "uniform" (UniformMass, equal mass per node: every sweep up to 21 Oct
2026) or "volume" (MeshMatrixMass with density = module mass / mesh volume, mass distributed with the material).

UniformMass piles weight into the finely meshed notches: base sleeve 11.5 % of the module mass instead of 18 %, centre
of mass 80.5 instead of 77.3 mm, about 2 % more gravity moment than the real part. Use "volume" for new sweeps.
Kept free of Sofa imports so it can be tested without SOFA.
"""
import os

import numpy as np

MASS_MODEL = os.environ.get("TDCR_MASS", "uniform")


def _read_tets(path):
    """Legacy ASCII VTK -> points (N, 3), tetrahedra (M, 4)."""
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


def tet_mesh_volume(path):
    """Total volume (mm^3) of a tetrahedral VTK mesh."""
    p, t = _read_tets(path)
    a, b, c, d = (p[t[:, i]] for i in range(4))
    return float(np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)).sum() / 6.0)


def density_for(total_mass, path):
    """Density (kg / mm^3) that gives the mesh the total mass (kg)."""
    return total_mass / tet_mesh_volume(path)


def mass_component(model, total_mass, mesh):
    """-> (SOFA component name, its arguments) for the chosen mass model."""
    if model == "uniform":
        return "UniformMass", {"totalMass": total_mass}
    if model == "volume":
        return "MeshMatrixMass", {"massDensity": density_for(total_mass, mesh), "lumping": True}
    raise ValueError(f"TDCR_MASS must be 'uniform' or 'volume', not {model!r}")
