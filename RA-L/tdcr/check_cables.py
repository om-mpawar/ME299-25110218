"""
Check that every tendon point in a cables.json lies inside the body mesh.

Why this matters
----------------
SOFA binds each cable point to the backbone with a BarycentricMapping: it
finds the tetrahedron containing the point and expresses the point in that
tet's barycentric coordinates, so the point moves with the material around
it. That is exactly right for a point inside a groove.

A point OUTSIDE the material has no containing tet. The mapping falls back
to the nearest one and extrapolates -- so the point is carried along by
material it is not in, and the cable path, its length change, and the load
it puts on the body are all wrong. Nothing warns you. The simulation runs
and the curvature is simply off.

This is the check to run whenever the tendon paths or the mesh change, and
before any sweep.

    .venv/bin/python -m tdcr.check_cables data/cables/std6_one_section_sym.cables.json data/meshes/std6_one_section_sym.vtk
"""

import argparse
import sys

import numpy as np

from tdcr.make_cables_json import load_cables_json


def read_tets(path):
    """-> (points (N,3), tets (M,4)) from a legacy ASCII VTK."""
    lines = open(path).read().split("\n")
    pts, tets = None, []
    i = 0
    while i < len(lines):
        head = lines[i].strip().upper()
        if head.startswith("POINTS"):
            n = int(lines[i].split()[1]); vals = []; i += 1
            while len(vals) < 3 * n:
                vals += [float(v) for v in lines[i].split()]; i += 1
            pts = np.array(vals[:3 * n]).reshape(n, 3); continue
        if head.startswith("CELLS"):
            n = int(lines[i].split()[1]); i += 1; read = 0
            while read < n:
                parts = [int(v) for v in lines[i].split()]; i += 1
                if not parts:
                    continue
                if parts[0] == 4:
                    tets.append(parts[1:5])
                read += 1
            continue
        i += 1
    return pts, np.array(tets, dtype=int)


def inside_mask(query, pts, tets, tol=1e-6):
    """True where a query point lies in at least one tetrahedron."""
    A = pts[tets[:, 0]]
    M = np.stack([pts[tets[:, 1]] - A, pts[tets[:, 2]] - A,
                  pts[tets[:, 3]] - A], axis=2)            # (M, 3, 3)
    det = np.linalg.det(M)
    ok = np.abs(det) > 1e-14
    Minv = np.zeros_like(M)
    Minv[ok] = np.linalg.inv(M[ok])
    lo, hi = np.minimum.reduce([pts[tets[:, k]] for k in range(4)]), \
        np.maximum.reduce([pts[tets[:, k]] for k in range(4)])

    res = np.zeros(len(query), dtype=bool)
    for qi, q in enumerate(query):
        c = ok & np.all(q >= lo - tol, axis=1) & np.all(q <= hi + tol, axis=1)
        if not c.any():
            continue
        lam = np.einsum("mij,mj->mi", Minv[c], q - A[c])
        l0 = 1.0 - lam.sum(axis=1)
        if np.any((lam >= -tol).all(axis=1) & (l0 >= -tol)):
            res[qi] = True
    return res


def check(cables_path, mesh_path, verbose=True):
    pts, tets = read_tets(mesh_path)
    cables = load_cables_json(cables_path)
    worst = 1.0
    if verbose:
        print(f"mesh   {mesh_path}: {len(pts)} nodes, {len(tets)} tets")
        print(f"cables {cables_path}: {len(cables)} tendons\n")
        print(f"  {'tendon':12s} {'points':>6} {'inside':>7}  outside at y (mm)")
    for c in cables:
        m = inside_mask(c["points"], pts, tets)
        frac = m.mean()
        worst = min(worst, frac)
        if verbose:
            ys = c["points"][~m, 1]
            where = (f"{len(ys)} pts, e.g. {', '.join(f'{v:.0f}' for v in ys[:8])}"
                     f"{' ...' if len(ys) > 8 else ''}") if len(ys) else "none"
            print(f"  {c['name']:12s} {len(m):6d} {100*frac:6.1f}%  {where}")
    if verbose:
        print(f"\n  worst tendon: {100*worst:.1f}% of its points inside the material")
        if worst < 1.0:
            print("  -> points outside the body are bound to the nearest tet and")
            print("     extrapolated. The cable path and the load it applies are wrong")
            print("     wherever that happens.")
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cables")
    ap.add_argument("mesh")
    args = ap.parse_args()
    sys.exit(0 if check(args.cables, args.mesh) >= 1.0 else 1)
