"""
Write the tendon paths for the two-section robot to a cables.json file.

Why a file rather than code
---------------------------
Tendon routing is a property of the HARDWARE -- which groove each tendon
runs in, at what radius, through which sections. It should be written down
once, inspected, and loaded, not recomputed inside the scene from a pair of
constants. With a file:

  * the path SOFA uses is the path you can open and read, point by point;
  * a path traced from CAD (or measured off the robot) drops straight in,
    with no change to the scene;
  * check_cables.py can verify every point actually lies inside the body
    before a sweep spends hours on it.

Format
------
Extends the convention used in the Simulating_TDCRs pipeline, where a cable
file is a plain list of [x, y, z] points. Six tendons with routing need a
little more, so the top level is an object:

    {
      "units": "mm",
      "frame": "base at origin, backbone axis +Y",
      "source": "how these paths were produced",
      "tendons": [
        {"name": "s1_0deg", "section": 1,
         "servo":  [x, y, z],              <- pull point, below the base
         "points": [[x, y, z], ...]},      <- base -> termination, in order
        ...
      ]
    }

Order in "tendons" is dl1..dl6: section 1's three, then section 2's three.
A section-2 tendon's points start at the BASE, not at section 2 -- it is
routed through section 1's grooves on its way up, and a cable whose first
point is at section 2 hangs as a free chord through section 1.

A bare list of points (a Simulating_TDCRs cable1.json) is also accepted by
the loader and treated as a single tendon.

    python3 -m tdcr.make_cables_json                       # current model geometry
    python3 -m tdcr.make_cables_json --radius 9.5 --azimuths 0 120 240 --spacing 5
"""

import argparse
import json
import math
import os

import numpy as np

from tdcr.cc_kinematics import TENDON_R, THETA_DEG
from tdcr.multisection import SECTION_LENGTHS, SECTION_GAMMAS, section_lengths_for, gammas_for

from tdcr.paths import ROOT as _DIR  # paths below are relative to the repo root
DEFAULT_OUT = os.path.join(_DIR, "data", "cables", "two_section.cables.json")

SERVO_Y = -10.0          # pull points sit below the base, where the servos are
END_INSET = 5.0          # keep points off section boundaries (see below)


def tendon_path(section, radius, azimuth_rad, section_lengths=SECTION_LENGTHS,
                spacing=5.0, inset=END_INSET):
    """Points along one tendon, base to termination, in order.

    A tendon terminating in section `section` (0-based) runs through every
    section up to and including it. Points are inset `inset` mm from each
    section boundary: a point exactly on the interface plane is ambiguous
    between the two bodies, and at the base it would sit inside the clamp
    box and be held rigid. The short jumps left at each boundary are chords
    across the joint, which is a fair description of the connector.
    """
    x, z = radius * math.cos(azimuth_rad), radius * math.sin(azimuth_rad)
    pts = []
    for j in range(section + 1):
        lo = float(np.sum(section_lengths[:j])) + inset
        hi = float(np.sum(section_lengths[:j + 1])) - inset
        n = max(2, int(math.ceil((hi - lo) / spacing)) + 1)
        pts += [[round(x, 6), round(float(y), 6), round(z, 6)]
                for y in np.linspace(lo, hi, n)]
    return pts


def build(radii, azimuths_deg, section_lengths=SECTION_LENGTHS,
          gammas=SECTION_GAMMAS, spacing=5.0, source="make_cables_json.py"):
    """-> cables.json dict. radii[s] and azimuths_deg[s] per section."""
    Gamma = np.cumsum(np.asarray(gammas, dtype=float))
    tendons = []
    for s in range(len(section_lengths)):
        for k, deg in enumerate(azimuths_deg[s]):
            a = math.radians(deg) + float(Gamma[s])
            r = radii[s]
            tendons.append({
                "name": f"s{s+1}_{deg:g}deg",
                "section": s + 1,
                "radius_mm": r,
                "azimuth_deg": round(math.degrees(a), 6),
                "servo": [round(r * math.cos(a), 6), SERVO_Y,
                          round(r * math.sin(a), 6)],
                "points": tendon_path(s, r, a, section_lengths, spacing),
            })
    return {"units": "mm",
            "frame": "base at origin, backbone axis +Y",
            "source": source,
            "section_lengths_mm": [float(L) for L in section_lengths],
            "tendons": tendons}


def load_cables_json(path):
    """-> list of dicts with name, section, servo, points (numpy arrays).

    Accepts the object format above, or a bare list of [x, y, z] points
    (a Simulating_TDCRs cable1.json), which becomes one tendon whose servo
    sits directly below its first point.
    """
    with open(path) as f:
        d = json.load(f)

    if isinstance(d, list):
        pts = np.asarray(d, dtype=float)
        return [{"name": os.path.basename(path), "section": 1,
                 "servo": np.array([pts[0, 0], SERVO_Y, pts[0, 2]]),
                 "points": pts}]

    out = []
    for t in d["tendons"]:
        pts = np.asarray(t["points"], dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise ValueError(f"{path}: tendon {t.get('name')} needs >= 2 [x,y,z] points")
        if np.any(np.diff(pts[:, 1]) <= 0):
            raise ValueError(f"{path}: tendon {t.get('name')} points must run base -> tip "
                             "with strictly increasing y")
        out.append({"name": t["name"], "section": int(t["section"]),
                    "servo": np.asarray(t["servo"], dtype=float), "points": pts})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sections", type=int, default=len(SECTION_LENGTHS),
                    help="number of stacked sections (each the same 220 mm module)")
    ap.add_argument("--radius", type=float, nargs="+", default=[TENDON_R],
                    help="groove radius in mm; one value, or one per section")
    ap.add_argument("--azimuths", type=float, nargs="+", default=list(THETA_DEG),
                    help="groove azimuths in deg; 3 values (same for both "
                         "sections) or 6 (section 1's three, then section 2's)")
    ap.add_argument("--spacing", type=float, default=5.0,
                    help="point spacing along each tendon, mm")
    ap.add_argument("--inside-mesh", default=None,
                    help="drop points that do not lie inside this body mesh. "
                         "For a notched/flexure backbone the tendon threads "
                         "through the discs and spans the gaps between them "
                         "freely; a cable point in a gap would be bound to the "
                         "nearest tet and extrapolated. Keeping only in-material "
                         "points makes each gap a straight chord, as on hardware.")
    args = ap.parse_args()

    n = args.sections
    lengths, gammas = section_lengths_for(n), gammas_for(n)
    radii = args.radius * n if len(args.radius) == 1 else args.radius
    az = args.azimuths
    azs = [az] * n if len(az) == 3 else [az[3 * s:3 * s + 3] for s in range(n)]

    d = build(radii, azs, section_lengths=lengths, gammas=gammas, spacing=args.spacing,
              source=f"make_cables_json.py --radius {' '.join(map(str, args.radius))} "
                     f"--azimuths {' '.join(f'{a:g}' for a in az)} --spacing {args.spacing}")
    if args.inside_mesh:
        from tdcr.check_cables import read_tets, inside_mask
        mp, mt = read_tets(args.inside_mesh)
        for t in d["tendons"]:
            p = np.asarray(t["points"])
            keep = inside_mask(p, mp, mt)
            if keep.sum() < 2:
                raise SystemExit(f"{t['name']}: fewer than 2 points inside "
                                 f"{args.inside_mesh} -- wrong radius/azimuth for this part?")
            t["points"] = p[keep].tolist()
            t["dropped_outside"] = int((~keep).sum())
        d["source"] += f" --inside-mesh {args.inside_mesh}"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(d, f, indent=1)
    print(f"wrote {args.out}")
    for t in d["tendons"]:
        p = np.array(t["points"])
        print(f"  {t['name']:12s} section {t['section']}  r={t['radius_mm']:.2f}  "
              f"az={t['azimuth_deg']:7.2f}  {len(p):3d} pts  y {p[0,1]:.0f}->{p[-1,1]:.0f}"
              + (f"  ({t['dropped_outside']} outside the body, dropped)"
                 if t.get("dropped_outside") else ""))
