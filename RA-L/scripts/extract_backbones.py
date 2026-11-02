"""
Compact backbones (centreline + cross-section rotations every 5 mm) straight from a pose
directory, for whatever poses exist so far (a sweep still running is fine).

Writes results/backbone_<tag>.npz with theta, alpha, dl, rows, stations, C, R, settle_mm, mesh --
the format tdcr/backbones.load reads (dl included, so no shape file is needed). The mesh named in the pose
directory's rest.npz must be in data/meshes (see data/README.md).

    .venv/bin/python scripts/extract_backbones.py --sections 4 \
        --dir results/sweeps/std6_4s_sym_test_g0 --tag std6_4s_sym_test_g0
"""
import argparse
import glob
import os

import numpy as np

from tdcr.backbones import backbone, node_sections, stations_for
from tdcr.paths import RESULTS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", type=int, required=True)
    ap.add_argument("--dir", required=True, nargs="+", help="one or more pose directories of the same robot")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    rest = np.load(os.path.join(args.dir[0], "rest.npz"))
    mesh = str(rest["mesh"])
    for d in args.dir[1:]:
        assert np.array_equal(np.load(os.path.join(d, "rest.npz"))["rest"], rest["rest"]), f"{d}: different mesh"
    files = sorted(f for d in args.dir for f in glob.glob(os.path.join(d, "pose_*.npz")))
    P = [np.load(f) for f in files]
    pos = np.array([p["pos"] for p in P], dtype=float)
    st = stations_for(args.sections)
    body = node_sections(mesh, len(rest["rest"]))
    C, R = backbone(np.asarray(rest["rest"], float), pos, st, body)
    out = os.path.join(RESULTS, f"backbone_{args.tag}.npz")
    np.savez_compressed(out, theta=np.array([p["theta"] for p in P]), alpha=np.array([p["alpha"] for p in P]),
                        dl=np.array([p["dl"] for p in P]), rows=np.array([int(p["row"]) for p in P]),
                        source=np.array([os.path.basename(os.path.dirname(f)) for f in files]),
                        stations=st, C=C.astype(np.float32), R=R.astype(np.float32),
                        settle_mm=np.array([float(p["settle_mm"]) for p in P]), mesh=mesh)
    print(f"{len(P)} poses -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
