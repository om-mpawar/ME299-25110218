"""
The FEM's mesh error (paper Sec. VI-A "about 0.6 mm", Sec. VII "0.6 mm"): STUB, needs SOFA.

What the paper's number is: results/meshcheck/meshcheck_std_{1s,2s}_sym_meshcheck_fine_g0.json (copied unchanged from
the old repository, commit 805f5be, 4 Oct 2026): 30 gravity-free poses of the EARLIER three-channel `std` robot, each
simulated on the coarse and on a fine mesh; the numbers are the coarse-vs-fine difference (tip mean 0.56 mm for one
module, 0.59 mm for two), not the error against a converged solution, and not the std6 robot or gravity (physics
audit, reviews/physics_audit.md, finding C5).

Regenerating it needs SOFA 25.06 + SoftRobots, the std coarse and fine meshes and the sweep pipeline (sim/,
scripts/extract_backbones.py); none of that runs here (the meshes are not in the repository, data/README.md). No std6
mesh-convergence study exists: it would need fine std6 meshes (to be generated) and SOFA runs of a few poses with
gravity (physics audit K2). This script only collects the stored values into results/mesh_error.json, so that
scripts/claims.py can report them, marked as not regenerated.

    .venv/bin/python scripts/mesh_error.py               # seconds -> results/mesh_error.json
"""
import json
import os

from tdcr.paths import RESULTS

SRC = os.path.join(RESULTS, "meshcheck")


def main():
    out = dict(regenerated=False,
               why="needs SOFA (not run here) and the coarse/fine std meshes, which are not in the repository",
               what="coarse-vs-fine mesh difference on the older 3-channel std robot, gravity-free, 30 poses per set",
               std6_study="none exists: needs fine std6 meshes and SOFA runs with gravity (physics audit K2)",
               stored={})
    for n in (1, 2):
        f = f"meshcheck_std_{n}s_sym_meshcheck_fine_g0.json"
        out["stored"][f"{n}s"] = dict(json.load(open(os.path.join(SRC, f))), file=f"results/meshcheck/{f}")
    out["tip_mean_mm"] = [out["stored"]["1s"]["tip_mean"], out["stored"]["2s"]["tip_mean"]]
    json.dump(out, open(os.path.join(RESULTS, "mesh_error.json"), "w"), indent=1)
    print("stored old-robot mesh check (not regenerated): tip mean "
          + " / ".join(f"{v:.2f}" for v in out["tip_mean_mm"]) + " mm -> results/mesh_error.json")


if __name__ == "__main__":
    main()
