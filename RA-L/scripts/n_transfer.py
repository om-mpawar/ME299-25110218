"""
Does a learned correction trained on 1-2 modules carry to 4? Section count N as an input vs local physics features.

Physics (L0 + L1): the variable-section Euler-Bernoulli module with the identified compliance profile
(tdcr/compliance.py, scripts/module_compliance.py -> results/mc_std6.json), the real robot's routing and channel holes, pull-only tendons,
gravity in any direction. It composes to any N by construction.

Learned correction (L3, strain form): for each module, the residual of the local 5-mm segment increments (rotation and
translation in the cross-section frame, tdcr/backbones.to_intrinsic) between the FEM and the physics, compressed
to PCA modes and regressed by one GP (Matern 5/2, automatic relevance determination). The corrected increments are
chained from the base, so a module's correction acts wherever the modules below have turned it. Trained on 1- and
2-module FEM sets only; every module of every training pose is one sample. The variants differ only in the GP inputs:
  local    the module's local physics state from the physics solution: axial load from its own and from passing
           tendons, bending moment, its bend and end displacement, gravity in its base frame and the weight it carries
  local+N  the same plus the section count N and the module's position j in the stack
  naive-N  no physics features: the module's own tendon displacements, N, j and the gravity vector (scene frame)
N = 4 never occurs in training, so 'local+N' and 'naive-N' must extrapolate in N.

Two ways the correction enters (--coupling):
  output   the predicted increments are added to the physics shape afterwards
  inside   (the method plan's L3) the prediction is an eigenstrain: it changes the tendon path lengths, the pull-only
           tendons re-solve, and the eigenstrain is added to the re-solved strains. Training targets come from inverting
           each FEM residual through the linearised tendon response (invert_eigenstrain); the script prints how well the
           exact targets reproduce the training FEM (the floor of this coupling).
Status 19 Oct 2026 (smoke tests, old L1 from results/mc_grav.json, 300 module samples, 4 modes): exact inside targets
reproduce the training FEM to 0.2 mm, but GP-predicted eigenstrains degrade the fit through the tendon re-solve
(active-set changes); output coupling helps 1 module and hurts 4. Treat 'inside' as work in progress.

On hardware the N-dependence would also come from physics terms that the FEM does not contain (capstan friction,
joint compliance, tendon stretch, zero offsets: the FEM is frictionless, its cables inextensible, its joints glued),
so only L0 + L1 stands for the physics here.

Coverage: for every test set, the share of module samples with a local feature outside the training range, and the
median nearest-neighbour distance to the training features relative to the training set's own (1 = as dense as
training; standardised features).

    .venv/bin/python scripts/module_compliance.py --train std6_1s_sym_train_g0 std6_2s_sym_train_g0 \\
        std6_1s_sym_train std6_2s_sym_train --out results/mc_std6.json          # L1, once
    OMP_NUM_THREADS=8 .venv/bin/python scripts/n_transfer.py --mc results/mc_std6.json
"""
import argparse
import json
import os
import time
import warnings
warnings.simplefilter("ignore")

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from tdcr.backbones import ModesGP, evaluate, from_intrinsic, mirror, to_intrinsic
from tdcr.compliance import E_FEM, G_MAG, TESTS_STD6, ModuleModel, load_set
from tdcr.eb_prior import predict_gravity
from tdcr.params import MASS_STD
from tdcr.paths import ROOT, RESULTS

TRAIN = ["std6_1s_sym_train_g0", "std6_2s_sym_train_g0", "std6_1s_sym_train", "std6_2s_sym_train"]
SEG = 32                                   # 5-mm segments per 160-mm module
VARIANTS = ("local", "local+N", "naive-N")
COUPLINGS = ("inside", "output")


def physics(model, theta, d, dl=None):
    """L0 + L1 prediction for the commands dl (default d["dl"]) -> centreline, frames, tensions, prior object.
    The gravity fixed point is run to a tight tolerance so that finite differences through it are clean."""
    eb = model.build(theta, d["n"], d.get("routing", "shared"), d.get("az"), d.get("holes"))
    dl = d["dl"] if dl is None else dl
    if d.get("g") is None:
        C, R, lam = eb.predict(dl, d["stations"])
    else:
        C, R, lam, _ = predict_gravity(eb, dl, d["stations"], g_vec=d["g"], E=E_FEM, mass=d.get("mass", MASS_STD),
                                       iters=400, tol=1e-9)
    return C, R, lam, eb


def path_map(eb, n):
    """Linear map H (3n, n SEG 6): eigenstrain increments per 5-mm segment (rotation vector, translation, segment
    frame) -> path shortening of every tendon, as in the gravity fixed point of tdcr/eb_prior.py: compression plus
    bend times channel offset, over the segments the tendon passes (fraction of the segment's cells)."""
    nc = eb.through.shape[1]
    Tseg = eb.through.reshape(3 * n, n * SEG, nc // (n * SEG)).mean(-1)            # (3n, n SEG)
    H = np.zeros((3 * n, n * SEG, 6))
    H[:, :, 4] = -Tseg                                                             # extension lengthens the path
    H[:, :, 2] = -Tseg * eb.D[:, :1]                                               # kx = -w_z: towards +x
    H[:, :, 0] = Tseg * eb.D[:, 1:]                                                # kz = w_x: towards +z
    return H.reshape(3 * n, -1)


def response(model, theta, d, X0, eps):
    """Finite-difference response of the physics' segment increments to every tendon displacement:
    J (P, n SEG 6, 3n) = dX / d(dl). It contains how the tendons redistribute a change of path length."""
    n = d["n"]
    J = np.zeros((len(d["dl"]), X0.shape[1], 3 * n))
    for i in range(3 * n):
        dl = d["dl"].copy(); dl[:, i] += eps
        C, R, _, _ = physics(model, theta, d, dl)
        J[:, :, i] = (to_intrinsic(C, R).reshape(len(dl), -1) - X0) / eps
    return J


def invert_eigenstrain(r, J, H, rho, lever=100.0):
    """Eigenstrain eta whose effect THROUGH the physics reproduces the residual r: with the tendons reacting,
    X(dl - H eta) + eta ~ X0 + (I - J H) eta, so per pose eta = argmin |W ((I - J H) eta - r)|^2 + rho |eta|^2.
    W weights the rotation increments by a lever arm (mm per rad) so that they count as the displacement they cause
    downstream; unweighted, radians against millimetres lets the fit trade tip accuracy for translation increments.
    (I - J H) is singular in the directions taut tendons forbid (a change of a held tendon length): that part of r
    cannot come from any module strain (for one module with taut tendons it is the total bend, fixed by the commands),
    and the ridge keeps eta small there."""
    m = r.shape[1]
    w = np.tile(np.r_[lever, lever, lever, 1.0, 1.0, 1.0], m // 6)
    out = np.empty_like(r)
    for p in range(len(r)):
        A = (np.eye(m) - J[p] @ H) * w[:, None]
        out[p] = np.linalg.solve(A.T @ A + rho * np.eye(m), A.T @ (w * r[p]))
    return out


def module_features(d, C, R, lam, eb):
    """-> {variant: (P, n, F)} inputs of every module, from the physics solution (C, R, lam) of one set."""
    n, P = d["n"], len(d["dl"])
    g = np.zeros(3) if d.get("g") is None else np.asarray(d["g"], float) / G_MAG
    loc, nai = [], []
    for j in range(n):
        b, t = SEG * j, SEG * (j + 1)
        Rb = R[:, b]
        own, pas = lam[:, 3 * j:3 * j + 3].sum(1), lam[:, 3 * j + 3:].sum(1)
        M = lam[:, 3 * j:] @ eb.D[3 * j:]                                         # (P, 2) moment towards +x, +z / E
        rel = np.einsum("pji,pjl->pil", Rb, R[:, t])
        bend = Rotation.from_matrix(rel).as_rotvec()[:, [0, 2]]                    # bend of the module (rad)
        disp = np.einsum("pji,pj->pi", Rb, C[:, t] - C[:, b])                      # top relative to base, base frame
        gl = np.einsum("pji,j->pi", Rb, g)                                         # gravity in the module's base frame
        loc.append(np.column_stack([own, pas, M, bend, disp, gl, gl * (n - j)]))
        nai.append(np.column_stack([d["dl"][:, 3 * j:3 * j + 3], np.full(P, n), np.full(P, j + 1), np.tile(g, (P, 1))]))
    loc, nai = np.stack(loc, 1), np.stack(nai, 1)
    nj = np.stack([np.full((P, n), n), np.tile(np.arange(1, n + 1), (P, 1))], -1).astype(float)
    return {"local": loc, "local+N": np.concatenate([loc, nj], -1), "naive-N": nai}


def coverage(Ftr, Fte):
    """Share of test samples outside the training range in any feature; NN distance ratio (standardised; features
    that are constant in training, e.g. gravity in gravity-free data, are scaled by 1)."""
    mu, sd = Ftr.mean(0), Ftr.std(0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    A, B = (Ftr - mu) / sd, (Fte - mu) / sd
    out = ((B < A.min(0) - 1e-9) | (B > A.max(0) + 1e-9)).any(1).mean()
    tree = cKDTree(A)
    d_tr = tree.query(A, k=2)[0][:, 1]
    d_te = tree.query(B, k=1)[0]
    return float(out), float(np.median(d_te) / np.median(d_tr))


def stats(s):
    return dict(tip=float(s["tip"].mean()), tip95=float(np.percentile(s["tip"], 95)), tip_max=float(s["tip"].max()),
                rot=float(s["tip_rot"].mean()), along=float(s["mean_along"].mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc", default="results/mc_std6.json", help="identified module compliance (L1)")
    ap.add_argument("--train", nargs="+", default=TRAIN)
    ap.add_argument("--tests", nargs="+", default=TESTS_STD6)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--no-mirror", action="store_true", help="no mirror augmentation of the training sets")
    ap.add_argument("--max-train", type=int, default=0, help="random subset of training module samples (0 = all)")
    ap.add_argument("--min-poses", type=int, default=10, help="skip test sets with fewer settled poses")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--coupling", nargs="+", default=list(COUPLINGS), choices=COUPLINGS,
                    help="inside: the learned eigenstrain changes the tendon paths and the tendons re-solve (the method "
                         "plan's L3); output: added to the physics shape afterwards")
    ap.add_argument("--rho", type=float, default=10.0, help="ridge of the eigenstrain inversion (scale with lever^2)")
    ap.add_argument("--lever", type=float, default=100.0, help="weight of rotation increments in the inversion (mm/rad)")
    ap.add_argument("--eps", type=float, default=0.05, help="finite-difference step of the tendon response (mm)")
    ap.add_argument("--out", default="results/n_transfer.json")
    a = ap.parse_args()

    mc = json.load(open(os.path.join(ROOT, a.mc)))
    model, theta = ModuleModel.from_result(mc)
    if not all(t.startswith("std6_") for t in mc.get("train", [])):
        print(f"WARNING: {a.mc} was not identified on std6 sets (train: {mc.get('train')}); the physics is then the old "
              f"3-hole, shared-routing module")

    # training samples: every module of every pose of the 1-2 module sets
    t0 = time.time()
    Ftr = {v: [] for v in a.variants}
    Ytr = {c: [] for c in a.coupling}
    for tag in a.train:
        d = load_set(tag)
        if not a.no_mirror:
            d, _ = mirror(d)
        P, n = len(d["C"]), d["n"]
        C, R, lam, eb = physics(model, theta, d)
        F = module_features(d, C, R, lam, eb)
        r = (to_intrinsic(d["C"], d["R"]) - to_intrinsic(C, R)).reshape(P, -1)        # FEM minus physics
        for v in a.variants:
            Ftr[v].append(F[v].reshape(-1, F[v].shape[-1]))
        if "output" in a.coupling:
            Ytr["output"].append(r.reshape(P * n, -1))
        if "inside" in a.coupling:
            J = response(model, theta, d, to_intrinsic(C, R).reshape(P, -1), a.eps)
            eta = invert_eigenstrain(r, J, path_map(eb, n), a.rho, a.lever)
            Ytr["inside"].append(eta.reshape(P * n, -1))
            C1, R1, _, _ = physics(model, theta, d, d["dl"] - eta @ path_map(eb, n).T)
            Cc, Rc = from_intrinsic((to_intrinsic(C1, R1).reshape(P, -1) + eta).reshape(P, -1, 6), C[:, 0], R[:, 0])
            print(f"   eigenstrain targets {tag}: exact targets through the physics reproduce the FEM to "
                  f"{evaluate(Cc, Rc, d['C'], d['R'])['tip'].mean():.3f} mm tip (physics alone "
                  f"{evaluate(C, R, d['C'], d['R'])['tip'].mean():.3f}): the floor of the inside correction")
        print(f"train {tag:24s} {P:4d} poses{'' if a.no_mirror else ' (mirrored x2)'} -> {n * P} module samples "
              f"({time.time() - t0:.0f} s)")
    Ftr = {v: np.concatenate(f) for v, f in Ftr.items()}
    Ytr = {c: np.concatenate(y) for c, y in Ytr.items()}
    N = len(next(iter(Ftr.values())))
    if a.max_train and N > a.max_train:
        pick = np.random.default_rng(0).choice(N, a.max_train, replace=False)
        Ftr = {v: f[pick] for v, f in Ftr.items()}
        Ytr = {c: y[pick] for c, y in Ytr.items()}
        N = a.max_train
    print(f"{N} module samples, {SEG * 6} residual values each, {a.modes} modes; {time.time() - t0:.0f} s\n")

    gps = {}
    for c in a.coupling:
        for v in a.variants:
            t1 = time.time()
            gps[c, v] = ModesGP(Ytr[c], Ftr[v], a.modes)
            print(f"GP {c:6s} {v:8s} ({Ftr[v].shape[1]:2d} inputs) fitted in {time.time() - t1:4.0f} s: {gps[c, v].gp.kernel_}")
    res = dict(mc=a.mc, train=a.train, modes=a.modes, mirror=not a.no_mirror, n_train=int(N), rho=a.rho, lever=a.lever,
               tests={})
    head = f"{'test set':26s} {'poses':>5s}  {'model':24s} {'tip mm':>7s} {'95%':>6s} {'max':>6s} {'orient deg':>10s} {'along mm':>8s}"
    lines = [head, "-" * len(head)]
    for tag in a.tests:
        if not os.path.exists(os.path.join(RESULTS, f"backbone_{tag}.npz")):
            print(f"{tag}: no backbone file, skipped")
            continue
        d = load_set(tag)
        P, n = len(d["C"]), d["n"]
        if P < a.min_poses:
            print(f"{tag}: {P} poses (< {a.min_poses}), skipped")
            continue
        t1 = time.time()
        C, R, lam, eb = physics(model, theta, d)
        F = module_features(d, C, R, lam, eb)
        out_frac, nn_ratio = coverage(Ftr["local"] if "local" in Ftr else next(iter(Ftr.values())),
                                      F["local" if "local" in Ftr else next(iter(Ftr))].reshape(P * n, -1))
        r = dict(poses=P, n=n, coverage=dict(outside=out_frac, nn_ratio=nn_ratio),
                 models={"L0+L1": stats(evaluate(C, R, d["C"], d["R"]))})
        H = path_map(eb, n)
        for (c, v), gp in gps.items():
            eta = gp.predict(F[v].reshape(P * n, -1)).reshape(P, -1)
            if c == "output":
                X = to_intrinsic(C, R).reshape(P, -1) + eta
            else:                                                     # tendons react to the eigenstrain's path change
                C1, R1, _, _ = physics(model, theta, d, d["dl"] - eta @ H.T)
                X = to_intrinsic(C1, R1).reshape(P, -1) + eta
            Cc, Rc = from_intrinsic(X.reshape(P, -1, 6), C[:, 0], R[:, 0])
            r["models"][f"+L3 {v} ({c})"] = stats(evaluate(Cc, Rc, d["C"], d["R"]))
        res["tests"][tag] = r
        for k, (name, s) in enumerate(r["models"].items()):
            lines.append(f"{tag if k == 0 else '':26s} {P if k == 0 else '':>5}  {name:24s} "
                         f"{s['tip']:7.2f} {s['tip95']:6.2f} {s['tip_max']:6.2f} {s['rot']:10.2f} {s['along']:8.2f}")
        lines.append(f"{'':26s} {'':>5}  coverage: {100 * out_frac:.0f} % of module samples outside the training range, "
                     f"NN distance x{nn_ratio:.1f} of training")
        print(f"{tag}: done in {time.time() - t1:.0f} s")
    print("\n" + "\n".join(lines))
    json.dump(res, open(os.path.join(ROOT, a.out), "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
