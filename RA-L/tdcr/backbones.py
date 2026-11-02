"""
Fast model study on cached backbones: which prior, representation, inputs and regressor give
the most accurate held-out shape for 2 and 4 sections?

Reads the compact backbone files that shape_analysis.py writes (results/backbone_<tag>.npz:
centreline C and cross-section rotations R at material stations every 5 mm), plus the tendon
pulls from the matching shape file, so nothing is re-extracted from the meshes.

Representations of the shape (what the regressor predicts, as PCA modes + GP):
  global     residual C - C_prior in world axes (the shape_analysis.py baseline)
  intrinsic  per 5-mm segment: rotation increment log(R_k^T R_k+1) and translation increment
             R_k^T (C_k+1 - C_k), both in the LOCAL cross-section frame, minus the prior's;
             the backbone is rebuilt by chaining the segments from the base. A section's local
             bending depends only on the tendons that pass through it, wherever the sections
             below have turned it, so this is a much simpler function of the inputs.

    OMP_NUM_THREADS=2 .venv/bin/python analysis/model_study.py --tag std_2s_sym_g0 --L 160 --r 10.55 \
        --variants global:cc:dl intrinsic:mech:dl intrinsic:mech:dlq
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tdcr`
import argparse
import os
import time
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from tdcr.paths import ROOT


# ── data ──────────────────────────────────────────────────────────────────

def load(tag, max_settle=0.01):
    b = np.load(os.path.join(ROOT, "results", f"backbone_{tag}.npz"))
    s = b if "dl" in b.files else np.load(os.path.join(ROOT, "results", f"shape_{tag}.npz"))
    th = np.asarray(b["theta"], float)
    n = 1 if th.ndim == 1 else th.shape[1]
    d = dict(C=np.asarray(b["C"], float), R=np.asarray(b["R"], float), theta=th,
             alpha=np.asarray(b["alpha"], float), dl=np.asarray(s["dl"], float),
             stations=np.asarray(b["stations"], float), n=n)
    # poses that hit the step cap still moving (a 4s pose snapping) are not equilibria: drop them
    settle = np.asarray(b["settle_mm"], float) if "settle_mm" in b.files else np.zeros(len(th))
    keep = ~(settle > max_settle)
    if not keep.all():
        print(f"[load] {tag}: dropped {int((~keep).sum())} unsettled poses (moving > {max_settle} mm at the end)")
        d = {k: (v[keep] if k not in ("stations", "n") else v) for k, v in d.items()}
    return d


def mirror_perm(n, az=None):
    """Tendon permutation of the mirror z -> -z (scene frame): tendon k at azimuth psi takes the command of the tendon
    at -psi. az: (n, 3) azimuths in the scene frame (deg); None = every section at 0/120/240 (tendons 2 <-> 3). For the
    real robot's sections 3-4 (scene 300/180/60) this swaps tendons 1 <-> 3 instead."""
    if az is None:
        return np.concatenate([[3 * j, 3 * j + 2, 3 * j + 1] for j in range(n)])
    az = np.asarray(az, float).reshape(n, 3)
    perm = []
    for j in range(n):
        for k in range(3):
            diff = np.abs(((az[j] + az[j, k]) + 180.0) % 360.0 - 180.0)     # |az_j,m - (-az_j,k)| on the circle
            m = int(np.argmin(diff))
            if diff[m] > 1e-6:
                raise ValueError(f"section {j + 1}: no tendon at the mirror image of {az[j, k]:g} deg")
            perm.append(3 * j + m)
    return np.array(perm)


def mirror(d):
    """Exact mirror z -> -z: alpha -> -alpha, each tendon swapped with the one at the mirrored azimuth (from d["az"]
    when present, else 120 <-> 240 in every section). Sets with gravity must have no z component (upright, hanging,
    lab x), which the mirror leaves unchanged. -> (data, groups)."""
    M = np.diag([1.0, 1.0, -1.0])
    n = d["n"]
    perm = mirror_perm(n, d.get("az"))
    if d.get("g") is not None and abs(d["g"][2]) > 0:
        raise ValueError("mirror: gravity with a z component changes under the mirror; one set holds one gravity vector")
    P = len(d["C"])
    out = dict(d)
    out["C"] = np.concatenate([d["C"], d["C"] * np.array([1.0, 1.0, -1.0])])
    out["R"] = np.concatenate([d["R"], np.einsum("ij,pkjl,lm->pkim", M, d["R"], M)])
    out["theta"] = np.concatenate([d["theta"], d["theta"]])
    out["alpha"] = np.concatenate([d["alpha"], (-d["alpha"]) % (2 * np.pi)])
    out["dl"] = np.concatenate([d["dl"], d["dl"][:, perm]])
    return out, np.concatenate([np.arange(P), np.arange(P)])


# ── priors ────────────────────────────────────────────────────────────────

def prior_backbone(d, kind, L, r, mu, base=0.0, top=0.0):
    from analysis.shape_analysis import cc_backbone  # noqa
    if kind == "cc":
        return cc_backbone(d["theta"], d["alpha"], d["stations"])
    if kind == "mech":
        from tdcr import mechanics_prior as mp
        q = mp.solve(d["dl"], d["n"], r, mu)
        d["q"] = q
        return mp.backbone(q, d["stations"], L, base, top)
    if kind == "none":        # straight
        P, K = len(d["C"]), len(d["stations"])
        pts = np.zeros((P, K, 3)); pts[:, :, 1] = d["stations"]
        return pts, np.tile(np.eye(3), (P, K, 1, 1))
    raise ValueError(kind)


# ── representations ───────────────────────────────────────────────────────

def to_intrinsic(C, R):
    """(P, K, 3), (P, K, 3, 3) -> (P, K-1, 6): local rotation and translation increments,
    plus the base pose (C0, R0) needed to rebuild."""
    rel = np.einsum("pkji,pkjl->pkil", R[:, :-1], R[:, 1:])
    P, K1 = rel.shape[:2]
    w = Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(P, K1, 3)
    t = np.einsum("pkji,pkj->pki", R[:, :-1], C[:, 1:] - C[:, :-1])
    return np.concatenate([w, t], axis=2)


def from_intrinsic(X, C0, R0):
    P, K1 = X.shape[:2]
    C = np.zeros((P, K1 + 1, 3)); R = np.zeros((P, K1 + 1, 3, 3))
    C[:, 0], R[:, 0] = C0, R0
    E = Rotation.from_rotvec(X[:, :, :3].reshape(-1, 3)).as_matrix().reshape(P, K1, 3, 3)
    for k in range(K1):
        C[:, k + 1] = C[:, k] + np.einsum("pij,pj->pi", R[:, k], X[:, k, 3:])
        R[:, k + 1] = np.einsum("pij,pjk->pik", R[:, k], E[:, k])
    return C, R


# ── regressors ────────────────────────────────────────────────────────────

def gp_fit_predict(Xtr, Ytr, Xte, nu=2.5, restarts=2, per_output=True):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
    base = (RBF(np.ones(Xtr.shape[1]), (1e-2, 1e3)) if nu == np.inf
            else Matern(np.ones(Xtr.shape[1]), (1e-2, 1e3), nu=nu))
    def make():
        return GaussianProcessRegressor(ConstantKernel(1.0, (1e-3, 1e3)) * base
                                        + WhiteKernel(1e-4, (1e-9, 1e0)), normalize_y=True,
                                        n_restarts_optimizer=restarts, random_state=0)
    if not per_output:
        return make().fit(Xtr, Ytr).predict(Xte)
    return np.column_stack([make().fit(Xtr, Ytr[:, j]).predict(Xte) for j in range(Ytr.shape[1])])


def modes_fit_predict(Ytr, Ftr, Fte, n_modes, **gp):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    mean = Ytr.mean(0)
    sd = Ytr.std(0) + 1e-12
    pca = PCA(n_components=min(n_modes, len(Ytr) - 1)).fit((Ytr - mean) / sd)
    Z = pca.transform((Ytr - mean) / sd)
    sc = StandardScaler().fit(Ftr)
    Zp = gp_fit_predict(sc.transform(Ftr), Z, sc.transform(Fte), **gp)
    return mean + (Zp @ pca.components_) * sd


# ── one variant ───────────────────────────────────────────────────────────

def features(d, kind):
    out = []
    for k in kind.split("+"):
        if k == "dl":
            out.append(d["dl"])
        elif k == "q":              # mechanics-prior configuration (bend vectors, shortening)
            out.append(d["q"].reshape(len(d["dl"]), -1))
        elif k == "cmd":
            out.append(np.column_stack([np.column_stack([d["theta"][:, j] * np.cos(d["alpha"][:, j]),
                                                         d["theta"][:, j] * np.sin(d["alpha"][:, j])])
                                        for j in range(d["n"])]))
        else:
            raise ValueError(k)
    return np.hstack(out)


def evaluate(Cp, Rp, C, R):
    e = np.linalg.norm(Cp - C, axis=2)
    rel = np.einsum("pji,pjl->pil", Rp[:, -1], R[:, -1])
    rot = np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))
    return dict(tip=e[:, -1], mean_along=e.mean(1), max_along=e.max(1), tip_rot=rot)


def run_variant(d, groups, rep, prior, feat, L, r, mu, n_modes, folds=5, base=0.0, top=0.0, **gp):
    from sklearn.model_selection import GroupKFold
    d = dict(d)
    Pp, Rpr = prior_backbone(d, prior, L, r, mu, base, top)
    if "q" in feat and "q" not in d:
        from tdcr import mechanics_prior as mp
        d["q"] = mp.solve(d["dl"], d["n"], r, mu)
    F = features(d, feat)
    C, R = d["C"], d["R"]
    P = len(C)
    if rep == "global":
        Y = (C - Pp).reshape(P, -1)
        W = np.einsum("pkji,pkjl->pkil", Rpr, R)
        Yw = Rotation.from_matrix(W.reshape(-1, 3, 3)).as_rotvec().reshape(P, -1)
        Y = np.hstack([Y, Yw])
    else:
        Y = (to_intrinsic(C, R) - to_intrinsic(Pp, Rpr)).reshape(P, -1)
    Yhat = np.zeros_like(Y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in GroupKFold(folds).split(F, groups=groups):
            Yhat[te] = modes_fit_predict(Y[tr], F[tr], F[te], n_modes, **gp)
    K = C.shape[1]
    if rep == "global":
        Cp = Pp + Yhat[:, :3 * K].reshape(P, K, 3)
        Wh = Rotation.from_rotvec(Yhat[:, 3 * K:].reshape(-1, 3)).as_matrix().reshape(P, K, 3, 3)
        Rp = np.einsum("pkij,pkjl->pkil", Rpr, Wh)
    else:
        X = to_intrinsic(Pp, Rpr) + Yhat.reshape(P, K - 1, 6)
        Cp, Rp = from_intrinsic(X, C[:, 0], R[:, 0])
    return evaluate(Cp, Rp, C, R), evaluate(Pp, Rpr, C, R)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--L", type=float, required=True, help="section length, mm")
    ap.add_argument("--r", type=float, required=True, help="mech prior bearing radius, mm")
    ap.add_argument("--mu", type=float, default=10.0)
    ap.add_argument("--base", type=float, default=0.0, help="mech prior: rigid length at each section's base, mm")
    ap.add_argument("--top", type=float, default=0.0, help="mech prior: rigid length at each section's top, mm")
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--nu", type=float, default=2.5)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--joint", action="store_true", help="one GP for all modes (shared kernel)")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--variants", nargs="+", default=["global:cc:dl", "intrinsic:mech:dl"],
                    help="rep:prior:features, e.g. intrinsic:mech:dl+q")
    args = ap.parse_args()
    d = load(args.tag)
    if args.subset:
        keep = np.sort(np.random.default_rng(1).permutation(len(d["C"]))[:args.subset])
        d = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(d["C"]) and k != "stations" else v)
             for k, v in d.items()}
    if args.no_mirror:
        groups = np.arange(len(d["C"]))
    else:
        d, groups = mirror(d)
    print(f"{args.tag}: {len(d['C'])} poses ({d['n']} sections), {len(d['stations'])} stations, "
          f"modes {args.modes}, nu {args.nu}", flush=True)
    for v in args.variants:
        rep, prior, feat = v.split(":")
        t = time.time()
        s, s0 = run_variant(d, groups, rep, prior, feat, args.L, args.r, args.mu, args.modes,
                            base=args.base, top=args.top, nu=args.nu, restarts=args.restarts, per_output=not args.joint)
        print(f"{v:28s} prior only: tip {s0['tip'].mean():6.2f} mm  {s0['tip_rot'].mean():5.2f} deg | "
              f"learned: tip {s['tip'].mean():6.2f} mm (median {np.median(s['tip']):5.2f}, 95% "
              f"{np.percentile(s['tip'], 95):5.2f})  mean-along {s['mean_along'].mean():5.2f}  "
              f"tip rot {s['tip_rot'].mean():5.2f} deg  [{time.time() - t:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
