"""
FEM backbones: loading, mirroring, the intrinsic (local-increment) shape representation, error metrics, the PCA + GP
residual regressor, and the extraction of backbones from SOFA node positions.

Backbone files (results/backbone_<tag>.npz, written by scripts/extract_backbones.py from a SOFA pose directory):
centreline C and cross-section rotations R at material stations every 5 mm, the tendon displacements dl, and the
commanded bends theta / alpha (metadata).

Intrinsic representation (to_intrinsic / from_intrinsic): per 5-mm segment, the rotation increment log(R_k^T R_k+1)
and the translation increment R_k^T (C_k+1 - C_k), both in the LOCAL cross-section frame; the backbone is rebuilt by
chaining the segments from the base. A module's local bending depends only on the tendons that pass through it,
wherever the modules below have turned it.

Extraction (backbone): the body is cut into slabs at the stations (arc length at rest); for each slab the rigid
motion that best maps its rest nodes to its deformed nodes is found (Kabsch). Applied to the slab's axis point
(0, s_k, 0) it gives the centreline point c(s_k); its rotation gives the cross-section orientation R(s_k). Stations
are MATERIAL points, so axial compression shows up as the backbone ending short, as it physically does.
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation

from tdcr.params import STD6
from tdcr.paths import MESHES, RESULTS

L = STD6.module_length_mm                          # one module (section)
STATIONS = np.arange(0.0, L + 1e-9, 5.0)         # material stations every 5 mm, base to tip (1 module)
HALF_SLAB = 2.5


# ── data ──────────────────────────────────────────────────────────────────

def load(tag, max_settle=0.01):
    b = np.load(os.path.join(RESULTS, f"backbone_{tag}.npz"))
    s = b if "dl" in b.files else np.load(os.path.join(RESULTS, f"shape_{tag}.npz"))
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


# ── representations and metrics ───────────────────────────────────────────

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


def evaluate(Cp, Rp, C, R):
    e = np.linalg.norm(Cp - C, axis=2)
    rel = np.einsum("pji,pjl->pil", Rp[:, -1], R[:, -1])
    rot = np.degrees(np.linalg.norm(Rotation.from_matrix(rel).as_rotvec(), axis=1))
    return dict(tip=e[:, -1], mean_along=e.mean(1), max_along=e.max(1), tip_rot=rot)


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


class ModesGP:
    """PCA modes of the standardised residuals + one GP over all modes (as modes_fit_predict),
    fitted once and reused for every test set."""

    def __init__(self, Y, F, n_modes, nu=2.5, restarts=1):
        from sklearn.decomposition import PCA
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        from sklearn.preprocessing import StandardScaler
        self.mean, self.sd = Y.mean(0), Y.std(0) + 1e-12
        self.pca = PCA(n_components=min(n_modes, len(Y) - 1)).fit((Y - self.mean) / self.sd)
        Z = self.pca.transform((Y - self.mean) / self.sd)
        self.sc = StandardScaler().fit(F)
        k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(F.shape[1]), (1e-2, 1e3), nu=nu) + WhiteKernel(1e-4, (1e-9, 1e0))
        self.gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=restarts, random_state=0)
        self.gp.fit(self.sc.transform(F), Z)

    def predict(self, F):
        return self.mean + (self.gp.predict(self.sc.transform(F)) @ self.pca.components_) * self.sd


# ── extraction from SOFA node positions ───────────────────────────────────

def stations_for(n):
    """Material stations every 5 mm along n sections: 45 for one, 89 for two, ..."""
    return np.arange(0.0, n * L + 1e-9, 5.0)


def node_sections(mesh_name, n_nodes):
    """Section index of every mesh node, from the mesh title ("offsets=" or "offset=")."""
    path = os.path.join(MESHES, str(mesh_name))
    offsets = [0]
    if os.path.exists(path):
        with open(path) as f:
            f.readline()
            for tok in f.readline().split():
                if tok.startswith("offsets="):
                    offsets = [int(v) for v in tok.split("=")[1].split(",")]
                elif tok.startswith("offset=") and len(offsets) == 1:
                    offsets = [0, int(tok.split("=")[1])]
    body = np.zeros(n_nodes, dtype=int)
    for k, off in enumerate(offsets[1:], start=1):
        body[off:] = k
    return body


def kabsch(P, Q):
    """R, t minimising |R P + t - Q| over rows (rigid, no reflection)."""
    mp, mq = P.mean(0), Q.mean(0)
    H = (P - mp).T @ (Q - mq)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def backbone(rest, pos, stations=STATIONS, body=None):
    """-> centreline (P, K, 3) and cross-section rotations (P, K, 3, 3).

    With several sections (body = section index per node), each station is fitted
    only to its own section's nodes, so a slab at a joint never mixes two bodies.
    """
    y = rest[:, 1]
    if body is None:
        body = np.zeros(len(rest), dtype=int)
    nb = body.max() + 1
    sec = [min(int(np.ceil(s / L)) - 1, nb - 1) if s > 0 else 0 for s in stations]
    slabs = [np.where((np.abs(y - s) <= HALF_SLAB) & (body == j))[0] for s, j in zip(stations, sec)]
    assert min(len(i) for i in slabs) >= 10, "a slab has too few nodes for a stable fit"
    P = pos.shape[0]
    C = np.zeros((P, len(stations), 3)); Rs = np.zeros((P, len(stations), 3, 3))
    for p in range(P):
        for k, idx in enumerate(slabs):
            R, t = kabsch(rest[idx], pos[p, idx])
            C[p, k] = R @ np.array([0.0, stations[k], 0.0]) + t
            Rs[p, k] = R
    return C, Rs
