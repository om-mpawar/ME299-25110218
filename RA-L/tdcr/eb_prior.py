"""
Variable-section Euler-Bernoulli prior for the standard rolling-disc notched module (no fitted parameters).

The module is treated as an Euler-Bernoulli beam whose cross-section varies along the axis exactly as the CAD
cuts it: a bored tube (OD 25 / ID 14) with two orthogonal families of hourglass notches (cut by a disc of
diameter D_RD, each leaving a bridge of half-width e), a plain base sleeve and a solid cap. For every height y the
remaining material gives the area A(y) and the second moments Jx(y) = int x^2 dA (bending towards +-x) and
Jz(y) = int z^2 dA (bending towards +-z).

Statics (tendon loading only, straight parallel channels): a tendon of tension T at channel offset d applies the
moment T d and the axial force T on every cross-section it passes (Camarillo et al. 2008/2009; Rucker & Webster
2011). Hence, with tensions lambda_i,
    curvature   kappa_x(y) = sum_i lambda_i d_ix / (E Jx(y)),   kappa_z(y) = sum_i lambda_i d_iz / (E Jz(y)),
    compression eps(y)     = sum_i lambda_i / (E A(y)),
summed over the tendons that pass y. Tendon i's path shortening is linear in the tensions,
    Delta l_i = sum_j G_ij lambda_j,   G_ij = int_{both paths} [1/A + d_ix d_jx / Jx + d_iz d_jz / Jz] dy / E,
and pull-only tendons give the complementarity problem  lambda >= 0,  G lambda >= dl,  lambda . (G lambda - dl) = 0,
solved as the convex QP  min 1/2 lambda^T G lambda - dl^T lambda, lambda >= 0 (non-negative least squares).
Young's modulus cancels without gravity. The backbone frames follow by integrating kappa and eps along y.

Friction (optional, predict_gravity mu > 0): capstan friction between a tendon and its channel walls. Tendon i
carries lambda_i f_i(y) with f_i = exp(-sigma_i mu Theta_i(y)), Theta_i the backbone turning accumulated along its
path from the base and sigma_i = +1 while the servo winds it in (tension falls towards the tip), -1 while it pays out
(tension rises towards the tip). The path shortening stays G-like, Delta l_i = sum_j M_ij lambda_j with
M_ij = int_paths f_j [1/A + d_i C d_j] dy / E, but M is no longer symmetric: the pull-only problem is a linear
complementarity problem, solved exactly by principal pivoting (lcp_pull_only). Constant wrap outside the backbone (the
base holes, the servo pulley) only scales every tension alike and is absorbed by the calibrated stiffness.

Frame: SOFA, backbone axis +y, base at the origin, tendon channel k at (r cos psi_k, ., r sin psi_k).
"""
import json
import hashlib
import math
import os

import numpy as np
from scipy.optimize import nnls

from tdcr.params import MASS_STD, STD6
from tdcr.paths import CAD, RESULTS

CAD_JSON = os.path.join(CAD, "std_tdcr_L150.json")   # std6 geometry: the same module; its holes are cut via `holes`
CACHE = os.path.join(RESULTS, "eb_profile_std.npz")
PSI = np.radians([0.0, 120.0, 240.0])
R_BEAR = STD6.bearing_radius_mm   # tendon bearing radius: cable points in the channel wall (data/cables/std_*)
Y_START, Y_END = 5.0, 155.0        # tendon path within its section (cable files: first / last point)


def timoshenko_factor(nu=STD6.poisson, ri=STD6.inner_radius_mm, ro=STD6.outer_radius_mm):
    """E / (kappa G) for the bored tube: 2 (1 + nu) / kappa, with Cowper's shear coefficient of a hollow circle
    kappa = 6 (1 + nu) (1 + m^2)^2 / ((7 + 6 nu) (1 + m^2)^2 + (20 + 12 nu) m^2), m = ri / ro.
    Source: Cowper, J. Appl. Mech. 1966 (refs.bib: cowper1966)."""
    m2 = (ri / ro) ** 2
    kappa = 6 * (1 + nu) * (1 + m2) ** 2 / ((7 + 6 * nu) * (1 + m2) ** 2 + (20 + 12 * nu) * m2)
    return 2 * (1 + nu) / kappa


def section_profile(dy=0.02, dc=0.1, force=False, holes=None, hole_r=STD6.tendon_radius_mm, hole_dia=STD6.hole_dia_mm,
                    cad_json=None):
    """-> y (cell centres over one module), A, Jx, Jz (mm^2, mm^4) from the CAD parameters.
    holes: channel azimuths in the module frame (deg), removed from the section as round holes of hole_dia at radius
    hole_r; None = no holes (the profile used for every result before 10 Oct 2026)."""
    with open(CAD_JSON if cad_json is None else cad_json) as f:
        g = json.load(f)
    # Include CAD contents and integration settings, not just channel angles.
    # Editing a design in place must never reuse another backbone's compliance.
    spec = dict(version=1, cad=g, dy=dy, dc=dc, holes=None if holes is None else list(holes),
                hole_r=hole_r, hole_dia=hole_dia)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    cache = CACHE.replace(".npz", "_" + digest + ".npz")
    if os.path.exists(cache) and not force:
        d = np.load(cache)
        return d["y"], d["A"], d["Jx"], d["Jz"]
    R, ri, rd, e = g["outer_dia"] / 2, g["inner_dia"] / 2, g["disc_dia"] / 2, g["offset"]
    H = g["total_height_mm"]
    ca, cb = np.array(g["notch_centres_A"]), np.array(g["notch_centres_B"])
    xs = np.arange(-R + dc / 2, R, dc)
    X, Z = np.meshgrid(xs, xs, indexing="ij")
    rho = np.hypot(X, Z)
    ring = (rho >= ri) & (rho <= R)
    for h in (holes or ()):
        ring &= np.hypot(X - hole_r * math.cos(math.radians(h)), Z - hole_r * math.sin(math.radians(h))) >= hole_dia / 2
    x, z = X[ring], Z[ring]
    # half-height of the hourglass cut at transverse coordinate w (cut only where w > e)
    hz = np.where(np.abs(z) > e, rd - np.sqrt(np.maximum(rd ** 2 - z ** 2, 0)), -1.0)   # family A: bridge |z| < e
    hx = np.where(np.abs(x) > e, rd - np.sqrt(np.maximum(rd ** 2 - x ** 2, 0)), -1.0)   # family B: bridge |x| < e
    y = np.arange(dy / 2, H, dy)
    A, Jx, Jz = np.empty(len(y)), np.empty(len(y)), np.empty(len(y))
    da = dc * dc
    full = (len(x) * da, (x ** 2).sum() * da, (z ** 2).sum() * da)
    for i0 in range(0, len(y), 400):
        yy = y[i0:i0 + 400]
        dA = np.abs(yy[:, None] - ca[None]).min(1)
        dB = np.abs(yy[:, None] - cb[None]).min(1)
        near = (dA < rd) | (dB < rd)
        for k, yk in enumerate(yy):
            if not near[k]:
                A[i0 + k], Jx[i0 + k], Jz[i0 + k] = full
                continue
            keep = ~((dA[k] < hz) | (dB[k] < hx))
            A[i0 + k] = keep.sum() * da
            Jx[i0 + k] = (x[keep] ** 2).sum() * da
            Jz[i0 + k] = (z[keep] ** 2).sum() * da
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, y=y, A=A, Jx=Jx, Jz=Jz)
    return y, A, Jx, Jz


def _rodrigues(w):
    """Rotation vectors (..., 3) -> matrices (..., 3, 3)."""
    ang = np.linalg.norm(w, axis=-1)
    a = w / np.where(ang < 1e-15, 1.0, ang)[..., None]
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -a[..., 2], a[..., 1]
    K[..., 1, 0], K[..., 1, 2] = a[..., 2], -a[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -a[..., 1], a[..., 0]
    s, c = np.sin(ang)[..., None, None], np.cos(ang)[..., None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def routing_angles(n, routing="shared"):
    """-> module rotations (deg, n) and tendon azimuths (deg, n x 3) in the base frame.
    shared:   every section uses the channels at 0/120/240 deg (all SOFA data before 10 Oct 2026).
    paired60: the physical robot (Om, 10 Oct 2026). Six channels every 60 deg in every module, all modules in the
              same orientation (notches aligned); the tendons of sections 1-2 use 0/120/240, those of sections 3-4
              use 60/180/300, running through the free channels of sections 1-2.
    Angles are in the frame the caller works in (the cable files give them in the SOFA frame)."""
    rot = np.zeros(n)
    if routing == "shared":
        off = np.zeros(n)
    elif routing == "paired60":
        off = np.array([60.0 * ((s // 2) % 2) for s in range(n)])
    else:
        raise ValueError(routing)
    return rot, off[:, None] + np.array([0.0, 120.0, 240.0])[None]


def geometry_from_cables(path):
    """Tendon azimuths (deg, n x 3, SOFA frame) straight from a cable file: the single source of truth for the
    routing of a simulated robot. Module rotations are zero (notches aligned in every robot built so far)."""
    d = json.load(open(path))
    n = max(t["section"] for t in d["tendons"])
    az = np.zeros((n, 3))
    count = [0] * n
    for t in d["tendons"]:
        s = t["section"] - 1
        az[s, count[s]] = t["azimuth_deg"]
        count[s] += 1
    return np.zeros(n), az


class EBPrior:
    """n stacked modules. cell: integration cell length (mm) for the frames; compliances are integrated exactly
    from the fine profile within each cell.
    routing / module_rot_deg / tendon_az_deg: geometry of the stack (see routing_angles); explicit angles override
    routing. Module s is rotated about the axis by module_rot_deg[s] (its notch families rotate with it); the bending
    compliance of its cells is then a 2x2 tensor in the base frame. holes: channel azimuths cut from the cross-section
    (module frame), see section_profile. gap: straight rigid length (mm) between consecutive modules (the printed
    coupler is gap mm thicker than the web the module length L already includes, see set_gap)."""

    def __init__(self, n, cell=0.5, r_bear=R_BEAR, L=STD6.module_length_mm, axial_scale=1.0, routing="shared",
                 module_rot_deg=None, tendon_az_deg=None, holes=None, gap=0.0, cad_json=None):
        self.n, self.L, self.cell = n, L, cell
        self.axial_scale_used = axial_scale
        rot, az = routing_angles(n, routing)
        if module_rot_deg is not None:
            rot = np.asarray(module_rot_deg, float)
        if tendon_az_deg is not None:
            az = np.asarray(tendon_az_deg, float).reshape(n, 3)
        self.module_rot_deg, self.tendon_az_deg = rot, az
        yf, A, Jx, Jz = section_profile(holes=holes, cad_json=cad_json)
        dyf = yf[1] - yf[0]
        if not np.isclose(yf[-1] + dyf / 2, L):
            raise ValueError("CAD module height must match L; length transfer requires matching routing and stations")
        m = int(round(cell / dyf))
        nc = len(yf) // m
        # per cell: integral of 1/A, 1/Jx, 1/Jz (mm^-1, mm^-3), and its centre
        cA = axial_scale * (dyf / A[:nc * m]).reshape(nc, m).sum(1)   # axial_scale: wave-spring compliance of the discs
        cX = (dyf / Jx[:nc * m]).reshape(nc, m).sum(1)
        cZ = (dyf / Jz[:nc * m]).reshape(nc, m).sum(1)
        cT = (2.0 * (1.0 + STD6.poisson) * dyf / (Jx + Jz)[:nc * m]).reshape(nc, m).sum(1)   # integral of E / (G J_p)
        self.cA, self.cX, self.cZ = np.tile(cA, n), np.tile(cX, n), np.tile(cZ, n)
        self.cA_cad = self.cA / axial_scale                            # plane-section integral of 1/A (mass distribution)
        # shear compliance per cell (integral of 1 / (kappa G A) times E, mm^-1): zero = Euler-Bernoulli (every result
        # before 19 Oct 2026). Only external transverse loads (gravity across the backbone) shear the cells: tendon
        # loads carry no shear force (Camarillo 2008, Sec. II), so the tendon map G and the pull-only QP are unchanged.
        self.cS = np.zeros_like(self.cA)
        # torsion compliance per cell (integral of 1 / (G J_p) times E, mm^-3): zero = rigid in twist (every result
        # before 3 Nov 2026). cT_plane is the plane-section value; ModuleModel sets cT = factor x cT_plane, the factor
        # identified from a SOFA twist test (sim/twist_test.py, scripts/torsion_identify.py). Only the twisting part
        # of the gravity moment loads it: tendon loads twist the backbone only at second order.
        self.cT_plane = np.tile(cT, n)
        self.cT = np.zeros_like(self.cA)
        self.yc = np.concatenate([(np.arange(nc) + 0.5) * cell + j * L for j in range(n)])
        self.edges = np.concatenate([[0.0], np.concatenate([(np.arange(1, nc + 1)) * cell + j * L for j in range(n)])])
        azr = np.radians(az.ravel())
        self.D_unit = np.stack([np.cos(azr), np.sin(azr)], 1)                                 # (3n, 2) x, z, base frame
        self.r_bear0 = r_bear
        self.set_arm(r_bear)
        self.phi = np.repeat(np.radians(rot), nc)                                             # module rotation per cell
        # tendon i = 3 s + k passes cells whose centre lies in [Y_START, Y_END + s L]
        self.through = np.array([(self.yc >= Y_START) & (self.yc <= Y_END + (i // 3) * L) for i in range(3 * n)])
        # tendon stretch (mm of servo travel per unit lambda = tension / E), added to diag(G): the servo displacement is
        # the path shortening plus the stretch, so the map stays symmetric positive definite and the QP convex
        self.tendon_compliance = np.zeros(3 * n)
        self.set_gap(gap)
        self.assemble()

    def set_arm(self, r_bear):
        """Tendon moment arm (mm from the backbone axis) of every tendon; call assemble() afterwards."""
        self.D = r_bear * self.D_unit
        self.d = self.D[:3]

    def set_gap(self, gap):
        """Rigid straight step of gap mm after the last cell of every module but the last, along that cell's end frame.
        Stations stay material coordinates (mm of module, 0..n L): a module's end station is the edge before its gap.
        The gap neither bends nor shortens, so tendon paths and G are unchanged (only the tendon stretch grows, see
        ModuleModel.build)."""
        per_mod = len(self.yc) // self.n
        self.gap_after = np.zeros(len(self.yc))
        self.gap_after[[(j + 1) * per_mod - 1 for j in range(self.n - 1)]] = gap

    def assemble(self):
        """Bending compliance tensor per cell in the base frame (module axes rotated by phi), then G (plus the tendon
        stretch on its diagonal) and its Cholesky factor. Call again after changing cA, cX, cZ or tendon_compliance.
        Source: the coupled tendon compliance map of Camarillo et al., T-RO 2008 (refs.bib: camarillo2008)."""
        c, s = np.cos(self.phi), np.sin(self.phi)
        self.cXX = self.cX * c * c + self.cZ * s * s
        self.cZZ = self.cX * s * s + self.cZ * c * c
        self.cXZ = (self.cX - self.cZ) * s * c
        T = self.through.astype(float)
        Dx, Dz = self.D[:, 0], self.D[:, 1]
        self.G = (T * self.cA) @ T.T + np.outer(Dx, Dx) * ((T * self.cXX) @ T.T) \
            + np.outer(Dz, Dz) * ((T * self.cZZ) @ T.T) + (np.outer(Dx, Dz) + np.outer(Dz, Dx)) * ((T * self.cXZ) @ T.T)
        self.G = self.G + np.diag(self.tendon_compliance)
        self.Lc = np.linalg.cholesky(self.G)

    def bend(self, Mx, Mz):
        """Integrated curvatures per cell from the moments bending towards +x and +z (divided by E)."""
        return self.cXX * Mx + self.cXZ * Mz, self.cXZ * Mx + self.cZZ * Mz

    def tensions(self, dl):
        """dl (P, 3n) -> lambda (P, 3n) >= 0 (tension / E, mm^2): pull-only complementarity via NNLS.
        Source: slack in the coupled tendon forward map, Camarillo et al., T-RO 2009 (refs.bib: camarillo2009)."""
        dl = np.atleast_2d(dl)
        b = np.linalg.solve(self.Lc, dl.T).T                     # min |Lc^T lam - Lc^-1 dl|, lam >= 0
        return np.array([nnls(self.Lc.T, bi)[0] for bi in b])

    def strains(self, lam, f=None):
        """lambda (P, 3n) -> per cell: rotation vector (P, nc, 3) and axial shortening (P, nc). f (P, 3n, nc): tension
        factors along each tendon (capstan_factors), None = no friction.
        Source: tendon load = moment T d plus axial force T on every section passed, Camarillo et al., T-RO 2008
        (refs.bib: camarillo2008); Rucker & Webster, T-RO 2011 (rucker2011)."""
        Tl = lam[:, :, None] * self.through[None].astype(float)       # (P, 3n, nc)
        if f is not None:
            Tl = Tl * f
        Mx = np.einsum("pic,i->pc", Tl, self.D[:, 0])
        Mz = np.einsum("pic,i->pc", Tl, self.D[:, 1])
        F = Tl.sum(1)
        kx, kz = self.bend(Mx, Mz)                                     # integrated curvature per cell (rad)
        w = np.stack([kz, np.zeros_like(kz), -kx], -1)                 # bend towards +x rotates about -z, +z about +x
        return w, F * self.cA

    def friction_map(self, f):
        """Tendon map with the tension factors f (P, 3n, nc): M (P, 3n, 3n), dl = M lambda; f = 1 gives G."""
        T = self.through.astype(float)
        Tf = T[None] * f
        Dx, Dz = self.D[:, 0], self.D[:, 1]
        part = lambda c: np.einsum("ic,pjc->pij", T * c, Tf)            # noqa: E731
        return part(self.cA) + np.outer(Dx, Dx) * part(self.cXX) + np.outer(Dz, Dz) * part(self.cZZ) \
            + (np.outer(Dx, Dz) + np.outer(Dz, Dx)) * part(self.cXZ) + np.diag(self.tendon_compliance)

    def frames(self, lam, stations):
        """Integrate the backbone: -> centreline (P, K, 3) and rotations (P, K, 3, 3) at material stations (mm)."""
        w, sh = self.strains(lam)
        _, _, pe, Re = _march(w, sh, np.zeros(sh.shape + (2,)), float(self.cell), self.gap_after)
        return self.at_stations(pe, Re, stations)

    def at_stations(self, pe, Re, stations):
        """Edge positions (P, nc + 1, 3) and frames (P, nc + 1, 3, 3) from _march -> those at the stations (cell edges,
        the first matching edge, i.e. before a joint gap); stations off the edges stay zero."""
        st = np.asarray(stations, float)
        C, Rs = np.zeros((len(pe), len(st), 3)), np.zeros((len(pe), len(st), 3, 3))
        for k, y in enumerate(st):
            j = np.flatnonzero(np.abs(self.edges - y) < 1e-6)
            if len(j):
                C[:, k], Rs[:, k] = pe[:, j[0]], Re[:, j[0]]
        return C, Rs

    def predict(self, dl, stations):
        lam = self.tensions(dl)
        C, R = self.frames(lam, stations)
        return C, R, lam


def capstan_factors(eb, w, sigma, mu):
    """Tension factor of every tendon in every cell, f = exp(-sigma mu Theta), Theta the turning |w| accumulated along
    the tendon's path from the base to the cell centre. w (P, nc, 3) cell rotation vectors, sigma (P, 3n) -> (P, 3n, nc);
    the twist (w[..., 1]) does not turn a tendon at first order and is left out.
    Source: the capstan (Euler-Eytelwein) equation applied cell by cell."""
    T = eb.through.astype(float)
    turn = T[None] * np.linalg.norm(w[..., [0, 2]], axis=-1)[:, None, :]
    theta = np.cumsum(turn, 2) - 0.5 * turn
    return np.exp(-np.asarray(sigma, float)[:, :, None] * mu * theta)


def lcp_pull_only(M, b, iters=500):
    """Pull-only tensions for a tendon map M that need not be symmetric: lambda >= 0, M lambda >= b,
    lambda . (M lambda - b) = 0. Murty's least-index principal pivoting, finite for P-matrices (M = G scaled by positive
    tension factors); started from the tendons with b > 0. Agrees with EBPrior.tensions when M = G.
    Source: Murty, Linear Complementarity, Linear and Nonlinear Programming, 1988 (refs.bib: murty1988)."""
    b = np.asarray(b, float)
    S = b > 0
    tb = 1e-12 * max(np.abs(b).max(), 1e-300)
    for _ in range(iters):
        lam = np.zeros(len(b))
        if S.any():
            lam[S] = np.linalg.solve(M[np.ix_(S, S)], b[S])
        r = M @ lam - b
        bad = np.flatnonzero((S & (lam < -1e-12 * max(np.abs(lam).max(), 1e-300))) | (~S & (r < -tb)))
        if not len(bad):
            return np.maximum(lam, 0.0)
        S[bad[0]] = ~S[bad[0]]
    raise RuntimeError("pull-only LCP did not converge (tendon map not a P-matrix?)")


# ---------------------------------------------------------------- gravity (any direction)
try:                                                    # compiled per-cell integration (falls back to plain Python)
    from numba import njit as _njit
except ImportError:                                     # pragma: no cover
    def _njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]


@_njit(cache=True)
def _rot_into(wx, wy, wz, out):
    """Rotation matrix of the rotation vector (wx, wy, wz) (Rodrigues, as _rodrigues), written into out (3, 3)."""
    ang = math.sqrt(wx * wx + wy * wy + wz * wz)
    if ang < 1e-15:
        for i in range(3):
            for j in range(3):
                out[i, j] = 1.0 if i == j else 0.0
        return
    x, y, z = wx / ang, wy / ang, wz / ang
    s, c = math.sin(ang), 1.0 - math.cos(ang)
    out[0, 0] = 1.0 - c * (y * y + z * z); out[0, 1] = -s * z + c * x * y; out[0, 2] = s * y + c * x * z
    out[1, 0] = s * z + c * x * y; out[1, 1] = 1.0 - c * (x * x + z * z); out[1, 2] = -s * x + c * y * z
    out[2, 0] = -s * y + c * x * z; out[2, 1] = s * x + c * y * z; out[2, 2] = 1.0 - c * (x * x + y * y)


@_njit(cache=True)
def _mul_into(A, B, out):
    for i in range(3):
        for j in range(3):
            out[i, j] = A[i, 0] * B[0, j] + A[i, 1] * B[1, j] + A[i, 2] * B[2, j]


@_njit(cache=True)
def _march(w, sh, sxz, cell, gap):
    """Integrate the backbone cell by cell (midpoint rule; used by EBPrior.frames and predict_gravity). w (P, nc, 3)
    integrated curvature per cell, sh (P, nc) axial shortening, sxz (P, nc, 2) shear offset in the cell frame, gap (nc,)
    rigid step after each cell (EBPrior.gap_after) -> cell-centre positions (P, nc, 3), half-step frames (P, nc, 3, 3), edge positions
    (P, nc + 1, 3, before the gap) and edge frames (P, nc + 1, 3, 3).
    Allocation-free 3x3 arithmetic: this loop is the twin's inner cost."""
    P, nc = sh.shape
    pc, Rc = np.zeros((P, nc, 3)), np.zeros((P, nc, 3, 3))
    pe, Re = np.zeros((P, nc + 1, 3)), np.zeros((P, nc + 1, 3, 3))
    R, Rn, Rh, Q = np.eye(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))
    p = np.zeros(3)
    for q in range(P):
        for i in range(3):
            p[i] = 0.0
            for j in range(3):
                R[i, j] = 1.0 if i == j else 0.0
        Re[q, 0] = R
        for c in range(nc):
            _rot_into(0.5 * w[q, c, 0], 0.5 * w[q, c, 1], 0.5 * w[q, c, 2], Q)
            _mul_into(R, Q, Rh)
            a, b0, b2 = cell - sh[q, c], sxz[q, c, 0], sxz[q, c, 1]
            for i in range(3):
                st = Rh[i, 1] * a + Rh[i, 0] * b0 + Rh[i, 2] * b2
                pc[q, c, i] = p[i] + 0.5 * st
                p[i] += st
                pe[q, c + 1, i] = p[i]
            Rc[q, c] = Rh
            _rot_into(w[q, c, 0], w[q, c, 1], w[q, c, 2], Q)
            _mul_into(R, Q, Rn)
            R[:, :] = Rn
            Re[q, c + 1] = R
            if gap[c] != 0.0:
                for i in range(3):
                    p[i] += R[i, 1] * gap[c]
    return pc, Rc, pe, Re


def predict_gravity(eb, dl, stations, g_vec=(0.0, -STD6.gravity_mm_s2, 0.0), E=STD6.youngs_kpa, mass=MASS_STD, iters=60,
                    relax=0.5, tol=1e-4, state=None, joint_mass=0.0, tip_mass=0.0, mu=0.0, direction=None):
    """Variable-section Euler-Bernoulli prior under gravity (units mm, kg, s; E in kPa = mN/mm^2, SOFA's units).

    Fixed point: shape -> internal moment and force of the weight distal to every cell (in the cell frame) ->
    curvature and compression = (tendon + gravity load) / (E J), / (E A) -> pull-only tensions re-solved with the
    gravity-induced tendon path change -> shape. Mass per module distributed in proportion to the CAD area;
    joint_mass (kg, coupler + dowel + plugs) at every joint, split between the two cells beside it, and tip_mass (kg)
    on the last cell. mu > 0: capstan friction (module docstring) with direction (P, 3n) sigma per tendon (+1 wound in,
    -1 paid out, 0 none; None = all +1); the tension factors follow the current shape inside the fixed point.
    Shear (eb.cS > 0, Timoshenko): the transverse part of the distal weight in each cell frame shears the cell, moving
    the centreline sideways by cS * F_transverse / E relative to the cross-section frame; frames are not rotated by it
    and, to first order, tendon path lengths do not change.

    The relaxation `relax` is halved (down to 0.05) whenever the change grows between iterations: heavy stacks
    oscillate at 0.5 (4 modules hanging) and converge at smaller steps; well-behaved cases never trigger it.

    -> C (P, K, 3), R (P, K, 3, 3), lam (P, 3n), info = {"iterations", "converged", "state"}. Passing the previous
    call's info["state"] (same number of poses) starts the fixed point from its gravity strains: a live twin following
    slowly changing commands then needs only a few iterations. converged is False when the iteration hit `iters` or
    left the finite range (a stack past its buckling load); C and R are then the last iterate."""
    dl = np.atleast_2d(dl)
    P, nc = len(dl), len(eb.yc)
    # mass per cell from the CAD area: A_cell * cell / (sum over a module) * mass
    area = eb.cell / eb.cA_cad                          # cell length / integral(1/A) = harmonic-mean CAD area
    per_mod = len(eb.yc) // eb.n
    m_cell = np.concatenate([mass * area[j * per_mod:(j + 1) * per_mod] / area[j * per_mod:(j + 1) * per_mod].sum()
                             for j in range(eb.n)])
    for j in range(eb.n - 1):
        m_cell[(j + 1) * per_mod - 1] += 0.5 * joint_mass
        m_cell[(j + 1) * per_mod] += 0.5 * joint_mass
    m_cell[-1] += tip_mass
    Fg = m_cell[:, None] * np.asarray(g_vec, float)[None] / E          # (nc, 3) world force / E per cell
    T = eb.through.astype(float)
    sigma = np.ones((P, 3 * eb.n)) if direction is None else np.broadcast_to(np.asarray(direction, float), (P, 3 * eb.n))
    Fsum = np.cumsum(Fg[::-1], 0)[::-1]                                 # cells c.. (inclusive)
    Fdist = np.concatenate([Fsum[1:], np.zeros((1, 3))])                # cells c+1..

    def path_shift(shg_, kx_, kz_):                                     # gravity-induced tendon path shortening
        return np.einsum("ic,pc->pi", T, shg_) + np.einsum("ic,pc,i->pi", T, kx_, eb.D[:, 0]) \
            + np.einsum("ic,pc,i->pi", T, kz_, eb.D[:, 1])

    if state is not None and state["wg"].shape == (P, nc, 3):
        wg, shg, sxz = state["wg"].copy(), state["shg"].copy(), state["sxz"].copy()
        lam = eb.tensions(dl - path_shift(shg, -wg[..., 2], wg[..., 0]))
    else:
        wg, shg = np.zeros((P, nc, 3)), np.zeros((P, nc))
        sxz = np.zeros((P, nc, 2))                                      # shear offset per cell (x, z), cell frame
        lam = eb.tensions(dl)
    w_t, sh_t = eb.strains(lam)
    converged, last_change, relax_min = False, np.inf, 0.05
    for it in range(iters):
        # current frames at cell centres
        w, sh = w_t + wg, sh_t + shg
        pc, Rc, _, _ = _march(w, sh, sxz, float(eb.cell), eb.gap_after)
        # distal weight and its moment about each cell centre (world), then in the cell frame
        mom_all = np.cumsum(np.cross(pc, Fg[None])[:, ::-1], 1)[:, ::-1]    # sum_{c'>=c} p_c' x F_c'
        mom_dist = np.concatenate([mom_all[:, 1:], np.zeros((P, 1, 3))], 1)
        Mw = mom_dist - np.cross(pc, Fdist[None])                           # sum_{c'>c} (p_c' - p_c) x F_c'
        Ml = np.einsum("pcji,pcj->pci", Rc, Mw)                             # in the cell frame
        Fl = np.einsum("pcji,cj->pci", Rc, Fdist)
        kx, kz = eb.bend(-Ml[..., 2], Ml[..., 0])                          # moments towards +x, +z (tendon convention)
        wg_new = np.stack([kz, eb.cT * Ml[..., 1], -kx], -1)              # twist: the moment about the backbone axis
        shg_new = -Fl[..., 1] * eb.cA                                       # compression when the distal weight pushes down
        sxz_new = Fl[..., [0, 2]] * eb.cS[None, :, None]                    # shear along the transverse distal force
        h = path_shift(shg_new, kx, kz)
        if not np.all(np.isfinite(h)):                                     # diverged (past the buckling load)
            break
        if mu:                                                             # factors from the current shape
            f = capstan_factors(eb, w, sigma, mu)
            M = eb.friction_map(f)
            lam_new = np.array([lcp_pull_only(M[q], b) for q, b in enumerate(dl - h)])
            w_t_new, sh_t_new = eb.strains(lam_new, f)
        else:
            lam_new = eb.tensions(dl - h)                                  # pull-only tensions with that offset
            w_t_new, sh_t_new = eb.strains(lam_new)
        # A straight hanging robot can change only in extension: rotations alone
        # incorrectly reported convergence after one under-relaxed update.
        # Normalize translations by cell length so tol bounds strain as well as rotation.
        change = max(np.abs(wg_new - wg).max() + np.abs(w_t_new - w_t).max(),
                     (np.abs(shg_new - shg).max() + np.abs(sh_t_new - sh_t).max()) / eb.cell,
                     np.abs(sxz_new - sxz).max() / eb.cell)
        if change > last_change:                                       # oscillating or growing: smaller steps
            relax = max(0.5 * relax, relax_min)
        last_change = change
        wg, shg = (1 - relax) * wg + relax * wg_new, (1 - relax) * shg + relax * shg_new
        sxz = (1 - relax) * sxz + relax * sxz_new
        w_t, sh_t, lam = (1 - relax) * w_t + relax * w_t_new, (1 - relax) * sh_t + relax * sh_t_new, lam_new
        if change < tol:
            converged = True
            break
    # final frames at the stations (cell edges)
    _, _, pe, Re = _march(w_t + wg, sh_t + shg, sxz, float(eb.cell), eb.gap_after)
    C, Rs = eb.at_stations(pe, Re, stations)
    info = dict(iterations=it + 1, converged=converged, state=dict(wg=wg, shg=shg, sxz=sxz))
    return C, Rs, lam, info
