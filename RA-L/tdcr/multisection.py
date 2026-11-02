"""
Multi-section PCC kinematics with full SE(3) pose, for the 2-section /
6-tendon backbone.

This generalises cc_kinematics.py in two directions at once:

  1. N sections instead of one. Each section i has its own curvature
     kappa_i, bend-plane angle alpha_i, and length L_i, and carries its own
     group of 3 tendons at azimuths 0/120/240 deg.

  2. Full pose instead of position. Cavity insertion needs to know which
     way the tip is *pointing*, not just where it is, so everything here
     returns 4x4 homogeneous transforms rather than 3-vectors.

cc_kinematics.py is left untouched and still owns the single-section case;
`_self_test` below checks that this module reproduces it exactly, so the
two cannot silently drift apart.

Tendon coupling
---------------
A distal section's tendons are routed *through* every proximal section, so
pulling them is affected by how the proximal sections are bent. Writing
lv_j = r * kappa_j * L_j (the pull-in amplitude of section j) and Gamma_i
for the cumulative routing twist at section i, the pull-in of tendon k in
group i is

    dl_{i,k} = sum_{j <= i} lv_j * cos(theta_k - (alpha_j - Gamma_i))

which is exactly the lower-triangular T matrix in tendon_mapping.py, and
reduces to cc_kinematics.py's dl = kappa*r*L*cos(theta - alpha) for one
section.

The inverse is NOT a hard coupled solve
---------------------------------------
Because that system is lower-triangular, it inverts by forward
substitution, applying the same exact 3-sample Fourier trick once per
section:

    group 0 depends on section 0 alone      -> Fourier -> (lv_0, alpha_0)
    group 1 depends on sections 0 and 1     -> subtract the now-known
                                               section-0 term, then
                                               Fourier the remainder
                                            -> (lv_1, alpha_1)
    ... and so on.

Still exact, still no least-squares, still no iteration. The elegance of
the single-section case survives into the multi-section case.

Pose residuals live in the tangent space
----------------------------------------
For the calibration step, the position residual is an ordinary vector
difference, but the *orientation* residual is not: rotations do not live
in a vector space, and fitting three independent GPs to Euler angles
breaks at wrapping discontinuities and gimbal lock. Instead we use the
right-composed rotation-vector residual

    R_real = R_sim @ exp([omega]_x),    omega in R^3

so omega is a genuine 3-vector, singularity-free for the small corrections
calibration actually learns, and can be regressed with exactly the same
machinery digitaltuning.py already uses for position.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation

from tdcr.cc_kinematics import TENDON_R, BACKBONE_L, THETA, THETA_DEG

# ── Geometry ──────────────────────────────────────────────────────────────
# A robot is a stack of identical modules: each section is a full BACKBONE_L
# (160 mm for std6, tdcr/params.py) and n sections are n * BACKBONE_L long.
SECTION_LENGTHS = (BACKBONE_L, BACKBONE_L)

# Routing twist applied between consecutive sections, radians. Zero means
# every section's tendon channels share the same azimuths -- confirmed for the
# lab robot (2026-09-28): the tendons of all sections pass through the same
# holes, with no rotation between sections.
SECTION_GAMMAS = (0.0, 0.0)


def section_lengths_for(n, section_lengths=None):
    """Lengths for an n-section robot. The defaults above describe the lab's two
    sections; a robot with n sections repeats the same module (the same
    backbone part, stacked). An explicit section_lengths always wins."""
    if section_lengths is not None:
        return tuple(section_lengths)
    return tuple(SECTION_LENGTHS) if n == len(SECTION_LENGTHS) else (BACKBONE_L,) * n


def gammas_for(n, gammas=None):
    """Routing twists for n sections (all zero: every section's tendons share the
    same holes, confirmed for the lab robot)."""
    if gammas is not None:
        return tuple(gammas)
    return tuple(SECTION_GAMMAS) if n == len(SECTION_GAMMAS) else (0.0,) * n


# ── Forward kinematics ────────────────────────────────────────────────────

def section_transform(kappa, alpha, L):
    """One PCC section -> 4x4 homogeneous transform, backbone axis = +Y.

    The section bends through theta = kappa*L in the plane at azimuth
    alpha. Translation matches cc_kinematics.curvature_to_tip exactly. The
    rotation is by theta about the axis perpendicular to both the incoming
    tangent (+Y) and the bend direction (cos alpha, 0, sin alpha), which
    is (sin alpha, 0, -cos alpha).
    """
    T = np.eye(4)

    if abs(kappa) < 1e-12:
        T[:3, 3] = (0.0, L, 0.0)
        return T

    R_arc = 1.0 / kappa
    theta = kappa * L
    u = R_arc * (1.0 - math.cos(theta))   # sideways, in the bend direction
    v = R_arc * math.sin(theta)           # along the incoming axis

    T[:3, 3] = (u * math.cos(alpha), v, u * math.sin(alpha))

    axis = np.array([math.sin(alpha), 0.0, -math.cos(alpha)])
    T[:3, :3] = Rotation.from_rotvec(axis * theta).as_matrix()
    return T


def forward_kinematics(sections):
    """[(kappa, alpha, L), ...] -> list of 4x4 transforms, one per section
    tip, each in the base frame. The last entry is the end-effector pose."""
    T = np.eye(4)
    frames = []
    for kappa, alpha, L in sections:
        T = T @ section_transform(kappa, alpha, L)
        frames.append(T.copy())
    return frames


def tip_pose(sections):
    """[(kappa, alpha, L), ...] -> 4x4 end-effector pose."""
    return forward_kinematics(sections)[-1]


def tip_position(sections):
    """Convenience: just the tip (x, y, z), for comparison with Level 0/1."""
    return tuple(tip_pose(sections)[:3, 3])


def tip_direction(sections):
    """Unit vector the tip points along (the transformed +Y axis).

    This is the quantity that matters for steering into a cavity.
    """
    return tuple(tip_pose(sections)[:3, 1])


# ── Tendon mapping (forward and inverse) ──────────────────────────────────

def tendon_deltas(sections, r=TENDON_R, gammas=None):
    """[(kappa, alpha, L), ...] -> tendon pull-ins, length 3*n.

    Ordering is [section-0 tendons at 0/120/240, section-1 tendons, ...].
    """
    n = len(sections)
    gammas = np.zeros(n) if gammas is None else np.asarray(gammas, dtype=float)
    Gamma = np.cumsum(gammas)

    lv = np.array([r * kappa * L for kappa, _, L in sections])
    alpha = np.array([a for _, a, _ in sections])

    out = np.zeros(3 * n)
    for i in range(n):
        for j in range(i + 1):                    # lower-triangular coupling
            beta = alpha[j] - Gamma[i]
            for k, th in enumerate(THETA):
                out[3 * i + k] += lv[j] * math.cos(th - beta)
    return out


def _fourier_recover(samples):
    """Three uniformly spaced samples of M*cos(theta - phi) -> (M, phi).

    The same exact 3-sample inversion cc_kinematics.py uses, factored out
    so the multi-section forward substitution can reuse it per section.
    """
    A = (2.0 / 3.0) * sum(s * math.cos(t) for s, t in zip(samples, THETA))
    B = (2.0 / 3.0) * sum(s * math.sin(t) for s, t in zip(samples, THETA))
    M = math.hypot(A, B)
    phi = math.atan2(B, A) if M > 1e-12 else 0.0
    return M, phi


def deltas_to_sections(dl, section_lengths=None,
                       r=TENDON_R, gammas=None):
    """Tendon pull-ins -> [(kappa, alpha, L), ...], exactly.

    Forward substitution through the lower-triangular coupling: recover the
    most proximal section first, subtract its now-known contribution from
    the next group, and repeat.
    """
    section_lengths = section_lengths_for(len(dl) // 3, section_lengths)
    n = len(section_lengths)
    if len(dl) != 3 * n:
        raise ValueError(f"expected {3*n} tendon deltas for {n} sections, got {len(dl)}")

    gammas = np.zeros(n) if gammas is None else np.asarray(gammas, dtype=float)
    Gamma = np.cumsum(gammas)
    dl = np.asarray(dl, dtype=float)

    lv = np.zeros(n)
    alpha = np.zeros(n)

    for i in range(n):
        residual = dl[3 * i: 3 * i + 3].copy()

        # strip out everything the already-solved proximal sections explain
        for j in range(i):
            beta = alpha[j] - Gamma[i]
            for k, th in enumerate(THETA):
                residual[k] -= lv[j] * math.cos(th - beta)

        M, phi = _fourier_recover(residual)
        lv[i] = M
        alpha[i] = phi + Gamma[i]      # phi was recovered as (alpha_i - Gamma_i)

    return [
        (lv[i] / (r * section_lengths[i]), alpha[i], section_lengths[i])
        for i in range(n)
    ]


# ── (theta, alpha) command interface ──────────────────────────────────────
#
# The DOE is specified in (theta, alpha) per section, not in raw tendon
# pulls, because that is what the lab actually commands and what the paper
# reports: theta is how far a section bends, alpha is the plane it bends
# in. These helpers convert a command to the section list and to the six
# tendon pulls SOFA needs.

def sections_from_theta_alpha(theta, alpha, section_lengths=None):
    """theta, alpha (radians, one per section) -> [(kappa, alpha, L), ...].

    kappa = theta / L, since theta is the arc-subtended angle and the
    backbone is inextensible under PCC.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    section_lengths = section_lengths_for(len(theta), section_lengths)
    if len(theta) != len(section_lengths) or len(alpha) != len(section_lengths):
        raise ValueError(f"expected {len(section_lengths)} theta/alpha values")
    return [(theta[i] / section_lengths[i], alpha[i], section_lengths[i])
            for i in range(len(section_lengths))]


def offset_to_nonnegative(dl, n_per_group=3):
    """Shift each group of tendons so its minimum pull is zero.

    A servo can pull a tendon in; it cannot push one out. Sampling in
    (theta, alpha) produces negative pulls for tendons on the OUTSIDE of the
    bend, which no CableConstraint can represent.

    Subtracting the group minimum fixes that without changing the commanded
    shape at all. The Fourier inversion recovers curvature from
    sum_i dl_i cos(theta_i) and sum_i dl_i sin(theta_i), and for three
    tendons at 0/120/240 both sum(cos) and sum(sin) are exactly zero -- so a
    constant added to all three lies precisely in the null space of the
    inversion. kappa and alpha come out identical.

    What the offset DOES add is a uniform pull on every tendon in the group,
    i.e. real axial compression. PCC has no term for that, so it is invisible
    at Level 0 but genuinely present in the FEM and on the hardware. It is
    part of what the residual model has to learn, and it is exactly what a
    real robot experiences when it holds tension on all tendons.
    """
    dl = np.asarray(dl, dtype=float).copy()
    for g in range(len(dl) // n_per_group):
        sl = slice(g * n_per_group, (g + 1) * n_per_group)
        dl[sl] -= dl[sl].min()
    return dl


def theta_alpha_to_deltas(theta, alpha, r=TENDON_R,
                          section_lengths=None,
                          gammas=None, nonnegative=True):
    """(theta, alpha) command -> 3n tendon pulls, ready for SOFA."""
    sections = sections_from_theta_alpha(theta, alpha, section_lengths)
    gammas = gammas_for(len(sections), gammas)
    dl = tendon_deltas(sections, r=r, gammas=gammas)
    return offset_to_nonnegative(dl) if nonnegative else dl


def theta_alpha_to_pose6(theta, alpha, section_lengths=None):
    """(theta, alpha) command -> Level-0 tip pose as [x,y,z,wx,wy,wz]."""
    return matrix_to_pose6(tip_pose(sections_from_theta_alpha(theta, alpha,
                                                              section_lengths)))


def circular_features(theta, alpha):
    """(theta, alpha) -> [theta_1, cos a_1, sin a_1, theta_2, cos a_2, sin a_2].

    alpha is an ANGLE ON A CIRCLE. Handing a regressor raw alpha is a real
    bug, not a nicety: alpha = 0 and alpha = 2*pi are the same configuration
    but sit at opposite ends of the input range, so any distance-based model
    (a GP kernel, a polynomial basis) treats identical robot poses as maximally
    far apart, and the fit is discontinuous across the seam.

    Encoding as (cos alpha, sin alpha) puts alpha on the unit circle where it
    belongs, and the seam disappears. Every model in this project consumes
    alpha through this function.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    alpha = np.atleast_2d(np.asarray(alpha, dtype=float))
    cols = []
    for j in range(theta.shape[1]):
        cols += [theta[:, j], np.cos(alpha[:, j]), np.sin(alpha[:, j])]
    return np.column_stack(cols)


# ── SE(3) residuals, for the calibration step ─────────────────────────────

def rotation_residual(R_sim, R_real):
    """omega such that R_real = R_sim @ exp([omega]_x). A plain 3-vector."""
    return Rotation.from_matrix(R_sim.T @ R_real).as_rotvec()


def apply_rotation_residual(R_sim, omega):
    """Inverse of rotation_residual: re-compose a corrected orientation."""
    return R_sim @ Rotation.from_rotvec(np.asarray(omega, dtype=float)).as_matrix()


def pose_residual(T_sim, T_real):
    """Two 4x4 poses -> 6-vector [dx, dy, dz, wx, wy, wz].

    This is what the GP regresses. The first three components are an
    ordinary position difference in mm; the last three are the tangent-space
    orientation correction in radians. Note the two halves have different
    units and scales -- digitaltuning.py's StandardScaler already handles
    that on the input side, and each output component is fit by its own GP,
    so no extra scaling is needed here.
    """
    dp = T_real[:3, 3] - T_sim[:3, 3]
    omega = rotation_residual(T_sim[:3, :3], T_real[:3, :3])
    return np.concatenate([dp, omega])


def apply_pose_residual(T_sim, residual):
    """Apply a 6-vector residual to a simulated pose -> calibrated pose."""
    residual = np.asarray(residual, dtype=float)
    T = np.eye(4)
    T[:3, 3] = T_sim[:3, 3] + residual[:3]
    T[:3, :3] = apply_rotation_residual(T_sim[:3, :3], residual[3:])
    return T


# ── Flat 6-vector pose interchange ────────────────────────────────────────
# CSV files (and GP targets) carry poses as [x, y, z, wx, wy, wz], where the
# last three are the ABSOLUTE tip orientation as a rotation vector in the
# base frame. These convert to and from the 4x4 form used above.

def pose6_to_matrix(p6):
    p6 = np.asarray(p6, dtype=float)
    T = np.eye(4)
    T[:3, 3] = p6[:3]
    T[:3, :3] = Rotation.from_rotvec(p6[3:6]).as_matrix()
    return T


def matrix_to_pose6(T):
    return np.concatenate([T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_rotvec()])


def pose6_residual(sim6, real6):
    """Residual between two flat poses -> 6-vector [dp, omega].

    Note this is NOT `real6 - sim6`: subtracting rotation vectors is
    meaningless. The orientation half is composed on SO(3) and returned in
    the tangent space at the simulated orientation.
    """
    return pose_residual(pose6_to_matrix(sim6), pose6_to_matrix(real6))


def apply_pose6_residual(sim6, residual):
    """Inverse of pose6_residual: correct a simulated pose by a residual."""
    return matrix_to_pose6(apply_pose_residual(pose6_to_matrix(sim6), residual))


# ── Self-test ─────────────────────────────────────────────────────────────

def _self_test():
    from tdcr.cc_kinematics import curvature_to_tip, cable_deltas_to_curvature

    def close(a, b, tol=1e-9):
        return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol)

    # 1. Single section must reproduce cc_kinematics exactly -----------------
    for kappa, alpha_deg in [(0.003, 50.0), (0.005, 0.0), (0.004, 120.0), (0.0015, -35.0)]:
        alpha = math.radians(alpha_deg)
        one = [(kappa, alpha, BACKBONE_L)]

        assert close(tip_position(one), curvature_to_tip(kappa, alpha, BACKBONE_L)), alpha_deg

        dl_ms = tendon_deltas(one)
        dl_cc = [kappa * TENDON_R * BACKBONE_L * math.cos(t - alpha) for t in THETA]
        assert close(dl_ms, dl_cc), (dl_ms, dl_cc)

        k_cc, a_cc = cable_deltas_to_curvature(*dl_cc)
        assert close([k_cc, a_cc], [kappa, alpha]), (k_cc, a_cc)
    print("[ok] single section reproduces cc_kinematics exactly (position, deltas, inverse)")

    # 2. Zero actuation, two sections ---------------------------------------
    straight = [(0.0, 0.0, SECTION_LENGTHS[0]), (0.0, 0.0, SECTION_LENGTHS[1])]
    T = tip_pose(straight)
    assert close(T[:3, 3], (0.0, sum(SECTION_LENGTHS), 0.0)), T[:3, 3]
    assert close(T[:3, :3], np.eye(3)), T[:3, :3]
    assert close(tendon_deltas(straight), np.zeros(6))
    print(f"[ok] two sections, zero actuation -> tip {tuple(np.round(T[:3,3],6))}, identity rotation")

    # 3. Forward -> inverse round trip, two sections, with routing twist -----
    for gammas in [(0.0, 0.0), (0.0, math.radians(30.0))]:
        sections = [
            (0.0040, math.radians(35.0), SECTION_LENGTHS[0]),
            (0.0062, math.radians(-70.0), SECTION_LENGTHS[1]),
        ]
        dl = tendon_deltas(sections, gammas=gammas)
        rec = deltas_to_sections(dl, SECTION_LENGTHS, gammas=gammas)

        for (k0, a0, _), (k1, a1, _) in zip(sections, rec):
            assert abs(k1 - k0) < 1e-12, (k0, k1)
            # compare angles modulo 2pi
            assert abs(math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))) < 1e-12, (a0, a1)
        g_deg = tuple(round(math.degrees(g), 1) for g in gammas)
        print(f"[ok] 2-section forward->inverse exact, routing twist {g_deg} deg")

    # 4. Coupling is real: distal deltas depend on the proximal section ------
    base = [(0.0, 0.0, SECTION_LENGTHS[0]), (0.005, 0.4, SECTION_LENGTHS[1])]
    bent = [(0.004, 0.9, SECTION_LENGTHS[0]), (0.005, 0.4, SECTION_LENGTHS[1])]
    d_base, d_bent = tendon_deltas(base), tendon_deltas(bent)
    assert close(d_base[:3], np.zeros(3)), d_base[:3]
    assert not close(d_base[3:], d_bent[3:], tol=1e-6), "distal tendons should feel proximal bending"
    print("[ok] distal tendon deltas change when only the proximal section bends (coupling present)")

    # 5. Orientation sanity: alpha=0 tilts the tip toward +X ------------------
    tilted = [(0.004, 0.0, BACKBONE_L)]
    d = tip_direction(tilted)
    assert d[0] > 0 and abs(d[2]) < 1e-12, d
    print(f"[ok] alpha=0 bend tilts tip direction toward +X: {tuple(np.round(d,6))}")

    # 6. SE(3) residual round trip -------------------------------------------
    T_sim = tip_pose([(0.0035, 0.5, SECTION_LENGTHS[0]), (0.0050, -0.8, SECTION_LENGTHS[1])])
    T_real = tip_pose([(0.0039, 0.55, SECTION_LENGTHS[0]), (0.0046, -0.75, SECTION_LENGTHS[1])])

    res = pose_residual(T_sim, T_real)
    T_rec = apply_pose_residual(T_sim, res)
    assert close(T_rec, T_real), (T_rec, T_real)
    print(f"[ok] SE(3) residual round trip: |dp|={np.linalg.norm(res[:3]):.4f} mm, "
          f"|omega|={math.degrees(np.linalg.norm(res[3:])):.4f} deg")

    # 7. Zero residual between a pose and itself ------------------------------
    assert close(pose_residual(T_sim, T_sim), np.zeros(6))
    print("[ok] residual of a pose against itself is exactly zero")

    # 8. (theta, alpha) command interface --------------------------------------
    th, al = [math.radians(30.0), math.radians(45.0)], [0.7, -1.9]
    secs = sections_from_theta_alpha(th, al)
    for i, (k, a, L) in enumerate(secs):
        assert abs(k * L - th[i]) < 1e-12, (k * L, th[i])   # theta == kappa*L
        assert abs(a - al[i]) < 1e-12
    assert close(theta_alpha_to_pose6(th, al), matrix_to_pose6(tip_pose(secs)))
    print("[ok] (theta,alpha) -> sections -> pose is consistent")

    # 9. The non-negativity offset must not change the recovered shape ----------
    raw = tendon_deltas(secs)
    off = offset_to_nonnegative(raw)
    assert off.min() >= -1e-12, off
    assert abs(off[:3].min()) < 1e-12 and abs(off[3:].min()) < 1e-12
    for (k0, a0, _), (k1, a1, _) in zip(deltas_to_sections(raw),
                                        deltas_to_sections(off)):
        assert abs(k1 - k0) < 1e-12, (k0, k1)
        assert abs(math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))) < 1e-12
    print(f"[ok] non-negativity offset is null-space: raw dl min "
          f"{raw.min():+.3f} -> {off.min():+.3f} mm, kappa/alpha unchanged")

    # 10. Circular encoding is continuous across the alpha = 0 / 2pi seam -------
    f_a = circular_features([[0.3, 0.3]], [[1e-9, 1e-9]])
    f_b = circular_features([[0.3, 0.3]], [[2 * math.pi - 1e-9, 2 * math.pi - 1e-9]])
    assert np.linalg.norm(f_a - f_b) < 1e-8, (f_a, f_b)
    assert f_a.shape == (1, 6)
    print("[ok] circular_features: alpha=0 and alpha=2pi map to the same point")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()
