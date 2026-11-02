"""
Level-0 "virtual sensor": closed-form piecewise-constant-curvature (PCC)
kinematics for the 3-tendon TDCR backbone.

No SOFA dependency -- pure math, instant to evaluate. This is the cheapest
of the three fidelity levels in the digital-tuning pipeline:

    Level 0 (this file)   theoretical constant curvature       ~free
    Level 1 (cylinder_scene.py)  SOFA cylinder FEM sim          cheap-ish
    Level 2 (mocap, not yet collected)  real TDCR tip           expensive, sparse

Geometry / sign convention matches tdcr_scene.py exactly:
  - Backbone axis is +Y, base at y=0, undeformed tip at (0, BACKBONE_L, 0).
  - Tendons sit in the X-Z plane at radius TENDON_R, at azimuths
    0 deg / 120 deg / 240 deg (see cable_xz() in tdcr_scene.py):
        x = r*cos(theta), z = r*sin(theta)
  - dl_i is the SOFA `PullingCable` convention: dl_i >= 0 is how much
    cable i has been pulled in (shortened). This is exactly the tendon
    length reduction used below, so cc_kinematics and SOFA agree on sign.

Derivation
----------
Under the constant-curvature assumption the backbone centerline is a
circular arc of curvature kappa and arc length L (inextensible). A tendon
offset by radius r at azimuth theta_i, in a backbone bending toward
azimuth alpha, has arc length

    l_i = L * (1 - kappa * r * cos(theta_i - alpha))

so the amount that tendon i has been pulled in is

    dl_i = L - l_i = kappa * r * L * cos(theta_i - alpha)

For 3 tendons spaced exactly 120 deg apart this is 3 samples of a single
sinusoid in theta, so (kappa, alpha) recover *exactly* (no least-squares
needed) via the discrete Fourier relations for 3 equally spaced samples:

    x_i = dl_i / (r*L) = A*cos(theta_i) + B*sin(theta_i),  A = kappa*cos(alpha), B = kappa*sin(alpha)
    A = (2/3) * sum_i x_i * cos(theta_i)
    B = (2/3) * sum_i x_i * sin(theta_i)
    kappa = sqrt(A^2 + B^2),  alpha = atan2(B, A)

Naming note: this "alpha" is the plane-of-curvature angle -- called `phi`
in earlier versions of this file, renamed to match the lab's convention
(same quantity `tendon_mapping.py` calls `alpha`).

Sanity checks (see `_self_test` / run this file directly):
  - dl1=dl2=dl3=0                -> straight, tip = (0, L, 0)
  - dl1=dl2=dl3=const            -> A=B=0 -> straight (PCC has no axial
    compression term; a real backbone would foreshorten slightly under
    symmetric pull -- that gap is exactly what Level-1/Level-2 tuning
    against real data is for)
  - only dl1 > 0 (cable at theta=0)  -> alpha=0, tip bends toward +X,
    i.e. toward the pulled cable, as expected physically
"""

import math

from tdcr.params import STD6

# ── Geometry ──────────────────────────────────────────────────────────────
# The std6 robot (tdcr/params.py): 160 mm modules, channels at r = 9.5 mm. (Until the packaging these came from the
# environment variables TDCR_TENDON_R / TDCR_SECTION_L with stale defaults 9.0 / 220; every std6 run set 9.5 / 160.)
TENDON_R = STD6.tendon_radius_mm      # mm, radius from backbone axis to tendon channel
BACKBONE_L = STD6.module_length_mm    # mm, section (module) length

THETA_DEG = (0.0, 120.0, 240.0)
THETA = tuple(math.radians(t) for t in THETA_DEG)


def cable_deltas_to_curvature(dl1, dl2, dl3, r=TENDON_R, L=BACKBONE_L):
    """(dl1,dl2,dl3) [mm pulled in] -> (kappa [1/mm], alpha [rad])."""
    if L <= 0 or r <= 0:
        raise ValueError("r and L must be positive")

    x = [dl / (r * L) for dl in (dl1, dl2, dl3)]
    A = (2.0 / 3.0) * sum(xi * math.cos(t) for xi, t in zip(x, THETA))
    B = (2.0 / 3.0) * sum(xi * math.sin(t) for xi, t in zip(x, THETA))

    kappa = math.hypot(A, B)
    alpha = math.atan2(B, A) if kappa > 1e-12 else 0.0
    return kappa, alpha


def curvature_to_tip(kappa, alpha, L=BACKBONE_L):
    """(kappa, alpha) -> tip (x, y, z) in mm, backbone axis = Y."""
    if abs(kappa) < 1e-9:
        return (0.0, L, 0.0)

    R = 1.0 / kappa
    theta_bend = kappa * L  # total subtended angle
    u = R * (1.0 - math.cos(theta_bend))  # displacement in bend direction
    v = R * math.sin(theta_bend)          # displacement along backbone axis

    x = u * math.cos(alpha)
    z = u * math.sin(alpha)
    y = v
    return (x, y, z)


def theoretical_tip(dl1, dl2, dl3, r=TENDON_R, L=BACKBONE_L):
    """(dl1,dl2,dl3) [mm pulled in] -> tip (x,y,z) [mm]. The Level-0 estimate."""
    kappa, alpha = cable_deltas_to_curvature(dl1, dl2, dl3, r=r, L=L)
    return curvature_to_tip(kappa, alpha, L=L)


def _self_test():
    def close(a, b, tol=1e-6):
        return all(abs(ai - bi) < tol for ai, bi in zip(a, b))

    # 1. Zero actuation -> straight, tip at (0, L, 0)
    tip = theoretical_tip(0, 0, 0)
    assert close(tip, (0.0, BACKBONE_L, 0.0)), tip
    print(f"[ok] zero actuation -> tip = {tip}")

    # 2. Symmetric pull -> still straight (PCC has no axial-compression term)
    tip = theoretical_tip(5, 5, 5)
    assert close(tip, (0.0, BACKBONE_L, 0.0)), tip
    print(f"[ok] symmetric pull (5,5,5) -> tip = {tip}  (still straight under ideal PCC)")

    # 3. Pull cable 1 only (theta=0) -> bends toward +X, alpha ~= 0
    kappa, alpha = cable_deltas_to_curvature(20, 0, 0)
    assert kappa > 0 and abs(alpha) < 1e-6, (kappa, alpha)
    tip = curvature_to_tip(kappa, alpha)
    assert tip[0] > 0 and abs(tip[2]) < 1e-6, tip
    print(f"[ok] pull cable 1 (theta=0) only -> kappa={kappa:.6f} 1/mm, alpha={math.degrees(alpha):.2f} deg, tip={tip}")

    # 4. Pull cable 2 only (theta=120 deg) -> bends toward alpha ~= 120 deg
    kappa, alpha = cable_deltas_to_curvature(0, 20, 0)
    assert abs(math.degrees(alpha) - 120.0) < 1e-6, math.degrees(alpha)
    print(f"[ok] pull cable 2 (theta=120deg) only -> alpha={math.degrees(alpha):.2f} deg")

    # 5. Round-trip consistency: forward (kappa,alpha)->dl, then dl->(kappa,alpha)
    kappa0, alpha0 = 0.003, math.radians(50)
    dls = []
    for t in THETA:
        dl = kappa0 * TENDON_R * BACKBONE_L * math.cos(t - alpha0)
        dls.append(dl)
    kappa1, alpha1 = cable_deltas_to_curvature(*dls)
    assert abs(kappa1 - kappa0) < 1e-9 and abs(alpha1 - alpha0) < 1e-9, (kappa1, alpha1)
    print(f"[ok] round-trip (kappa,alpha) -> dl -> (kappa,alpha): {kappa0:.6f},{math.degrees(alpha0):.2f} -> {kappa1:.6f},{math.degrees(alpha1):.2f}")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()
