"""
Tendon mapping: (theta, alpha) actuation-plane commands -> per-tendon
pull-in lengths (dl1, dl2, dl3) for the 3-tendon TDCR backbone.

Ported from the lab's hardware-control tendon_mapping.py and wired into the
digital-tuning pipeline's own conventions (TENDON_R, BACKBONE_L, alpha --
see cc_kinematics.py, the single source of truth for both). Kept as its own
file rather than folded into digitaltuning.py: this is actuation-command
math, the same kind of thing cc_kinematics.py is, not DOE/GP orchestration.

theta here is the total bend angle (radians*L = arc-subtended angle), the
same quantity cc_kinematics.py calls kappa*L. alpha is the plane-of-
curvature angle (cc_kinematics.py's alpha, called phi before the rename).

Mathematical link to cc_kinematics.py
--------------------------------------
cc_kinematics.py's forward tendon-pull formula, for 3 tendons at
theta_i in {0, 120, 240} deg:

    dl_i = kappa * r * L * cos(theta_i - alpha)

This file computes the same thing via a matrix T built from routing angles,
using an intermediate quantity `lv = R_D * theta_rad`. For a single section
(gamma=0), T reduces to exactly cos(theta_i - alpha), so:

    dl_i = lv * cos(theta_i - alpha) = (R_D * theta_rad) * cos(theta_i - alpha)

which is identical to cc_kinematics.py's formula once R_D == r (TENDON_R)
and theta_rad == kappa*L. `_self_test` below checks this numerically rather
than just asserting it.
"""

import math

import numpy as np

from tdcr.cc_kinematics import TENDON_R, BACKBONE_L, THETA_DEG, curvature_to_tip


class TendonMapping:
    """n_sections=1, gamma=(0.0,) matches this project's single-section,
    3-tendon backbone (see cc_kinematics.py's THETA_DEG). R_D defaults to
    TENDON_R so theta/alpha commands stay consistent with the Level-0 model
    without having to pass it explicitly every time."""

    def __init__(self, n_sections=1, gamma=(0.0,), R_D=TENDON_R, R_pulley=None):
        self.n = n_sections
        self.gamma = np.array(gamma)
        self.R_D = R_D
        self.R_pulley = R_pulley  # only needed for compute_motor_angles

    def compute_T(self, alpha_deg):
        # atleast_1d: a single-section robot naturally gets called with a
        # bare scalar (e.g. compute_T(50.0)) -- without this, np.radians()
        # on a scalar returns a 0-d array and alpha[j] below raises
        # IndexError.
        alpha = np.atleast_1d(np.radians(alpha_deg))
        Gamma = np.cumsum(np.radians(self.gamma))
        T = np.zeros((3 * self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i >= j:
                    T[3*i + 0, j] = np.cos(alpha[j] - Gamma[i])
                    T[3*i + 1, j] = np.cos(2*np.pi/3 - alpha[j] + Gamma[i])
                    T[3*i + 2, j] = np.cos(4*np.pi/3 - alpha[j] + Gamma[i])
        return T

    def compute_lengths(self, alpha_deg, lv):
        T = self.compute_T(alpha_deg)
        lv = np.array(lv)
        return T @ lv

    def compute_motor_angles(self, alpha_deg, lv):
        if self.R_pulley is None:
            raise ValueError("R_pulley was not set on this TendonMapping (pass R_pulley= in __init__)")
        l = self.compute_lengths(alpha_deg, lv)
        return (l / self.R_pulley) * (180.0 / np.pi)

    def theta_to_lv(self, theta_deg):
        """theta [deg] (per-section bend angle, == cc_kinematics.py's kappa*L in degrees)
        -> lv [same units as R_D] = R_D * theta_rad."""
        theta_deg = np.atleast_1d(np.array(theta_deg, dtype=float))
        return self.R_D * np.radians(theta_deg)

    def compute_lengths_from_theta(self, theta_deg, alpha_deg):
        """(theta_deg, alpha_deg) -> per-tendon pull-in lengths (dl1, dl2, dl3).

        The actual top-level command (bend angle + plane of curvature) ->
        actuation, in one call -- lv is computed internally, never exposed.
        """
        return self.compute_lengths(alpha_deg, self.theta_to_lv(theta_deg))

    def compute_motor_angles_from_theta(self, theta_deg, alpha_deg):
        return self.compute_motor_angles(alpha_deg, self.theta_to_lv(theta_deg))


def theta_alpha_to_dl(theta_deg, alpha_deg, r=TENDON_R):
    """Plain-function convenience wrapper, matching cc_kinematics.py's
    procedural style: (theta_deg, alpha_deg) -> (dl1, dl2, dl3) tuple."""
    mapping = TendonMapping(n_sections=1, gamma=(0.0,), R_D=r)
    dl = mapping.compute_lengths_from_theta(theta_deg, alpha_deg)
    return tuple(float(v) for v in dl)


def theta_alpha_to_tip(theta_deg, alpha_deg, r=TENDON_R, L=BACKBONE_L):
    """(theta_deg, alpha_deg) -> Level-0 theoretical tip (x, y, z) [mm],
    going straight through curvature_to_tip -- skips the round trip through
    (dl1,dl2,dl3) and back to (kappa,alpha) that theoretical_tip() would do,
    since theta/alpha are already the curvature-domain quantities."""
    kappa = math.radians(theta_deg) / L
    return curvature_to_tip(kappa, math.radians(alpha_deg), L=L)


def _self_test():
    def close(a, b, tol=1e-6):
        return all(abs(ai - bi) < tol for ai, bi in zip(a, b))

    mapping = TendonMapping(n_sections=1, gamma=(0.0,), R_D=TENDON_R)

    # Cross-check against cc_kinematics.py's own forward formula: for the
    # same (kappa, alpha), both must produce the same (dl1, dl2, dl3).
    for kappa0, alpha0_deg in [(0.003, 50.0), (0.005, 0.0), (0.004, 120.0), (0.006, 240.0), (0.002, -35.0)]:
        theta0_deg = math.degrees(kappa0 * BACKBONE_L)  # theta == kappa*L

        dl_cc = tuple(
            kappa0 * TENDON_R * BACKBONE_L * math.cos(math.radians(t) - math.radians(alpha0_deg))
            for t in THETA_DEG
        )
        dl_tm = mapping.compute_lengths_from_theta(theta0_deg, alpha0_deg)

        assert close(dl_cc, dl_tm, tol=1e-9), (dl_cc, dl_tm)
        print(f"[ok] kappa={kappa0}, alpha={alpha0_deg}deg -> "
              f"cc_kinematics dl={tuple(round(v,6) for v in dl_cc)} == "
              f"tendon_mapping dl={tuple(round(v,6) for v in dl_tm)}")

    # Zero bend -> zero pull-in on all three tendons
    dl0 = mapping.compute_lengths_from_theta(0.0, 0.0)
    assert close(dl0, (0.0, 0.0, 0.0)), dl0
    print(f"[ok] theta=0 -> dl = {tuple(dl0)}")

    # theta_alpha_to_tip agrees with cc_kinematics.py's own round trip
    # (dl -> kappa,alpha -> tip) for the same (theta, alpha)
    theta0_deg, alpha0_deg = 15.0, 40.0
    kappa0 = math.radians(theta0_deg) / BACKBONE_L
    tip_direct = curvature_to_tip(kappa0, math.radians(alpha0_deg))
    tip_via_wrapper = theta_alpha_to_tip(theta0_deg, alpha0_deg)
    assert close(tip_direct, tip_via_wrapper), (tip_direct, tip_via_wrapper)
    print(f"[ok] theta_alpha_to_tip matches curvature_to_tip directly: {tip_via_wrapper}")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    _self_test()
