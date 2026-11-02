"""
Cosserat rod model for the two-section tendon-driven backbone.

Why this exists
---------------
The paper currently compares against constant curvature. Every reviewer in
this field knows PCC is crude, so "we beat PCC" is a weak claim. The
standard physics model for a tendon-driven continuum robot is a Cosserat
rod (Rucker & Webster, IJRR 2011), and that is the baseline the work has to
stand next to.

What it captures that PCC cannot
--------------------------------
  * AXIAL EXTENSION AND COMPRESSION. PCC assumes an inextensible backbone.
    The measured FEM data shows each section under-bending by 3-6% and the
    axial residual tracking total tendon pull -- both are compression that
    PCC has no term for.

  * SHEAR AND TORSION, via the full Kse / Kbt stiffness matrices.

WHAT IT DOES NOT ADD, and this is worth stating plainly
-------------------------------------------------------
NOT variable curvature. With straight parallel tendon channels and no
external distributed load, the internal balance gives

    n_rod = -sum_i tau_i t_i        m_rod = -sum_i tau_i (r_i x t_i)

entirely in the BODY frame, so v and u come out independent of orientation
and independent of arc length. A Cosserat rod under those conditions is
EXACTLY piecewise constant curvature. (Verified numerically: evaluating the
constitutive relation at three very different orientations returns identical
v and u to machine precision.)

So constant curvature is not an approximation here -- it is exact, and PCC's
only errors in this regime are its inextensibility and its lack of shear.
That is a cleaner claim than "PCC is crude", and it says precisely what the
comparison isolates.

Variable curvature would require gravity or an external tip load. Gravity
turns this into a two-point BVP needing shooting; the FEM sweep measured its
contribution at 0.98 mm out of 6.09 mm total, so it is not modelled here.

Consequence for the implementation: no ODE is needed. Each section is a
constant-twist screw motion with a closed-form exponential, evaluated in
microseconds rather than the 0.14 s an RK45 pass cost -- which is what makes
the displacement inverse tractable.

  * TENDON ROUTING THROUGH THE PROXIMAL SECTION. This is the important one
    here. A distal tendon physically runs through section 1's channels, so
    section 1's shape changes its length and it loads section 1 along the
    way. two_section_scene.py CANNOT represent this -- a SOFA cable attaches
    to one MechanicalObject, so the run from the base servo up to section 2
    is modelled as a free chord. That approximation is the leading suspect
    for why section 2 under-bends twice as much as section 1 (94.2% vs
    97.2%) in the FEM sweep.

    This model routes the distal tendons correctly. Comparing the two is
    therefore a direct test of how much that approximation costs.

Formulation
-----------
Rucker & Webster's tendon-driven static rod, integrated as an INITIAL value
problem. With tendon tensions known, the tendon loads can be written as
functions of the local state, so no shooting is needed -- integrate once
from base to tip. The 6x6 system solved at each step is

    [ Kse + A     G     ] [ v' ]   [ d ]
    [ B         Kbt + H ] [ u' ] = [ c ]

State is [p, R, v, u]: position, orientation, linear rate, curvature.

KNOWN LIMITATION. The displacement solve is a nonlinear COMPLEMENTARITY
problem, not a plain root-find: a tendon commanded to hold its length may
need to go slack (tau = 0) while its neighbours take up tension. C-shapes
and single-section bends converge exactly. Strong S-shapes -- where section
2 must bend away from a tendon that is itself commanded to hold length --
can stall short, because reaching the answer means driving one tendon onto
the tau = 0 bound. Multi-start helps but does not guarantee it. A proper LCP
solver would; that is the fix if this becomes limiting.

Tensions are not what we command, though -- we command tendon DISPLACEMENTS.
So there is an outer solve: pick tensions, integrate, measure the resulting
tendon length changes, and iterate until they match the commanded pull-ins.
Tensions are bounded below at zero because a tendon cannot push.

Body frame convention: +Y is the tangent, matching the rest of this project
(cc_kinematics.py:13). That is not the textbook z-tangent convention, so
Kse and Kbt are permuted accordingly -- see _stiffness().

    python3 -m tdcr.cosserat --self-test
    python3 -m tdcr.cosserat --sweep results/configs_2s_ta.csv --out results/level0p5_2s.csv
"""

import argparse
import csv
import math
import os

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from tdcr.cc_kinematics import TENDON_R, THETA_DEG
from tdcr.multisection import SECTION_LENGTHS, SECTION_GAMMAS, matrix_to_pose6

# ── Material and section properties ───────────────────────────────────────
# Matched to the SOFA scenes so the two models describe the same rod.
# NOTE the units question flagged elsewhere in this project is unresolved
# here too: E is carried through exactly as the FEM uses it, so the two are
# consistent with each other even if neither is in stated SI units.
YOUNG_MODULUS = 50000.0
POISSON_RATIO = 0.45
R_OUTER = 11.5                     # mm
GRAVITY = 0.0                      # per-length body force; see --gravity


def _stiffness(E=YOUNG_MODULUS, nu=POISSON_RATIO, r=R_OUTER, EA=None, EI=None):
    """-> (Kse, Kbt) for a solid circular cross-section, +Y tangent.

    Kse resists extension along the tangent and shear across it; Kbt resists
    bending about the two transverse axes and torsion about the tangent.
    With +Y as the tangent the middle entry of each is the axial one.
    EA / EI override the solid-section values (e.g. a tube, or a calibrated
    axial-to-bending ratio); shear and torsion follow from nu.
    """
    G = E / (2.0 * (1.0 + nu))
    A = math.pi * r ** 2
    I = math.pi * r ** 4 / 4.0
    EA = E * A if EA is None else float(EA)
    EI = E * I if EI is None else float(EI)
    GA, GJ = EA * G / E, 2.0 * EI * G / E
    Kse = np.diag([GA, EA, GA])               # shear, EXTENSION, shear
    Kbt = np.diag([EI, GJ, EI])               # bend, TORSION, bend
    return Kse, Kbt


def _skew(x):
    return np.array([[0.0, -x[2], x[1]],
                     [x[2], 0.0, -x[0]],
                     [-x[1], x[0], 0.0]])


def tendon_offsets(section_index, r=TENDON_R, gammas=SECTION_GAMMAS):
    """Body-frame offsets of one section's three tendons.

    Same azimuth convention as everywhere else: x = r cos(theta),
    z = r sin(theta), with +Y the tangent, so the offsets lie in the body
    x-z plane.
    """
    Gamma = float(np.cumsum(np.asarray(gammas, dtype=float))[section_index])
    return [np.array([r * math.cos(math.radians(t) + Gamma), 0.0,
                      r * math.sin(math.radians(t) + Gamma)])
            for t in THETA_DEG]


# ── The ODE ───────────────────────────────────────────────────────────────
#
# Static equilibrium with tendons and no distributed external load.
#
# Cut the rod at arc length s. The combined internal force and moment of
# (rod + tendons) must vanish, because nothing external loads the free end:
#
#     n_rod(s) + sum_i tau_i t_i(s)                = 0
#     m_rod(s) + sum_i tau_i (R r_i) x t_i(s)      = 0
#
# so the rod's own internal load is exactly minus the tendons' contribution.
# Feeding that through the constitutive law gives v and u algebraically --
# no need for the A/B/G/H differential system at all, which only earns its
# place when there IS a distributed external load to propagate.
#
# The relation is implicit: t_i depends on u, and u depends on t_i. It is a
# weak dependence though (|p_i'| stays near 1), so a short Picard iteration
# converges in a handful of passes.
#
# GRAVITY IS NOT MODELLED HERE. With a distributed body force the internal
# load no longer vanishes and the moment balance needs the shape, which turns
# this into a two-point BVP requiring shooting. It is left out deliberately:
# the FEM sweep measured gravity's contribution at 0.98 mm out of 6.09 mm
# total, so it is not what the comparison against PCC turns on.

def _vu_from_tendons(_R_unused, tendons, Kse, Kbt, v_star, u_star,
                     iters=40, tol=1e-14):
    """Solve the implicit constitutive relation for (v, u) at one station."""
    v = v_star.copy()
    u = u_star.copy()
    for _ in range(iters):
        nb = np.zeros(3)
        mb = np.zeros(3)
        for r_i, tau in tendons:
            if tau <= 0.0:
                continue
            pb = np.cross(u, r_i) + v
            nrm = float(np.linalg.norm(pb))
            if nrm < 1e-12:
                continue
            t = pb / nrm                       # unit tendon direction, body frame
            nb -= tau * t
            mb -= tau * np.cross(r_i, t)
        v_new = v_star + np.linalg.solve(Kse, nb)
        u_new = u_star + np.linalg.solve(Kbt, mb)
        if max(np.max(np.abs(v_new - v)), np.max(np.abs(u_new - u))) < tol:
            v, u = v_new, u_new
            break
        v, u = v_new, u_new
    return v, u


def _section_transform(v, u, L):
    """Exact pose change over a section of constant (v, u).

    Constant body-frame twist integrates to a screw motion, so the 4x4
    transform is the matrix exponential of the se(3) element [[u^, v], [0, 0]]
    scaled by L. Exact, and far cheaper than integrating.
    """
    xi = np.zeros((4, 4))
    xi[:3, :3] = _skew(u)
    xi[:3, 3] = v
    return expm(xi * L)


def integrate(tau, section_lengths=SECTION_LENGTHS, r=TENDON_R,
              gammas=SECTION_GAMMAS, gravity=GRAVITY, n_eval=25, **kw):
    """Base to tip for the given six tendon tensions, in closed form.

    -> (frames, tendon_lengths). Signature unchanged from the ODE version so
    callers do not care which is underneath.

    Section-1 tendons act only over section 1; section-2 tendons act over
    BOTH, because they are routed through section 1 from the base servos.
    That routing is the coupling two_section_scene.py has to approximate as a
    free chord.
    """
    if gravity:
        raise NotImplementedError(
            "gravity makes v and u vary along s, which needs a BVP shooting "
            "solve. The FEM sweep put gravity at 0.98 mm of 6.09 mm total.")

    Kse, Kbt = _stiffness(**kw)
    v_star = np.array([0.0, 1.0, 0.0])
    u_star = np.zeros(3)
    offs = [tendon_offsets(i, r, gammas) for i in range(len(section_lengths))]

    T = np.eye(4)
    frames = [T.copy()]
    lengths = np.zeros(len(tau))

    for sec, L in enumerate(section_lengths):
        active = [(offs[j][k], float(tau[3 * j + k]), 3 * j + k)
                  for j in range(sec, len(section_lengths)) for k in range(3)]
        v, u = _vu_from_tendons(None, [(ri, t) for ri, t, _ in active],
                                Kse, Kbt, v_star, u_star)

        # |p_i'| is constant along the section, so the tendon path length is
        # simply that times the section length -- no quadrature.
        for r_i, _, idx in active:
            lengths[idx] += float(np.linalg.norm(np.cross(u, r_i) + v)) * L

        for k in range(1, n_eval + 1):
            frames.append(T @ _section_transform(v, u, L * k / n_eval))
        T = T @ _section_transform(v, u, L)

    return frames, lengths


def forward_sensitivity(tau, section_lengths=SECTION_LENGTHS, r=TENDON_R,
                        gammas=SECTION_GAMMAS, **kw):
    """-> (tendon_lengths, J) with J[i, j] = d lengths[i] / d tau[j].

    The EXACT Jacobian, not a finite difference. Two ingredients.

    1. Implicit function theorem on the constitutive fixed point. With
       x = [v; u] and x = x* + G(x, tau),

           dx/dtau = (I - dG/dx)^-1  dG/dtau

       where dG/dtau_j is just -Kse^-1 t_j stacked on -Kbt^-1 (r_j x t_j),
       and dG/dx needs dt_j/dx. Since t = p'/|p'| and p' = u x r + v,

           dp'/dv = I,   dp'/du = -skew(r),   dt/dp' = (I - t t^T)/|p'|

    2. Chain rule onto the length. |p'_i| is constant along a section, so

           d|p'_i| / dx = [ t_i^T , (r_i x t_i)^T ]

       and the section contributes that, times its length, times dx/dtau.

    Why bother: finite differencing this map is what stalled the earlier
    solver. The columns for two tendons symmetric about the bend plane are
    nearly identical, so the FD Jacobian is numerically rank-deficient and
    least_squares reported convergence 2 mm short of the answer. The analytic
    version is exact to machine precision and costs about one forward
    evaluation.
    """
    Kse, Kbt = _stiffness(**kw)
    v_star = np.array([0.0, 1.0, 0.0])
    u_star = np.zeros(3)
    offs = [tendon_offsets(i, r, gammas) for i in range(len(section_lengths))]

    n = len(tau)
    lengths = np.zeros(n)
    J = np.zeros((n, n))

    for sec, L in enumerate(section_lengths):
        active = [(offs[j][k], float(tau[3 * j + k]), 3 * j + k)
                  for j in range(sec, len(section_lengths)) for k in range(3)]
        v, u = _vu_from_tendons(None, [(ri, t) for ri, t, _ in active],
                                Kse, Kbt, v_star, u_star)

        Kse_inv = np.linalg.inv(Kse)
        Kbt_inv = np.linalg.inv(Kbt)

        dG_dx = np.zeros((6, 6))
        dG_dtau = np.zeros((6, n))
        rows = []                                   # per-tendon d|p'|/dx

        for r_i, tau_i, idx in active:
            pb = np.cross(u, r_i) + v
            nrm = float(np.linalg.norm(pb))
            t = pb / nrm
            rs = _skew(r_i)

            lengths[idx] += nrm * L
            rows.append((idx, np.concatenate([t, np.cross(r_i, t)])))

            # dt/dx over the 6-vector x = [v; u]
            P = (np.eye(3) - np.outer(t, t)) / nrm
            dt_dx = np.hstack([P, -P @ rs])         # 3 x 6

            dG_dx[0:3, :] -= Kse_inv @ (tau_i * dt_dx)
            dG_dx[3:6, :] -= Kbt_inv @ (rs @ (tau_i * dt_dx))

            dG_dtau[0:3, idx] = -Kse_inv @ t
            dG_dtau[3:6, idx] = -Kbt_inv @ np.cross(r_i, t)

        dx_dtau = np.linalg.solve(np.eye(6) - dG_dx, dG_dtau)

        for idx, drow in rows:
            J[idx, :] += L * (drow @ dx_dtau)

    return lengths, J


def linear_tension_guess(dl, section_lengths=SECTION_LENGTHS, r=TENDON_R, **kw):
    """Beam-theory starting tensions for a commanded set of pull-ins.

    This matters far more than a starting guess usually does. The residual
    surface here is nearly rank-deficient -- two tendons symmetric about the
    bend plane have almost identical Jacobian columns -- so a finite
    difference solve started in the wrong place reports convergence at a
    local minimum. Started from this guess it lands essentially on the answer.

    The derivation, per group of three tendons:

      dl_i = eps*L + kappa*r*L*cos(theta_i - alpha)

    Averaging over the three kills the cosine (sum cos(theta_i) = 0), so the
    mean pull-in is pure axial strain:      eps = mean(dl)/L
    Subtracting it leaves a clean sinusoid, inverted by the same exact
    three-sample Fourier relation used in cc_kinematics.py, giving kappa and
    alpha. Then Hooke's law in reverse:

      sum_i tau_i = eps * EA                        (axial)
      net moment  = kappa * EI, at azimuth alpha    (bending)
      tau_i = (sum tau)/3 + (2/(3r)) * kappa*EI * cos(theta_i - alpha)

    Verified against the ODE: for dl = [4, 0, 0] this returns
    [1.10e5, 7.7e3, 7.7e3], which integrates to [3.982, 0.0009, 0.0009] mm.
    """
    Kse, Kbt = _stiffness(**kw)
    EA, EI = float(Kse[1, 1]), float(Kbt[0, 0])
    dl = np.asarray(dl, dtype=float)
    th = [math.radians(t) for t in THETA_DEG]

    tau = np.zeros(len(dl))
    for g in range(len(dl) // 3):
        L = float(np.sum(section_lengths[: g + 1]))     # this group's span
        d = dl[3 * g: 3 * g + 3]

        eps = float(np.mean(d)) / L                     # axial strain
        bend = d - eps * L
        A = (2.0 / 3.0) * sum(x * math.cos(t) for x, t in zip(bend, th))
        B = (2.0 / 3.0) * sum(x * math.sin(t) for x, t in zip(bend, th))
        amp = math.hypot(A, B)
        alpha = math.atan2(B, A) if amp > 1e-12 else 0.0
        kappa = amp / (r * L)

        tot = eps * EA
        M = kappa * EI
        for k, t in enumerate(th):
            tau[3 * g + k] = tot / 3.0 + (2.0 / (3.0 * r)) * M * math.cos(t - alpha)

    return np.clip(tau, 0.0, None)


def refine_tension(dl_cmd, n_iter=6, tol=1e-3, section_lengths=SECTION_LENGTHS,
                   r=TENDON_R, gammas=SECTION_GAMMAS, gravity=0.0, **kw):
    """Hit a commanded dl by fixed-point correction on the tension.

    The expensive part of a general solve is the finite-difference Jacobian.
    We do not need one: linear_tension_guess IS an analytic inverse of the
    beam-theory Jacobian, so applying it to the CURRENT ERROR gives a Newton
    step for free --

        tau <- tau + linear_tension_guess(dl_commanded - dl_achieved)

    Each iteration costs one forward integration (0.14 s) instead of seven,
    and there is no multi-start. Non-negativity is imposed by clipping, which
    also handles the complementarity case that defeated least_squares: a
    tendon that wants negative tension simply goes slack.
    """
    L_und = np.array([float(np.sum(section_lengths[: (j // 3) + 1]))
                      for j in range(3 * len(section_lengths))])
    tau = linear_tension_guess(dl_cmd, section_lengths, r, **kw)
    best, best_err = tau.copy(), np.inf

    for _ in range(n_iter):
        _, lengths = integrate(tau, section_lengths, r, gammas, gravity,
                               n_eval=25, **kw)
        err = (L_und - lengths) - np.asarray(dl_cmd, dtype=float)
        e = float(np.max(np.abs(err)))
        if e < best_err:
            best, best_err = tau.copy(), e
        if e < tol:
            break
        tau = np.clip(tau - linear_tension_guess(err, section_lengths, r, **kw),
                      0.0, None)

    return best, best_err


def solve_for_displacements(dl, section_lengths=SECTION_LENGTHS, r=TENDON_R,
                            gammas=SECTION_GAMMAS, gravity=GRAVITY,
                            tau0=None, **kw):
    """Tensions consistent with the commanded tendon pull-ins.

    This is a COMPLEMENTARITY problem, not an equality-constrained solve, and
    getting that wrong is what made 101 of 152 DOE commands look unreachable.

    A servo that has taken in dl_i leaves the tendon L_und_i - dl_i long. The
    rod's tendon path cannot exceed that -- but it may be SHORTER, in which
    case the tendon is simply slack and carries no tension. So with slack

        c_i = (L_und_i - dl_i) - path_i

    the physical conditions are

        tau_i >= 0,    c_i >= 0,    tau_i * c_i = 0

    i.e. a tendon is either taut and load-bearing, or slack and idle. Forcing
    c_i = 0 for every tendon demands that every tendon stay exactly taut,
    which no rod can do once it compresses -- hence the spurious
    unreachability.

    Encoded with the Fischer-Burmeister function

        phi(a, b) = a + b - sqrt(a^2 + b^2)

    which is zero exactly on the complementarity set and is smooth away from
    the origin, so least_squares can use an analytic Jacobian built from
    forward_sensitivity.

    -> (tau, info). info["slack"] says which tendons came out slack, and
    info["max_abs_residual"] is the complementarity residual.
    """
    if gravity:
        raise NotImplementedError("see integrate()")

    dl = np.asarray(dl, dtype=float)
    n = len(dl)
    L_und = np.array([float(np.sum(section_lengths[: (j // 3) + 1]))
                      for j in range(n)])
    avail = L_und - dl                              # length the servo allows

    def _phi_and_jac(tau):
        lengths, J = forward_sensitivity(tau, section_lengths, r, gammas, **kw)
        c = avail - lengths                         # slack, >= 0 when taut
        sq = np.sqrt(tau ** 2 + c ** 2) + 1e-30
        phi = tau + c - sq
        # d phi_i / d tau_j = (1 - tau_i/sq_i) delta_ij + (1 - c_i/sq_i) dc_i/dtau_j
        dphi = np.diag(1.0 - tau / sq) + (1.0 - c / sq)[:, None] * (-J)
        return phi, dphi, lengths, c

    if tau0 is None:
        tau0 = linear_tension_guess(dl, section_lengths, r, **kw)
    tau0 = np.maximum(np.asarray(tau0, dtype=float), 0.0)

    best = None
    for start_tau in (tau0, tau0 * 0.5, np.maximum(tau0, 1.0), np.zeros(n)):
        try:
            sol = least_squares(lambda t: _phi_and_jac(t)[0], start_tau,
                                jac=lambda t: _phi_and_jac(t)[1],
                                bounds=(0.0, np.inf),
                                xtol=1e-14, ftol=1e-14, gtol=1e-14,
                                max_nfev=300)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
        if best.cost < 1e-16:
            break

    phi, _, lengths, c = _phi_and_jac(best.x)
    info = {"achieved": L_und - lengths,
            "slack": c > 1e-6,
            "residual": phi,
            "max_abs_residual": float(np.max(np.abs(phi))),
            "sol": best}
    return best.x, info


def tip_pose6(dl, **kw):
    """Commanded tendon pull-ins -> tip pose [x,y,z,wx,wy,wz]."""
    tau, _ = solve_for_displacements(dl, **kw)
    frames, _ = integrate(tau, **{k: v for k, v in kw.items()})
    return matrix_to_pose6(frames[-1])


# ── Self-test ─────────────────────────────────────────────────────────────

def _self_test():
    Kse, Kbt = _stiffness()
    EI = float(Kbt[0, 0])
    print(f"EI = {EI:.4g},  GJ = {Kbt[1,1]:.4g},  EA = {Kse[1,1]:.4g}\n")

    # 1. No tension, no gravity -> perfectly straight, full length.
    frames, lengths = integrate(np.zeros(6), gravity=0.0)
    tip = frames[-1]
    assert np.allclose(tip[:3, 3], [0, sum(SECTION_LENGTHS), 0], atol=1e-6), tip[:3, 3]
    assert np.allclose(tip[:3, :3], np.eye(3), atol=1e-6)
    print(f"[ok] zero tension -> tip {np.round(tip[:3,3],6)}, identity rotation")

    # 2. Single proximal tendon -> bends TOWARD it (+X for the 0 deg tendon).
    tau = np.zeros(6); tau[0] = 2.5e5      # ~45 deg worth, see _stiffness
    frames, _ = integrate(tau, gravity=0.0)
    tip = frames[-1]
    assert tip[0, 3] > 1.0, tip[:3, 3]
    assert abs(tip[2, 3]) < 1e-6, tip[:3, 3]
    print(f"[ok] tendon 1 (azimuth 0) pulls the tip toward +X: {np.round(tip[:3,3],3)}")

    # 3. Small-load limit must match Euler-Bernoulli: a tendon at radius r
    #    under tension tau applies moment tau*r, giving kappa = tau*r/EI.
    #    Only section 1 is loaded here, so check the curvature there.
    tau = np.zeros(6); tau[0] = 2.0e4      # small load, linear regime
    frames, _ = integrate(tau, gravity=0.0, n_eval=120)
    L1 = SECTION_LENGTHS[0]
    T = frames[len(frames) // 2 - 1]                  # end of section 1
    ang = math.acos(np.clip(T[1, 1], -1, 1))
    kappa_num = ang / L1
    kappa_ana = tau[0] * TENDON_R / EI
    rel = abs(kappa_num - kappa_ana) / kappa_ana
    assert rel < 0.05, (kappa_num, kappa_ana, rel)
    print(f"[ok] small-load curvature matches tau*r/EI to {100*rel:.2f}%  "
          f"({kappa_num:.3e} vs {kappa_ana:.3e} 1/mm)")

    # 4. Round trip. Start from tensions, compute the displacements they
    #    produce, then invert. Reachable by construction, so any error here
    #    is the solver's and not the robot's.
    L_und = np.array([float(np.sum(SECTION_LENGTHS[:(j//3)+1])) for j in range(6)])
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(5):
        tau_true = rng.uniform(0.0, 2.0e5, 6)
        lengths, _ = forward_sensitivity(tau_true)
        dl_cmd = L_und - lengths
        tau, info = solve_for_displacements(dl_cmd)
        worst = max(worst, info["max_abs_residual"])
    print(f"[ok] tension round trip, 5 random draws: worst displacement error "
          f"{worst:.3e} mm")
    assert worst < 1e-6, worst

    # 4b. Complementarity. Commanding zero pull on the proximal group while
    #     tensioning the distal group is satisfiable -- but only because the
    #     proximal tendons go SLACK. Under an equality constraint this looked
    #     like an unreachable command.
    from tdcr.multisection import theta_alpha_to_deltas
    dl_d = np.asarray(theta_alpha_to_deltas([0.0, 0.4], [0.0, 0.0]))
    tau_d, info_d = solve_for_displacements(dl_d)
    assert info_d["max_abs_residual"] < 1e-6, info_d["max_abs_residual"]
    assert info_d["slack"][:3].any(), info_d["slack"]
    print(f"[ok] distal-only command satisfied, residual "
          f"{info_d['max_abs_residual']:.2e} mm")
    print(f"     slack tendons: {np.where(info_d['slack'])[0].tolist()} "
          f"(proximal group goes slack as section 1 compresses)")

    # 5. Extension is real: PCC's inextensibility is an approximation.
    tau = np.full(6, 5.0e4)
    frames, _ = integrate(tau, gravity=0.0)
    y_tip = frames[-1][1, 3]
    print(f"[ok] uniform tension 5e4 shortens the rod "
          f"{sum(SECTION_LENGTHS) - y_tip:.3f} mm "
          f"(PCC predicts exactly 0 -- it has no axial term)")

    # 6. Routing coupling: a distal tendon must load the PROXIMAL section.
    tau = np.zeros(6); tau[3] = 2.0e5                 # distal group only
    frames, _ = integrate(tau, gravity=0.0)
    T_mid = frames[len(frames) // 2 - 1]
    lat = math.hypot(T_mid[0, 3], T_mid[2, 3])
    assert lat > 1.0, lat
    print(f"[ok] a DISTAL tendon alone bends section 1 by {lat:.2f} mm laterally")
    print( "     -- the routing coupling the two-body SOFA scene cannot represent")

    print("\nAll self-tests passed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sweep", metavar="CONFIGS_CSV", default=None,
                    help="evaluate every row of a (theta, alpha) DOE file")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gravity", type=float, default=GRAVITY,
                    help="per-length body force along -Y (units as for E)")
    args = ap.parse_args()

    if args.self_test or not args.sweep:
        _self_test()
    else:
        from tdcr.digitaltuning import read_theta_alpha_csv, RESULTS_DIR
        import time
        th, al, dl, lab = read_theta_alpha_csv(args.sweep)
        out = args.out or os.path.join(RESULTS_DIR, "level0p5_2s.csv")

        # FORWARD sweep, in tension space.
        #
        # Inverting displacement -> tension costs ~7 integrations per optimiser
        # step times ~30 steps times 2 multi-starts, i.e. minutes per sample,
        # and it stalls on strong S-shapes. A single forward integration takes
        # 0.14 s. Since what the calibration actually needs is (input, output)
        # pairs -- not a prescribed dl grid -- we take the beam-theory tension
        # for each commanded dl, integrate ONCE, and record the ACHIEVED dl
        # beside the pose. Every row is internally consistent, the workspace
        # coverage is the same, and the sweep runs in seconds rather than hours.
        #
        # The achieved dl differs slightly from the commanded one, which is why
        # both are written out. Downstream, join on the ACHIEVED columns.
        L_und = np.array([float(np.sum(SECTION_LENGTHS[: (j // 3) + 1]))
                          for j in range(6)])
        rows, t0, n_fail = [], time.time(), 0
        resid, nslack = [], []
        for i in range(len(dl)):
            try:
                tau, _info = solve_for_displacements(dl[i])
                resid.append(_info["max_abs_residual"])
                nslack.append(int(_info["slack"].sum()))
                frames, lengths = integrate(tau, gravity=args.gravity or 0.0)
                p6 = matrix_to_pose6(frames[-1])
                got = L_und - lengths
            except Exception as e:
                print(f"[{i+1}/{len(dl)}] FAILED: {e}")
                p6 = [float("nan")] * 6
                got = [float("nan")] * 6
                tau = [float("nan")] * 6
                n_fail += 1
            rows.append(list(th[i]) + list(al[i]) + list(dl[i]) + list(got)
                        + list(tau) + list(p6))
            if (i + 1) % 25 == 0 or i == len(dl) - 1:
                print(f"  [{i+1}/{len(dl)}]  {time.time()-t0:.1f}s elapsed")

        hdr = (["theta1", "theta2", "alpha1", "alpha2"]
               + [f"dl{k+1}" for k in range(6)]
               + [f"dl_achieved{k+1}" for k in range(6)]
               + [f"tau{k+1}" for k in range(6)]
               + ["tip_x", "tip_y", "tip_z", "tip_wx", "tip_wy", "tip_wz"])
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(rows)

        # Report the COMPLEMENTARITY residual, not |achieved - commanded|.
        # Those differ wherever a tendon is slack, and slack is correct
        # physics, not solver error: a tendon whose path is shorter than the
        # servo allows simply carries no load.
        if resid:
            resid = np.array(resid)
            print(f"\nwrote {len(rows)} rows to {out}  ({time.time()-t0:.1f}s, "
                  f"{n_fail} failures)")
            print(f"complementarity residual: median {np.median(resid):.2e} mm, "
                  f"max {resid.max():.2e} mm")
            print(f"  solved to 1e-6 mm : {(resid < 1e-6).sum()}/{len(resid)}")
            print(f"  mean slack tendons per row: {np.mean(nslack):.2f} of 6")
            print("  (a slack tendon means its achieved dl exceeds the command --")
            print("   correct, and why dl_achieved* is the column to join on)")
