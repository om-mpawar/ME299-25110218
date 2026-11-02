"""
SOFA sweep that keeps the WHOLE deformed body, not just the tip (1 or 2 sections).

--sections 1 uses the single-section cylinder scene (sim/cylinder_scene.py);
--sections 2 uses the ROUTED two-section scene (sim/two_section_scene.py,
createSceneRouted): one merged mesh, section-2 tendons running through
section 1 as on the hardware, sections joined by the spring web.

For every command in a (theta, alpha) DOE the scene is reset, the tendons are
pulled, the body settles, and the positions of ALL mesh nodes are stored. The
backbone centreline, the orientation at any arc length, and the shape error are
all derived afterwards from these node positions (shape_analysis.py), so the
simulation never has to be re-run to try a different shape metric.

The file is re-written after EVERY pose, so an interrupted sweep keeps what it
has done; --resume continues from the first missing pose.

A settling check is stored per pose: how far the tip moved over the last
`--check-steps` steps. A pose counts as settled when that is below 1e-3 mm.

    .venv/bin/python sim/shape_sweep.py --doe results/configs_1s_ta.csv \
        --out results/shape_1s_ta.npz

(SOFA 25.06 on PYTHONPATH / LD_LIBRARY_PATH, as for cylinder_scene.py.)

Output (.npz):
    theta, alpha      (P,)       command, radians
    dl                (P, 3)     tendon pulls, mm (pull-only offset applied)
    rest              (N, 3)     node positions at rest
    pos               (P, N, 3)  node positions after settling
    settle_mm         (P,)       tip movement over the last check-steps
    mesh, settle_steps, dt       run metadata

--pose-dir DIR writes one small file per pose instead (DIR/pose_0123.npz, named
by the DOE row; positions as float32, ~6e-5 mm rounding at 880 mm) plus
DIR/rest.npz, and skips poses whose file exists. Used by the cloud sessions,
which push each finished pose to GitHub; sim/merge_poses.py assembles the
standard file above from one or more such directories.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root, for `import tdcr`
import argparse
import csv
import os
import sys
import time

import numpy as np

import Sofa.Core
import Sofa.Simulation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cylinder_scene import createScene, find_tip_index  # noqa: E402
from tdcr.paths import MESHES  # noqa: E402

from tdcr.paths import ROOT as _DIR  # paths below are relative to the repo root
def load_doe(path, n_sections=1):
    """-> theta (P,) or (P, n), alpha likewise, dl (P, 3n)."""
    rows = list(csv.DictReader(open(path)))
    theta = np.array([[float(r[f"theta{j+1}"]) for j in range(n_sections)] for r in rows])
    alpha = np.array([[float(r[f"alpha{j+1}"]) for j in range(n_sections)] for r in rows])
    dl = np.array([[float(r[f"dl{i+1}"]) for i in range(3 * n_sections)] for r in rows])
    if n_sections == 1:
        theta, alpha = theta[:, 0], alpha[:, 0]
    return theta, alpha, dl


def nn_order(dl, idx):
    """Greedy nearest-neighbour tour through the commands (tendon pulls), starting nearest zero."""
    left = list(idx)
    cur = min(left, key=lambda k: np.linalg.norm(dl[k]))
    out = [cur]; left.remove(cur)
    while left:
        cur = min(left, key=lambda k: np.linalg.norm(dl[k] - dl[cur]))
        out.append(cur); left.remove(cur)
    return out


def build(root, n_sections, mesh, cables_json=None):
    """-> (body node with .dofs, cables, tip index)."""
    if n_sections == 1 and not cables_json:
        _, body, cables = createScene(root, mesh_vtk=mesh, include_visual=False, add_controller=False)
        import Sofa.Simulation
        Sofa.Simulation.init(root)
        return body, cables, find_tip_index(body.dofs)
    import two_section_scene as _ts2
    if n_sections == 2 and os.environ.get("TDCR_LEGACY_2S") == "1":   # the original two-body scene
        from two_section_scene import createSceneRouted
        _, body, _off, cables = createSceneRouted(root, mesh=mesh, include_visual=False, add_controller=False,
                                                 cables_json=cables_json)
    else:
        from multi_section_scene import createSceneRoutedN
        _, body, _off, cables = createSceneRoutedN(root, mesh, cables_json, n_sections)
    import Sofa.Simulation
    Sofa.Simulation.init(root)
    rest = np.array(body.dofs.position.value, dtype=float)
    top = rest[:, 1].max()
    tip = int(np.argmin(np.hypot(rest[:, 0], rest[:, 2]) + 100 * np.abs(rest[:, 1] - top)))
    return body, cables, tip


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", type=int, default=1,
                    help="1 (cylinder scene; with --cables the routed multi_section_scene), 2 (two-section scene) or more (multi_section_scene; needs --mesh and --cables)")
    ap.add_argument("--doe", default=None, help="default: results/configs_{1s,2s}_disk.csv")
    ap.add_argument("--out", default=None, help="default: results/shape_{1s,2s}_disk.npz")
    ap.add_argument("--mesh", default=None,
                    help="default: cyl_nom_b.vtk (1 section) or two_section.vtk (2 sections)")
    ap.add_argument("--settle-steps", type=int, default=300)
    ap.add_argument("--check-steps", type=int, default=20)
    ap.add_argument("--cables", default=None,
                    help="tendon paths (cables.json), two sections only; default: data/cables/two_section.cables.json")
    ap.add_argument("--pretension", type=float, default=0.0,
                    help="extra pull on EVERY tendon, mm, added to the DOE's dl (the robot's "
                         "pretension; two or more sections need it to stand under real gravity)")
    ap.add_argument("--ramp-steps", type=int, default=0,
                    help="apply the tendon pulls gradually over this many steps (needed with contact)")
    ap.add_argument("--dt", type=float, default=None, help="time step, s (default: the scene's, 0.01)")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="stop a pose as soon as the tip moves less than this (mm) over --check-steps "
                         "steps, after the ramp (0: always run --settle-steps)")
    ap.add_argument("--contact-above", type=float, default=None,
                    help="with disc contact in the scene: switch it on only for poses where some section "
                         "bends more than this (deg); below ~45 deg no notch gap closes (gap check). "
                         "With --contact-auto: poses above this go straight to contact, without the trial")
    ap.add_argument("--contact-auto", type=float, default=None, metavar="GAP_MM",
                    help="with disc contact in the scene: settle each pose WITHOUT contact first and redo it "
                         "with contact only if a notch gap ends below GAP_MM (sim/contact_gap.py). Exact: "
                         "an open gap carries no contact force, so that pose is the same either way")
    ap.add_argument("--ramp-steps-nocontact", type=int, default=None,
                    help="ramp for poses run without contact (default: --ramp-steps)")
    ap.add_argument("--extrapolate", type=int, default=0, metavar="N",
                    help="without contact, every N steps after the ramp try an Aitken jump to the "
                         "equilibrium: when the body creeps along one slowly decaying mode (two sections "
                         "under gravity: ~4 %% per step, because the corotational FEM has no "
                         "geometric stiffness), the rest of the path is d*rho/(1-rho). 0: off")
    ap.add_argument("--continuation", action="store_true",
                    help="no reset between poses: visit them in nearest-neighbour order of the tendon "
                         "pulls and start each from the previous equilibrium")
    ap.add_argument("--rows", default=None, help="e.g. 0:3 to run a slice (smoke test)")
    ap.add_argument("--pose-dir", default=None,
                    help="one file per pose in this directory, instead of --out (resumes by itself)")
    ap.add_argument("--resume", action="store_true",
                    help="keep the poses already in --out and run only the missing ones")
    args = ap.parse_args()
    tag = f"{args.sections}s"
    args.doe = args.doe or os.path.join(_DIR, "results", f"configs_{tag}_disk.csv")
    args.out = args.out or os.path.join(_DIR, "results", f"shape_{tag}_disk.npz")
    if args.sections > 2 and not (args.mesh and args.cables):
        ap.error("--sections > 2 needs --mesh and --cables (e.g. four_section_cad.vtk / .cables.json)")
    args.mesh = args.mesh or os.path.join(MESHES, "cyl_nom_b.vtk" if args.sections == 1 else "two_section.vtk")

    import two_section_scene as _ts                 # material actually used (env overrides)
    material = dict(young_kpa=_ts.YOUNG_MODULUS, poisson=_ts.POISSON_RATIO,
                    section_mass_kg=_ts.SECTION_MASS, gravity_mm_s2=_ts.GRAVITY, gravity_dir_lab=_ts.GRAVITY_DIR,
                    pretension_mm=args.pretension, contact=int(_ts.CONTACT),
                    stops=int(_ts.STOPS), stop_frac=_ts.STOP_FRAC, stop_k=_ts.STOP_K, stop_kd=_ts.STOP_KD,
                    struts=int(_ts.STRUTS), strut_frac=_ts.STRUT_FRAC, strut_maxgap=_ts.STRUT_MAXGAP, strut_rim=_ts.STRUT_RIM,
                    contact_alarm_mm=_ts.CONTACT_ALARM, contact_dist_mm=_ts.CONTACT_DIST,
                    ramp_steps=args.ramp_steps, tol_mm=args.tol, continuation=int(args.continuation),
                    extrapolate=args.extrapolate,
                    ode=os.environ.get("TDCR_ODE", "euler"), fem_method=_ts.FEM_METHOD,
                    contact_above_deg=-1.0 if args.contact_above is None else args.contact_above,
                    contact_auto_mm=-1.0 if args.contact_auto is None else args.contact_auto)
    theta, alpha, dl = load_doe(args.doe, args.sections)
    dl = dl + args.pretension
    rows = np.arange(len(theta))                     # DOE row of every pose kept
    if args.rows:
        a, b = (int(v) if v else None for v in args.rows.split(":"))
        theta, alpha, dl, rows = theta[a:b], alpha[a:b], dl[a:b], rows[a:b]

    root = Sofa.Core.Node("root")
    tdcr, cables, tip = build(root, args.sections, args.mesh, args.cables)
    if args.dt:
        root.dt.value = args.dt
    rest = np.array(tdcr.dofs.position.value, dtype=float)
    def collision_models(node):
        out = [o for o in node.objects if o.getClassName().endswith("CollisionModel")]
        for ch in node.children:
            out += collision_models(ch)
        return out
    contact_models = collision_models(tdcr.getChild("contact")) if tdcr.getChild("contact") is not None else []
    gaps = None
    if contact_models and args.contact_auto is not None:
        from contact_gap import GapCheck, contact_file
        gaps = GapCheck(contact_file(args.mesh), rest=rest)
    elif args.contact_auto is not None:
        ap.error("--contact-auto needs disc contact in the scene (TDCR_CONTACT=1)")
    print(f"[shape_sweep] {len(rest)} nodes, {len(theta)} poses, mesh {os.path.basename(args.mesh)}",
          flush=True)

    pos = np.full((len(theta), len(rest), 3), np.nan)
    settle = np.full(len(theta), np.nan)
    steps_used = np.full(len(theta), -1)
    secs = np.full(len(theta), np.nan)
    if args.resume and os.path.exists(args.out):
        old = np.load(args.out)
        assert np.allclose(old["theta"], theta) and np.allclose(old["alpha"], alpha), \
            "--resume: the existing file was made from a different DOE"
        pos, settle = old["pos"].copy(), old["settle_mm"].copy()
    if args.pose_dir:
        os.makedirs(args.pose_dir, exist_ok=True)
        pose_file = lambda i: os.path.join(args.pose_dir, f"pose_{rows[i]:04d}.npz")
        rest_file = os.path.join(args.pose_dir, "rest.npz")
        if not os.path.exists(rest_file):
            tmp = f"{rest_file}.{os.getpid()}.tmp.npz"     # several jobs may write it at once
            np.savez_compressed(tmp, rest=rest, mesh=os.path.basename(args.mesh),
                                settle_steps=args.settle_steps, dt=root.dt.value, **material)
            os.replace(tmp, rest_file)
        settle[[i for i in range(len(theta)) if os.path.exists(pose_file(i))]] = 0.0   # marks done
    todo = [i for i in range(len(theta)) if np.isnan(settle[i])]
    print(f"[shape_sweep] {len(theta) - len(todo)} poses already done, {len(todo)} to run", flush=True)

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        tmp = args.out + ".tmp.npz"           # write then rename: never a half-written file
        np.savez_compressed(tmp, theta=theta, alpha=alpha, dl=dl, rest=rest, pos=pos,
                            settle_mm=settle, mesh=os.path.basename(args.mesh), steps_used=steps_used,
                            seconds=secs, settle_steps=args.settle_steps, dt=root.dt.value, **material)
        os.replace(tmp, args.out)

    def aitken(hist):
        """Jump target from the last three states if they move along one geometrically decaying
        mode (consecutive steps parallel, constant ratio rho), else None."""
        d1, d2 = hist[1] - hist[0], hist[2] - hist[1]
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 == 0 or n2 == 0:
            return None
        cos = float(np.vdot(d1, d2) / (n1 * n2))
        rho = float(np.vdot(d1, d2) / n1 ** 2)
        if cos < 0.995 or not 0.3 < rho < 0.995:
            return None
        return hist[2] + d2 * min(rho / (1.0 - rho), 100.0)

    def settle_pose(start, target, use_contact):
        """Ramp the pulls from `start` to `target` and settle. -> (steps, tip movement at the end, jumps)."""
        for o in contact_models:
            o.active.value = bool(use_contact)
        ramp = args.ramp_steps if (use_contact or args.ramp_steps_nocontact is None) else args.ramp_steps_nocontact
        steps, last, settle_i = 0, np.array(tdcr.dofs.position.value[tip], dtype=float), np.inf
        hist, jumps = [], 0
        while steps < args.settle_steps:
            if steps <= ramp:                         # ramp the pulls from the previous command
                f = 1.0 if ramp == 0 else min(1.0, (steps + 1) / ramp)
                for c, a0, v in zip(cables, start, target):
                    c.CableConstraint.value = [float(a0 + f * (v - a0))]
            Sofa.Simulation.animate(root, root.dt.value)
            steps += 1
            if args.extrapolate and not use_contact and steps > ramp + 1:
                hist = (hist + [np.array(tdcr.dofs.position.value, dtype=float)])[-(args.extrapolate + 1):]
                if len(hist) == args.extrapolate + 1:             # N steps since the ramp or the last jump
                    x = aitken(hist[-3:])
                    if x is not None:                 # jump; the next settle check sees the jump, so
                        with tdcr.dofs.position.writeableArray() as w:     # it cannot stop early on it
                            w[:] = x
                        with tdcr.dofs.velocity.writeableArray() as w:
                            w[:] = 0.0
                        jumps += 1
                    hist = []
            if steps % args.check_steps == 0:
                cur = np.array(tdcr.dofs.position.value[tip], dtype=float)
                if not np.all(np.isfinite(cur)) or np.abs(cur).max() > 1e4:   # blew up: give up on the pose
                    return steps, np.inf, -1
                if steps > ramp + args.check_steps // 2:
                    settle_i = np.linalg.norm(cur - last)
                    if args.tol > 0 and settle_i < args.tol:
                        break
                last = cur
        return steps, settle_i, jumps

    t0 = time.time()
    order = nn_order(dl, todo) if (args.continuation and len(todo) > 1) else todo
    prev = np.zeros(dl.shape[1])
    for i in order:
        tp = time.time()
        if not args.continuation:
            Sofa.Simulation.reset(root)
            prev = np.zeros(dl.shape[1])
        start, target = prev.copy(), dl[i]
        bend = np.degrees(np.max(np.atleast_1d(theta[i])))
        above = args.contact_above is not None and bend > args.contact_above
        if not contact_models:
            use_contact = False
        elif gaps is None:                            # fixed rule: contact everywhere, or above an angle
            use_contact = args.contact_above is None or above
        else:                                         # contact only where a gap closes
            use_contact = above
        x0 = np.array(tdcr.dofs.position.value, dtype=float)
        v0 = np.array(tdcr.dofs.velocity.value, dtype=float)
        steps, settle_i, jumps = settle_pose(start, target, use_contact)
        if jumps < 0:                          # diverged: record it and move on (the claim stops retries)
            print(f"[{i+1}/{len(theta)}] row {rows[i]} DIVERGED after {steps} steps", flush=True)
            if args.pose_dir:
                open(os.path.join(args.pose_dir, f"failed_{rows[i]:04d}.txt"), "w").write(
                    f"diverged after {steps} steps ({time.time() - tp:.0f} s)\n")
            continue
        trial_gap = gaps.min_gap(tdcr.dofs.position.value) if gaps is not None else np.nan
        if gaps is not None and not use_contact and trial_gap < args.contact_auto:
            if args.continuation:                     # back to where this pose started
                tdcr.dofs.position.value = x0.tolist()
                tdcr.dofs.velocity.value = v0.tolist()
            else:
                Sofa.Simulation.reset(root)
            use_contact = True
            more, settle_i, _ = settle_pose(start, target, True)
            steps += more
        final_gap = gaps.min_gap(tdcr.dofs.position.value) if gaps is not None else np.nan
        prev = target.copy()
        pos[i] = np.array(tdcr.dofs.position.value, dtype=float)
        settle[i], steps_used[i], secs[i] = settle_i, steps, time.time() - tp
        cmd = " ".join(f"{v:5.1f}" for v in np.degrees(np.atleast_1d(theta[i])))
        dirs = " ".join(f"{v:6.1f}" for v in np.degrees(np.atleast_1d(alpha[i])))
        gap_txt = (f" gap {trial_gap:.2f}->{final_gap:.2f} mm{' contact' if use_contact else ''}"
                   if gaps is not None else "") + (f" {jumps} jumps" if args.extrapolate else "")
        print(f"[{i+1}/{len(theta)}] theta={cmd} alpha={dirs} "
              f"tip={np.round(pos[i, tip], 1)} settle={settle[i]:.2e} mm{gap_txt}  "
              f"({time.time() - t0:.0f}s; this pose {steps_used[i]} steps, {secs[i]:.0f}s)", flush=True)
        if args.pose_dir:
            tmp = pose_file(i) + ".tmp.npz"
            np.savez_compressed(tmp, row=rows[i], theta=theta[i], alpha=alpha[i], dl=dl[i],
                                pos=pos[i].astype(np.float32), settle_mm=settle[i],
                                steps_used=steps_used[i], seconds=secs[i], contact_on=use_contact,
                                trial_gap_mm=trial_gap, min_gap_mm=final_gap, jumps=jumps,
                                contact_auto_mm=-1.0 if args.contact_auto is None else args.contact_auto)
            os.replace(tmp, pose_file(i))
        else:
            save()

    if args.pose_dir:
        print(f"[shape_sweep] rows {rows[0]}-{rows[-1]} complete in {args.pose_dir}")
        return
    save()
    unsettled = int((settle > 1e-3).sum())
    print(f"[shape_sweep] wrote {os.path.relpath(args.out, _DIR)}; "
          f"{unsettled}/{len(theta)} poses moved more than 1e-3 mm in the last {args.check_steps} steps")


if __name__ == "__main__":
    main()
