"""Figures of the study guide, computed from the code and the result files (run from the repository root):
    .venv/bin/python docs/study_guide/make_figs.py
Every number in a figure comes from tdcr/ or results/; nothing is drawn by hand."""
import json
import math
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
OUT = os.path.join(ROOT, "docs", "study_guide", "figs")
os.makedirs(OUT, exist_ok=True)

from analysis.module_compliance import HOLES6, MASS_STD6, ModuleModel, load_set   # noqa: E402
from tdcr import eb_prior as ebp                                                  # noqa: E402
from tdcr.eb_prior import EBPrior, capstan_factors, predict_gravity, section_profile  # noqa: E402
from tdcr.twin import Twin                                                        # noqa: E402

INK, MUTED, GRID = "#222222", "#6b6a66", "#e2e1dc"
BLUE, ORANGE, AQUA, YELLOW, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#8a8984"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("  ", name)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ 1. cross-section material maps
def fig_sections():
    g = json.load(open(ebp.CAD_JSON))
    R, ri, rd, e = g["outer_dia"] / 2, g["inner_dia"] / 2, g["disc_dia"] / 2, g["offset"]
    ca, cb = np.array(g["notch_centres_A"]), np.array(g["notch_centres_B"])
    dc = 0.05
    xs = np.arange(-R + dc / 2, R, dc)
    X, Z = np.meshgrid(xs, xs, indexing="ij")
    ring = (np.hypot(X, Z) >= ri) & (np.hypot(X, Z) <= R)
    for h in HOLES6:
        ring &= np.hypot(X - 9.5 * math.cos(math.radians(h)), Z - 9.5 * math.sin(math.radians(h))) >= 0.9
    hz = np.where(np.abs(Z) > e, rd - np.sqrt(np.maximum(rd ** 2 - Z ** 2, 0)), -1.0)
    hx = np.where(np.abs(X) > e, rd - np.sqrt(np.maximum(rd ** 2 - X ** 2, 0)), -1.0)
    cases = [("plain sleeve\n$y=5$ mm", 5.0), ("A notch\n(centre)", ca[1]),
             ("2 mm above\nthe A notch", ca[1] + 2.0), ("B notch\n(centre)", cb[1])]
    fig, axs = plt.subplots(1, 4, figsize=(6.6, 1.95))
    for ax, (title, y) in zip(axs, cases):
        dA, dB = np.abs(y - ca).min(), np.abs(y - cb).min()
        keep = ring & ~((dA < hz) | (dB < hx))
        A = keep.sum() * dc * dc
        Jx, Jz = (X[keep] ** 2).sum() * dc * dc, (Z[keep] ** 2).sum() * dc * dc
        ax.imshow(np.where(keep, 1.0, np.where(np.hypot(X, Z) <= R, 0.25, 0.0)).T, origin="lower", cmap="Greys",
                  vmin=0, vmax=1.15, extent=[-R, R, -R, R])
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.text(0, -R - 2.2, f"$A$={A:.0f} mm$^2$\n$J_x$={Jx:.0f}, $J_z$={Jz:.0f} mm$^4$", ha="center", va="top",
                fontsize=6.8, color=INK)
    save(fig, "sections")


# ------------------------------------------------------------------ 2. D30 vs D40 profiles
def fig_disc():
    from analysis.module_compliance import HOLES6 as H
    y30, A30, Jx30, _ = section_profile(holes=H)
    ebp_json, ebp_cache = ebp.CAD_JSON, ebp.CACHE
    ebp.CAD_JSON = os.path.join(ROOT, "cad", "std6_tdcr_L150_D40.json")
    ebp.CACHE = os.path.join(ROOT, "results", "eb_profile_D40.npz")
    y40, A40, Jx40, _ = section_profile(holes=H)
    ebp.CAD_JSON, ebp.CACHE = ebp_json, ebp_cache
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 2.9), sharex=True)
    for ax, (a, b, lab) in zip(axs, ((A30, A40, "$A(y)$ (mm$^2$)"), (Jx30, Jx40, "$J_x(y)$ (mm$^4$)"))):
        ax.plot(y30, a, color=GREY, lw=0.8, label="30 mm disc (simulated, printed first)")
        ax.plot(y40, b, color=BLUE, lw=0.8, label="40 mm disc, same 10+9 notches")
        ax.set_ylabel(lab)
        grid(ax)
    axs[1].set_yscale("log")
    axs[1].set_xlabel("$y$ along the module (mm)")
    axs[0].legend(loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    save(fig, "disc")


# ------------------------------------------------------------------ 3. G heatmap, 2 modules (identified)
def fig_G():
    tw = Twin(2, (0.0, 0.0, 0.0))
    G = tw.eb.G
    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    lim = np.abs(G).max()
    im = ax.imshow(G, cmap="RdBu_r", vmin=-lim, vmax=lim)
    labs = ["s1t1", "s1t2", "s1t3", "s2t1", "s2t2", "s2t3"]
    ax.set_xticks(range(6)); ax.set_xticklabels(labs, rotation=45, fontsize=7)
    ax.set_yticks(range(6)); ax.set_yticklabels(labs, fontsize=7)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{G[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(G[i, j]) > 0.55 * lim else INK)
    ax.axhline(2.5, color=INK, lw=0.6); ax.axvline(2.5, color=INK, lw=0.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("$G_{ij}$ (mm$^{-1}$)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    save(fig, "G")
    return G


# ------------------------------------------------------------------ 4. the QP in two dimensions
def fig_qp():
    H = np.array([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, 0.0])
    t = np.linspace(-0.6, 1.0, 300)
    T1, T2 = np.meshgrid(t, t)
    Q = 0.5 * (H[0, 0] * T1 ** 2 + 2 * H[0, 1] * T1 * T2 + H[1, 1] * T2 ** 2) - b[0] * T1 - b[1] * T2
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    ax.fill_between([0, 1.0], 0, 1.0, color="#eef4fb", zorder=0)
    ax.contour(T1, T2, Q, levels=np.linspace(-0.3, 0.8, 12), colors=GREY, linewidths=0.6)
    un = np.linalg.solve(H, b)
    ax.plot(*un, "x", color=ORANGE, ms=7, mew=1.6)
    ax.annotate("unconstrained minimum $(2/3,\\,-1/3)$:\ntendon 2 would have to push", un, (-0.55, -0.55), fontsize=7,
                color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7))
    ax.plot(0.5, 0.0, "o", color=BLUE, ms=6)
    ax.annotate("pull-only solution $(1/2,\\,0)$\ntendon 2 slack by 0.5 mm", (0.5, 0.0), (0.08, 0.5), fontsize=7,
                color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7))
    ax.axhline(0, color=INK, lw=0.8); ax.axvline(0, color=INK, lw=0.8)
    ax.text(0.05, 0.88, "feasible region: $T_1, T_2\\geq0$", fontsize=7, color=BLUE)
    ax.set_xlabel("$T_1$ (N)"); ax.set_ylabel("$T_2$ (N)")
    ax.set_xlim(-0.6, 1.0); ax.set_ylim(-0.6, 1.0)
    save(fig, "qp")


# ------------------------------------------------------------------ 5. Proposition 1: same commands, other stiffness
def fig_prop1():
    st = np.arange(0.0, 160.0 + 1e-9, 0.5)
    dl = np.array([[3.0, 0.5, 0.5]])
    eb = EBPrior(1, holes=HOLES6)
    C0, R0, lam0 = eb.predict(dl, st)
    eb2 = EBPrior(1, holes=HOLES6)
    f = np.where(eb2.yc < 80.0, 3.0, 1.0 / 3.0)
    eb2.cX, eb2.cZ = eb2.cX * f, eb2.cZ * f
    eb2.assemble()
    C1, R1, lam1 = eb2.predict(dl, st)
    ang = lambda R: math.degrees(math.acos(np.clip(R[1, 1], -1, 1)))
    fig, ax = plt.subplots(figsize=(3.0, 3.2))
    ax.plot(C0[0, :, 0], C0[0, :, 1], color=BLUE, lw=1.6, label="CAD stiffness profile")
    ax.plot(C1[0, :, 0], C1[0, :, 1], color=ORANGE, lw=1.6, ls="--",
            label="lower half 3$\\times$ softer,\nupper half 3$\\times$ stiffer")
    for C, col in ((C0, BLUE), (C1, ORANGE)):
        ax.plot(C[0, -1, 0], C[0, -1, 1], "o", color=col, ms=4)
    ax.set_xlim(-5, 40)
    ax.set_aspect("equal")
    ax.set_xlabel("sideways (mm)"); ax.set_ylabel("along the base axis (mm)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.8)
    ax.set_title(f"$\\delta\\ell=(3,0.5,0.5)$ mm, all taut: tip angle {ang(R0[0, -1]):.2f}$^\\circ$ in both;\n"
                 f"tip moves {np.linalg.norm(C0[0, -1] - C1[0, -1]):.1f} mm", fontsize=7)
    grid(ax, "both")
    save(fig, "prop1")
    return ang(R0[0, -1]), ang(R1[0, -1]), np.linalg.norm(C0[0, -1] - C1[0, -1]), lam0, lam1


# ------------------------------------------------------------------ 6. slack on our robot
def fig_slack():
    eb = EBPrior(1, holes=HOLES6)
    cases = [("(2, $-1$, $-1$)", [2.0, -1.0, -1.0]), ("(3, 0.5, 0.5)", [3.0, 0.5, 0.5]),
             ("(1, 1, 1)", [1.0, 1.0, 1.0]), ("(0, $-1$, 0)", [0.0, -1.0, 0.0])]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    w = 0.25
    for k, (lab, dl) in enumerate(cases):
        lam = eb.tensions(np.array([dl]))[0] * 56.0
        for i in range(3):
            ax.bar(k + (i - 1) * w, lam[i], w * 0.9, color=(BLUE, ORANGE, AQUA)[i],
                   label=f"tendon {i + 1}" if k == 0 else None)
            if lam[i] < 1e-9:
                ax.text(k + (i - 1) * w, 0.3, "slack", rotation=90, ha="center", va="bottom", fontsize=6, color=MUTED)
    ax.set_xticks(range(len(cases))); ax.set_xticklabels([c[0] for c in cases], fontsize=7)
    ax.set_xlabel("$\\delta\\ell$ (mm), one module, plane-section model")
    ax.set_ylabel("tension $T=56\\lambda$ (N)")
    ax.legend(fontsize=6.8, ncol=3, loc="upper right")
    grid(ax)
    save(fig, "slack")


# ------------------------------------------------------------------ 7. gravity fixed point, and the stopping bug
def fig_fixedpoint():
    tw = Twin(2, "upright")
    rows = []
    for k in range(1, 13):
        C, *_ = predict_gravity(tw.eb, np.zeros((1, 6)), tw.stations, g_vec=tw.g_scene, mass=tw.mass, iters=k,
                                tol=0.0, relax=0.5)
        rows.append(320.0 - C[0, -1, 1])
    Cf, *_ = predict_gravity(tw.eb, np.zeros((1, 6)), tw.stations, g_vec=tw.g_scene, mass=tw.mass, tol=1e-12,
                             iters=400, relax=1.0)
    exact = 320.0 - Cf[0, -1, 1]
    dl = np.zeros(6); dl[0] = 3.0; dl[3] = 1.5
    tips = []
    for k in range(1, 41):
        C, *_ = predict_gravity(tw.eb, dl[None], tw.stations, g_vec=tw.g_scene, mass=tw.mass, iters=k, tol=0.0)
        tips.append(C[0, -1].copy())
    Cb, *_ = predict_gravity(tw.eb, dl[None], tw.stations, g_vec=tw.g_scene, mass=tw.mass, iters=600, tol=1e-12)
    err = [np.linalg.norm(t - Cb[0, -1]) for t in tips]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.2))
    axs[0].plot(range(1, 13), rows, "o-", color=BLUE, ms=3.5, lw=1)
    axs[0].axhline(exact, color=INK, lw=0.7, ls=":")
    axs[0].text(6.5, exact + 0.004, "exact fixed point", va="bottom", fontsize=7)
    axs[0].annotate("old rule stopped here\n(no rotation change)", (1, rows[0]), (4.0, 0.15), fontsize=7,
                    color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7))
    axs[0].set_xlabel("iteration"); axs[0].set_ylabel("self-weight compression (mm)")
    axs[0].set_title("2 modules upright, straight", fontsize=7.5)
    axs[1].semilogy(range(1, 41), err, "o-", color=BLUE, ms=3, lw=1)
    axs[1].set_xlabel("iteration"); axs[1].set_ylabel("tip distance to fixed point (mm)")
    axs[1].set_title("2 modules upright, bent ($\\delta\\ell_1=3$, $\\delta\\ell_4=1.5$)", fontsize=7.5)
    for ax in axs:
        grid(ax)
    save(fig, "fixedpoint")
    return rows[0], exact


# ------------------------------------------------------------------ 8. capstan factors along the tendons
def fig_capstan():
    cal = dict(log_axial=0.0, log_bend_x=0.0, log_bend_z=0.0, log_mass=0.0, tendon_c=0.01, free_len_mm=300.0,
               mass_kg=MASS_STD6)
    tw = Twin(2, (0.0, 0.0, 0.0), calib_json=cal)
    dl = np.array([[5.0, -2.5, -2.5, 3.0, -1.5, -1.5]])
    lam = tw.eb.tensions(dl)
    w, _ = tw.eb.strains(lam)
    y = tw.eb.yc
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for sig, ls, lab in ((1.0, "-", "winding in ($\\sigma=+1$)"), (-1.0, "--", "paying out ($\\sigma=-1$)")):
        f = capstan_factors(tw.eb, w, np.full((1, 6), sig), 0.3)[0]
        for i, col in ((3, ORANGE), (0, BLUE)):                         # section 1 on top: same path in module 1
            m = tw.eb.through[i]
            ax.plot(y[m], f[i, m], ls, color=col, lw=2.4 if i == 3 else 1.1,
                    label=f"section-{1 + i // 3} tendon, " + lab)
    ax.axhline(1, color=MUTED, lw=0.6)
    ax.set_xlabel("$y$ along the robot (mm)"); ax.set_ylabel("tension / tension at the base")
    ax.legend(fontsize=6.2, loc="center right")
    grid(ax)
    save(fig, "capstan")


# ------------------------------------------------------------------ 9. pretension: equal pulls vs equal tension
def fig_pretension():
    from hardware.pose_lists import pretension_pattern, screen_model
    eb = screen_model(4)
    flat = eb.tensions(np.full((1, 12), 2.0))[0] * 56.0
    pat = eb.tensions((2.0 * pretension_pattern(4))[None])[0] * 56.0
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.arange(12)
    ax.bar(x - 0.2, flat, 0.38, color=GREY, label="every tendon pulled 2 mm")
    ax.bar(x + 0.2, pat, 0.38, color=BLUE, label="pattern $2\\times(1, 1.75, 2.25, 2.5)$ mm")
    ax.set_xticks(x); ax.set_xticklabels([f"s{s}t{k}" for s in range(1, 5) for k in range(1, 4)], rotation=60,
                                         fontsize=6.5)
    ax.set_ylabel("tension (N)")
    ax.legend(fontsize=6.6, loc="upper right")
    grid(ax)
    save(fig, "pretension")
    return flat, pat


# ------------------------------------------------------------------ 10. GP: interpolation vs extrapolation
def fig_gp():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
    rng = np.random.default_rng(3)
    f = lambda x: 0.6 * x + 0.35 * np.sin(2.2 * x) + 0.08 * x ** 2      # noqa: E731  ("true" response)
    xt = np.sort(rng.uniform(0, 3, 12))
    yt = f(xt) + rng.normal(0, 0.03, len(xt))
    gp = GaussianProcessRegressor(Matern(1.0, nu=2.5) + WhiteKernel(1e-3), normalize_y=True).fit(xt[:, None], yt)
    xs = np.linspace(-0.2, 6, 300)
    m, s = gp.predict(xs[:, None], return_std=True)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.axvspan(3, 6, color="#f3f2ee", lw=0)
    ax.fill_between(xs, m - 2 * s, m + 2 * s, color=BLUE, alpha=0.15, lw=0, label="GP mean $\\pm2\\sigma$")
    ax.plot(xs, f(xs), color=INK, lw=1, ls=":", label="true response")
    ax.plot(xs, m, color=BLUE, lw=1.2, label="GP mean")
    ax.plot(xt, yt, "o", color=ORANGE, ms=3.5, label="training data")
    ax.text(4.5, -0.2, "outside the data:\nthe GP returns to its prior", ha="center", fontsize=7, color=MUTED)
    ax.set_xlabel("load (arbitrary units)"); ax.set_ylabel("response")
    ax.legend(fontsize=6.6, loc="upper left")
    grid(ax)
    save(fig, "gp")


# ------------------------------------------------------------------ 11. coverage vs learned error (n_transfer)
def fig_coverage():
    nt = json.load(open("results/n_transfer.json"))["tests"]
    names = {"std6_4s_sym_test_g0": "4 modules", "std6_2s_sym_test_ghang": "2 hanging",
             "std6_2s_sym_test_gx": "2 horiz. $x$", "std6_2s_sym_test_gy": "2 horiz. $y$",
             "std6_1s_sym_test_g0": "1 module", "std6_2s_sym_test_g0": "2 modules", "std6_2s_sym_test": "2 upright"}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for k, lab in names.items():
        t = nt[k]
        x = t["coverage"]["nn_ratio"]
        phys = t["models"]["L0+L1"]["tip"]
        for key, col, mk in (("+L3 local (inside)", ORANGE, "^"), ("+L3 local (output)", YELLOW, "s")):
            ax.plot(x, t["models"][key]["tip"] / phys, mk, color=col, ms=5, mec="white", mew=0.5)
        ax.text(x * 1.08, t["models"]["+L3 local (inside)"]["tip"] / phys, lab, fontsize=6.3, va="center",
                color=MUTED)
    ax.axhline(1, color=BLUE, lw=1)
    ax.text(0.62, 0.86, "physics alone", color=BLUE, fontsize=7)
    ax.plot([], [], "^", color=ORANGE, label="learned residual inside the solve")
    ax.plot([], [], "s", color=YELLOW, label="learned residual on the output")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.5, 20)
    ax.set_xlabel("distance of test module states from training\n(relative to the training spacing)")
    ax.set_ylabel("tip error / physics-alone tip error")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    grid(ax, "both")
    save(fig, "coverage")


# ------------------------------------------------------------------ 12. error grows with bend (identified physics vs FEM)
def fig_bend():
    m, th = ModuleModel.from_result(json.load(open("results/mc_std6.json")))
    sets = [("std6_2s_sym_test_g0", "2 modules, no gravity"), ("std6_2s_sym_test", "2 upright"),
            ("std6_2s_sym_test_gx", "2 horizontal"), ("std6_4s_sym_test_g0", "4 modules, no gravity")]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for k, (tag, lab) in enumerate(sets):
        d = load_set(tag)
        C, R, lam = m.predict(th, d)
        err = np.linalg.norm(C[:, -1] - d["C"][:, -1], axis=1)
        bend = np.degrees(np.arccos(np.clip(d["R"][:, -1, 1, 1], -1, 1)))
        q = np.quantile(bend, [0, 1 / 3, 2 / 3, 1])
        vals = [err[(bend >= a) & (bend <= b)].mean() for a, b in zip(q[:-1], q[1:])]
        ax.plot([1, 2, 3], vals, "o-", color=(BLUE, ORANGE, AQUA, VIOLET)[k], ms=4, lw=1.1, label=lab)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["smallest third", "middle third", "largest third"])
    ax.set_xlabel("tip bend of the FEM pose"); ax.set_ylabel("mean tip error (mm)")
    ax.legend(fontsize=6.6)
    grid(ax)
    save(fig, "bend")


# ------------------------------------------------------------------ 13. convex notch laws
def fig_convex():
    M = np.linspace(-3, 3, 400)
    c, M0 = 1.0, 1.2
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.1))
    axs[0].plot(M, 0.5 * c * M ** 2, color=GREY, lw=1.2, label="quadratic (linear law, today)")
    axs[0].plot(M, c * M0 ** 2 * (np.sqrt(1 + (M / M0) ** 2) - 1), color=BLUE, lw=1.4,
                label="saturating (closure moment $M_0$)")
    axs[0].set_xlabel("load $M$"); axs[0].set_ylabel("complementary energy $\\psi(M)$")
    axs[1].plot(M, c * M, color=GREY, lw=1.2)
    axs[1].plot(M, c * M / np.sqrt(1 + (M / M0) ** 2), color=BLUE, lw=1.4)
    axs[1].axhline(c * M0, color=MUTED, lw=0.6, ls=":"); axs[1].axhline(-c * M0, color=MUTED, lw=0.6, ls=":")
    axs[1].text(-2.9, c * M0 + 0.12, "$\\pm cM_0$: the notch closes", fontsize=6.8, color=MUTED)
    axs[1].set_xlabel("load $M$"); axs[1].set_ylabel("bend $\\kappa=\\psi'(M)$")
    axs[0].legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for ax in axs:
        grid(ax)
    save(fig, "convex")


# ------------------------------------------------------------------ 14. identifiability: Jacobian SVD of the calibration
def fig_svd():
    names = ["log_axial", "log_bend_x", "log_bend_z", "tendon_c", "log_mass"]
    scale = dict(log_axial=1.0, log_bend_x=1.0, log_bend_z=1.0, tendon_c=0.01, log_mass=1.0)
    base = dict(log_axial=0.0, log_bend_x=0.0, log_bend_z=0.0, log_mass=0.0, tendon_c=0.002, free_len_mm=300.0,
                mass_kg=MASS_STD6)
    cmds = load_set("std6_2s_sym_test_g0")["dl"][:8]

    def ends(cal):
        out = []
        for mount in ("upright", "horizontal_x"):
            tw = Twin(2, mount, calib_json=cal, tol=1e-10)
            for x in cmds:
                s = tw.shape(x)
                for k in (32, 64):
                    out.append(s.positions[k])
        return np.concatenate(out)

    y0 = ends(base)
    J = []
    for p in names:
        h = 1e-3 * scale[p]
        c = dict(base); c[p] = base[p] + h
        J.append((ends(c) - y0) / h * scale[p])
    J = np.array(J).T
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.1))
    axs[0].bar(range(5), S, color=BLUE)
    axs[0].set_yscale("log")
    axs[0].set_xticks(range(5)); axs[0].set_xticklabels([f"$\\sigma_{k + 1}$" for k in range(5)])
    axs[0].set_ylabel("singular value (mm per unit)")
    v = Vt[-1] * np.sign(Vt[-1][np.argmax(np.abs(Vt[-1]))])
    axs[1].barh(range(5), v, color=[ORANGE if abs(t) > 0.3 else GREY for t in v])
    axs[1].set_yticks(range(5)); axs[1].set_yticklabels(names, fontsize=7)
    axs[1].axvline(0, color=INK, lw=0.6)
    axs[1].set_xlabel("weakest direction $v_5$ (components)")
    for ax in axs:
        grid(ax, "x" if ax is axs[1] else "y")
    save(fig, "svd")
    return S, v


# ------------------------------------------------------------------ 15. run time
def fig_time():
    rows = []
    for n, tag in ((1, "std6_1s_sym_test_g0"), (2, "std6_2s_sym_test_g0"), (4, "std6_4s_sym_test_g0")):
        dl = load_set(tag)["dl"][:21]
        seq = np.concatenate([np.linspace(dl[i], dl[i + 1], 10) for i in range(20)])
        for mount in ((0.0, 0.0, 0.0), "hanging"):
            tw = Twin(n, mount); tw.shape(seq[0])
            t = [tw.shape(x).solve_ms for x in seq]
            rows.append((n, "no gravity" if mount != "hanging" else "hanging", np.median(t), np.percentile(t, 95)))
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for k, mount in enumerate(("no gravity", "hanging")):
        r = [x for x in rows if x[1] == mount]
        ax.bar(np.arange(3) + (k - 0.5) * 0.36, [x[2] for x in r], 0.34, color=(GREY, BLUE)[k], label=mount)
        for i, x in enumerate(r):
            ax.text(i + (k - 0.5) * 0.36, x[2], f"{x[2]:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(["1 module", "2 modules", "4 modules"])
    ax.set_ylabel("median ms per pose")
    ax.legend(fontsize=7)
    grid(ax)
    save(fig, "time")
    return rows


if __name__ == "__main__":
    out = {}
    fig_sections(); fig_disc()
    out["G"] = fig_G().round(2).tolist()
    fig_qp()
    out["prop1"] = [float(v) if np.isscalar(v) else np.round(v, 5).tolist() for v in fig_prop1()]
    fig_slack()
    out["fixedpoint"] = fig_fixedpoint()
    fig_capstan()
    out["pretension"] = [np.round(v, 3).tolist() for v in fig_pretension()]
    fig_gp(); fig_coverage(); fig_bend(); fig_convex()
    S, v = fig_svd(); out["svd"] = dict(S=np.round(S, 4).tolist(), v=np.round(v, 3).tolist())
    out["time"] = [(a, b, round(float(c), 3), round(float(d), 3)) for a, b, c, d in fig_time()]
    json.dump(out, open(os.path.join(OUT, "numbers.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:3000])
