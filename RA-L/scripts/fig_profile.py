"""Figures for docs/eb_prior/eb_prior_derivation.tex.  Run from the repo root."""
import json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tdcr.eb_prior import section_profile, CAD_JSON

OUT = os.path.join("docs", "eb_prior")
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False})
g = json.load(open(CAD_JSON))
R, ri, rd, e = g["outer_dia"] / 2, g["inner_dia"] / 2, g["disc_dia"] / 2, g["offset"]
ca, cb = np.array(g["notch_centres_A"]), np.array(g["notch_centres_B"])

# 1. cross-sections: plain tube, family-A notch centre, between notches
xs = np.arange(-R, R, 0.05); X, Z = np.meshgrid(xs, xs, indexing="ij"); rho = np.hypot(X, Z)
ring = (rho >= ri) & (rho <= R)
hz = np.where(np.abs(Z) > e, rd - np.sqrt(np.maximum(rd**2 - Z**2, 0)), -1)
hx = np.where(np.abs(X) > e, rd - np.sqrt(np.maximum(rd**2 - X**2, 0)), -1)
def mat(y):
    dA, dB = np.abs(y - ca).min(), np.abs(y - cb).min()
    return ring & ~((dA < hz) | (dB < hx))
fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
for ax, (y, t) in zip(axs, ((5.0, "plain tube ($y$ = 5 mm)"), (ca[0], "notch A centre ($y$ = %.1f)" % ca[0]),
                            (ca[0] + 2.0, "2 mm above notch A"), (cb[0], "notch B centre ($y$ = %.1f)" % cb[0]))):
    ax.imshow(mat(y).T, origin="lower", extent=[-R, R, -R, R], cmap="Greys", vmin=0, vmax=1.6)
    for k, ps in enumerate((0, 120, 240)):
        ax.plot(10.55 * np.cos(np.radians(ps)), 10.55 * np.sin(np.radians(ps)), "o", color="#D55E00", ms=5)
    ax.set_title(t, fontsize=12); ax.set_xlabel("$x$ (mm)"); ax.set_aspect("equal")
axs[0].set_ylabel("$z$ (mm)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "sections.pdf")); plt.close(fig)

# 2. stiffness profile along the module
y, A, Jx, Jz = section_profile()
fig, axs = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
axs[0].semilogy(y, Jz, color="#0072B2", lw=1.2, label="$J_z(y)=\\int z^2\\,dA$ (bending towards $\\pm z$)")
axs[0].semilogy(y, Jx, color="#E69F00", lw=1.2, label="$J_x(y)=\\int x^2\\,dA$ (bending towards $\\pm x$)")
axs[0].set_ylabel("mm$^4$"); axs[0].legend(frameon=False, fontsize=11, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
axs[1].plot(y, A, color="#333333", lw=1.2); axs[1].set_ylabel("$A(y)$ (mm$^2$)"); axs[1].set_xlabel("$y$ along the module (mm)")
for a in axs:
    for c in ca: a.axvline(c, color="#0072B2", lw=0.4, alpha=0.4)
    for c in cb: a.axvline(c, color="#E69F00", lw=0.4, alpha=0.4)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "profile.pdf")); plt.close(fig)
print("wrote sections.pdf, profile.pdf")
