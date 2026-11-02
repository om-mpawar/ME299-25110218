"""Fig. 1 of the paper: the module's cross-section properties along its height, from the CAD (tdcr/eb_prior.py):
    .venv/bin/python scripts/fig_profile.py            # -> paper/figures/fig_profile.pdf
(Before the packaging this was the second figure of docs/eb_prior/make_figs.py, profile.pdf, copied into the paper by
hand.) As then, the profile is section_profile() with no channel holes cut (holes=None) on the std CAD."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tdcr.eb_prior import CAD_JSON, section_profile
from tdcr.paths import FIGURES

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out-dir", default=FIGURES)
a = ap.parse_args()

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False})
g = json.load(open(CAD_JSON))
ca, cb = np.array(g["notch_centres_A"]), np.array(g["notch_centres_B"])

y, A, Jx, Jz = section_profile()
fig, axs = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
axs[0].semilogy(y, Jz, color="#0072B2", lw=1.2, label="$J_z(y)=\\int z^2\\,dA$ (bending towards $\\pm z$)")
axs[0].semilogy(y, Jx, color="#E69F00", lw=1.2, label="$J_x(y)=\\int x^2\\,dA$ (bending towards $\\pm x$)")
axs[0].set_ylabel("mm$^4$"); axs[0].legend(frameon=False, fontsize=11, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
axs[1].plot(y, A, color="#333333", lw=1.2); axs[1].set_ylabel("$A(y)$ (mm$^2$)"); axs[1].set_xlabel("$y$ along the module (mm)")
for ax in axs:
    for c in ca: ax.axvline(c, color="#0072B2", lw=0.4, alpha=0.4)
    for c in cb: ax.axvline(c, color="#E69F00", lw=0.4, alpha=0.4)
fig.tight_layout()
out = os.path.join(a.out_dir, "fig_profile.pdf")
fig.savefig(out)
print("->", out)
