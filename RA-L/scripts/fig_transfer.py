"""Figures of the RA-L draft from the result files (run from the repository root):
    .venv/bin/python paper_ral/figures/make_figs.py
fig_transfer.pdf: mean tip error per test configuration (FEM, real routing) for plane-section Euler-Bernoulli, the
identified physics (results/mc_std6.json) and the physics with a learned module residual (results/n_transfer.json)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
mc = json.load(open("results/mc_std6.json"))
nt = json.load(open("results/n_transfer.json"))["tests"]

CASES = [("std6_1s_sym_test_g0", "1 module", True), ("std6_2s_sym_test_g0", "2 modules", True),
         ("std6_1s_sym_test", "1 upright", True), ("std6_2s_sym_test", "2 upright", True),
         ("std6_4s_sym_test_g0", "4 modules", False), ("std6_2s_sym_test_ghang", "2 hanging", False),
         ("std6_2s_sym_test_gx", "2 horiz. $x$", False), ("std6_2s_sym_test_gy", "2 horiz. $y$", False)]
SERIES = [("plane-section EB (CAD only)", "#8a8984", "s", lambda k: mc["plane_section"][k]["tip"]),
          ("physics + learned module residual", "#eb6834", "^", lambda k: nt[k]["models"]["+L3 local (output)"]["tip"]),
          ("identified physics (ours)", "#2a78d6", "o", lambda k: mc["learned"][k]["tip"])]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.45, 2.55))
x = range(len(CASES))
seen = sum(c[2] for c in CASES)
ax.axvspan(seen - 0.5, len(CASES) - 0.5, color="#ecebe8", lw=0, zorder=0)
ax.text((seen + len(CASES) - 1) / 2, 140, "not seen in training", ha="center", va="top", fontsize=7, color="#52514e")
for label, col, mk, f in SERIES:
    ax.plot(x, [f(k) for k, _, _ in CASES], marker=mk, ms=5, ls="none", color=col, label=label, zorder=3,
            mec="white", mew=0.6)
ax.set_yscale("log")
ax.set_ylim(0.1, 150)
ax.set_xticks(list(x))
ax.set_xticklabels([c[1] for c in CASES], rotation=35, ha="right")
ax.set_ylabel("mean tip error (mm)")
ax.grid(axis="y", color="#e2e1dc", lw=0.5, zorder=0)
ax.tick_params(length=2)
fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.8))
fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(0.12, 1.0),
           handlelength=1.6)
fig.savefig(os.path.join(OUT, "fig_transfer.pdf"))
fig.savefig(os.path.join(OUT, "fig_transfer.png"), dpi=200)
print("->", os.path.join(OUT, "fig_transfer.pdf"))
