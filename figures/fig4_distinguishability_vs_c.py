"""
Main text, Fig. 4: Bob's maximal distinguishability max_t D_t (total-variation
distance for the detection region outside the light cone) versus kappa, for the
trap R = 2 and several values of the dimensionless speed of light c~.

Data: data/c_scan.npz             Output: fig_M_vs_c.png
"""
import matplotlib.pyplot as plt
import numpy as np
from plot_style import load, save, BLUES, LINESTYLES, MARKERS

d = load("c_scan")
iR = int(np.argmin(np.abs(d["radii"] - 2.0)))

fig, ax = plt.subplots(figsize=(6, 4.5))
for j, c in enumerate(d["c_values"]):
    ax.semilogy(d["kappas"], d["Dmax"][:, iR, j], LINESTYLES[j], color=BLUES[j], marker=MARKERS[j],
                ms=5, label=rf"$\tilde c={c:g}$")
ax.axhspan(1e-9, 1e-6, color="#f0efec", zorder=0)      # values below 1e-6 are treated as zero
ax.set_ylim(1e-7, 1)
ax.set_xlabel(r"$\kappa$")
ax.set_ylabel(r"$\max_t D_t$   ($R=2$)")
ax.legend(frameon=False, fontsize=12, ncol=2)
ax.grid(color="#e6e6e6", lw=0.6)
save(fig, "fig_M_vs_c.png")
