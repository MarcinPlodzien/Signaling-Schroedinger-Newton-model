"""
Main text, Fig. 5: maximal signalling measure M~(kappa, R) = max_t M(t, R) for the
trap-release protocol, as a function of the self-coupling and the trap size.

Data: data/trap_scan_eps0.15_N2048.npz      Output: fig_M_max_g_R_color_long_range_gs.png
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from plot_style import load, save

d = load("trap_scan_eps0.15_N2048")
Z = np.clip(d["Mmax"].T, 1e-6, None)          # values below 1e-6 shown at the bottom of the scale

fig, ax = plt.subplots(figsize=(6.6, 6))
cs = ax.contourf(d["kappas"], d["radii"], Z, levels=np.logspace(-6, np.log10(Z.max()), 25),
                 norm=LogNorm(), cmap="plasma")
cb = fig.colorbar(cs, ticks=10.0 ** np.arange(-6, 0))
cb.set_label(r"$\widetilde{M}(\kappa,R)$")
ax.set_xlabel(r"$\kappa$")
ax.set_ylabel("R")
save(fig, "fig_M_max_g_R_color_long_range_gs.png")
