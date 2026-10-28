"""
SM, robustness tests of M~(kappa, R):
  grid refinement at fixed eps = 0.15 (N = 1024, 2048, 4096)  -> fig_convergence_dx.png
  dependence on the softening eps of the kernel             -> fig_convergence_eps.png
Values below 1e-6 are treated as zero and shown at 1e-6.

Data: data/trap_scan_*.npz
"""
import matplotlib.pyplot as plt
import numpy as np
from plot_style import load, save, BLUES, LINESTYLES, MARKERS

FLOOR = 1e-6


def curve(d, R):
    return d["kappas"], np.maximum(d["Mmax"][:, int(np.argmin(np.abs(d["radii"] - R)))], FLOOR)


def panel_figure(runs, radii, title, filename, legend_panel=-1, legend_loc="best"):
    fig, axs = plt.subplots(1, len(radii), figsize=(4.2 * len(radii), 4), sharey=True)
    for ax, R in zip(axs, radii):
        for i, (name, label) in enumerate(runs):
            k, M = curve(load(name), R)
            ax.semilogy(k, M, LINESTYLES[i], color=BLUES[i], marker=MARKERS[i], ms=5, label=label)
        ax.set_title(f"R = {R:g}")
        ax.set_xlabel(r"$\kappa$")
        ax.grid(color="#e6e6e6", lw=0.6)
    axs[0].set_ylabel(r"$\widetilde{M}(\kappa,R)$")
    axs[legend_panel].legend(frameon=False, fontsize=12, loc=legend_loc)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    save(fig, filename)


panel_figure([("trap_scan_eps0.15_N1024", r"$N=1024$"),
              ("trap_scan_eps0.15_N2048", r"$N=2048$"),
              ("trap_scan_eps0.15_N4096_subset", r"$N=4096$")],
             [1, 1.5, 2, 3], r"Grid refinement at fixed $\varepsilon=0.15$ ($L=212$)", "fig_convergence_dx.png")

panel_figure([("trap_scan_eps0.1_N2048_R123", r"$\varepsilon=0.1$"),
              ("trap_scan_eps0.15_N2048", r"$\varepsilon=0.15$"),
              ("trap_scan_eps0.2_N2048_R123", r"$\varepsilon=0.2$"),
              ("trap_scan_eps0.4_N2048_R123", r"$\varepsilon=0.4$"),
              ("trap_scan_eps0.8_N2048_R123", r"$\varepsilon=0.8$")],
             [1, 2, 3], r"Dependence on the softening $\varepsilon$ ($L=212$, $N=2048$)",
             "fig_convergence_eps.png", legend_panel=0, legend_loc="lower left")
