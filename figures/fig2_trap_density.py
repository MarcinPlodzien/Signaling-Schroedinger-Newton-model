"""
Main text, Fig. 2: time evolution of the density of the released ground state,
R = 2, kappa = 0.1, 1, 3 (linear colour scale, t <= 15).

Data: data/trap_dynamics.npz      Output: rho_SN_gs_2.0_{-0.1,-1,-3}.png
"""
import numpy as np
from plot_style import load, save, density_map

d = load("trap_dynamics")
for kappa, tag in [(0.1, "-0.1"), (1.0, "-1"), (3.0, "-3")]:
    i = np.flatnonzero((d["radii"] == 2.0) & (d["kappas"] == kappa))[0]
    fig = density_map(d["x"], d["times"], d["rho"][i], xlim=(-2, 2), tmax=15)
    save(fig, f"rho_SN_gs_2.0_{tag}.png")
