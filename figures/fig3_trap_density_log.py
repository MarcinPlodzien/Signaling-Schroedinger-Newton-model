"""
Main text, Fig. 3: density of the released ground state (R = 2) on a logarithmic
scale and a wide spatial range, with the future light cone of the trap.

Data: data/trap_dynamics.npz      Output: rho_wideSN_gs_2.0_{-0.1,-1.0,-3.0}.png
"""
import numpy as np
from plot_style import load, save, density_map, kappa_tag

d = load("trap_dynamics")
for kappa in (0.1, 1.0, 3.0):
    i = np.flatnonzero((d["radii"] == 2.0) & (d["kappas"] == kappa))[0]
    fig = density_map(d["x"], d["times"], d["rho"][i], xlim=(-100, 100), tmax=30, log=True, cone=2.0)
    save(fig, f"rho_wideSN_gs_2.0_{kappa_tag(kappa)}.png")
