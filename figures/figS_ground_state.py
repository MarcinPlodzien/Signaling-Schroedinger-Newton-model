"""
SM: structure of the ground state in the box potential.
  (a) ground-state densities for R = 2, kappa = 0 and 1       -> rho_INI_SN_gs_2.0_-1.0.png
  (b) ratio delta = E_int / E_kin over the (kappa, R) plane   -> fig_E_g_R_long_range_gs.png

Data: data/trap_dynamics.npz, data/trap_scan_eps0.15_N2048.npz
"""
import matplotlib.pyplot as plt
import numpy as np
from plot_style import load, save

R = 2.0
d = load("trap_dynamics")
fig, ax = plt.subplots(figsize=(8, 5))
for kappa, style, color in [(0.0, "--", "purple"), (1.0, ":", "black")]:
    i = int(np.flatnonzero((d["radii"] == R) & (d["kappas"] == kappa))[0])
    ax.plot(d["x"], d["rho"][i, 0], style, color=color, label=rf"$\kappa={kappa:g}$")
for sign in (-1, 1):
    ax.axvline(sign * R, color="red", lw=1.5)
ax.set_xlim(-5, 5)
ax.set_ylim(0, None)
ax.set_xlabel("x")
ax.set_ylabel(r"$\rho(0,x)$")
ax.legend(frameon=False)
save(fig, "rho_INI_SN_gs_2.0_-1.0.png")

s = load("trap_scan_eps0.15_N2048")
delta = (s["E_int"] / s["E_kin"]).T
fig, ax = plt.subplots(figsize=(6.6, 6))
cs = ax.contourf(s["kappas"], s["radii"], delta, levels=24, cmap="plasma")
cb = fig.colorbar(cs)
cb.set_label(r"$\delta = E_{\rm int}/E_{\rm kin}$")
ax.set_xlabel(r"$\kappa$")
ax.set_ylabel("R")
save(fig, "fig_E_g_R_long_range_gs.png")
