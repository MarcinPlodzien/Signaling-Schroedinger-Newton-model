"""
SM: spread Delta_x(t, kappa) of the Gaussian wavepacket; the boundary between
spreading and self-localisation gives the critical coupling kappa_c.

Data: data/spread_scan.npz        Output: fig_dx_t_g_long_range_gauss.png
"""
import matplotlib.pyplot as plt
import numpy as np
from plot_style import load, save

d = load("spread_scan")
fig, ax = plt.subplots(figsize=(6.6, 6))
cs = ax.contourf(d["times"], d["kappas"], d["spread"], levels=np.arange(0, 17, 2), cmap="plasma", extend="max")
cb = fig.colorbar(cs)
cb.set_label(r"$\Delta x(t,\kappa)$")
ax.set_xlabel("t")
ax.set_ylabel(r"$\kappa$")
save(fig, "fig_dx_t_g_long_range_gauss.png")
