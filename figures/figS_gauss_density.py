"""
SM: time evolution of the density for the Gaussian initial state (width SIGMA),
kappa = 0.1, 1, 3.
  linear scale, |x| <= 3, t <= 15        -> rho_SN_gauss_5.0_{-0.1,-1.0,-3.0}.png
  log scale, |x| <= 100, light cone of [-5, 5] -> rho_wideSN_gauss_5.0_{-0.1,-1.0,-3.0}.png

Data: data/gauss_dynamics.npz
"""
import numpy as np
from plot_style import load, save, density_map, kappa_tag

R_CONE = 5.0  # region K = [-5, 5] whose light cone is drawn

d = load("gauss_dynamics")
for kappa in (0.1, 1.0, 3.0):
    i = int(np.flatnonzero(d["kappas"] == kappa)[0])
    fig = density_map(d["x"], d["times"], d["rho"][i], xlim=(-3, 3), tmax=15)
    save(fig, f"rho_SN_gauss_5.0_{kappa_tag(kappa)}.png")
    fig = density_map(d["x"], d["times"], d["rho"][i], xlim=(-100, 100), tmax=30, log=True, cone=R_CONE)
    save(fig, f"rho_wideSN_gauss_5.0_{kappa_tag(kappa)}.png")
