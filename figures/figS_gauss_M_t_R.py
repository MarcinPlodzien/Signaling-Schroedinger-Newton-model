"""
SM: probability M(t, R) leaking out of the future light cone of [-R, R] for the
Gaussian initial state, kappa = 0, 0.5, 1, 3.

Data: data/gauss_dynamics.npz     Output: fig_M_t_R_long_range_gauss_g_{0,-0.5,-1.0,-3.0}.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import load, save

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402
import sn1d  # noqa: E402

RADII = np.arange(1, 20.01, 0.25)


def leakage_map(x, dx, times, rho):
    """M(t, R) = max{0, P_0(|x| <= R) - P_t(|x| <= R + c t)} on the (R, t) grid."""
    def prob(r, a):
        return np.sum(r * sn1d.cell_overlap(x, dx, a))
    return np.array([[max(0.0, prob(rho[0], R) - prob(r, R + cfg.C_LIGHT * t))
                      for t, r in zip(times, rho)] for R in RADII])


d = load("gauss_dynamics")
for kappa, tag in [(0.0, "0"), (0.5, "-0.5"), (1.0, "-1.0"), (3.0, "-3.0")]:
    rho = d["rho"][int(np.flatnonzero(d["kappas"] == kappa)[0])].astype(float)
    M = leakage_map(d["x"], float(d["dx"]), d["times"], rho)
    p = int(np.floor(np.log10(M.max())))                 # colour bar in units of 10^p
    fig, ax = plt.subplots(figsize=(6.6, 6))
    cs = ax.contourf(d["times"], RADII, M / 10.0**p, levels=12, cmap="plasma")
    cb = fig.colorbar(cs)
    cb.set_label(rf"$M(t,R)\ [10^{{{p}}}]$")
    ax.set_xlabel("t")
    ax.set_ylabel("R")
    save(fig, f"fig_M_t_R_long_range_gauss_g_{tag}.png")
