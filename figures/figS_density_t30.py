"""
SM: density of the released ground state (R = 1) at t = 30 compared with the
initial density, for kappa = 0 and 1, with the light cone of the trap.

Data: data/trap_dynamics.npz      Output: rho_x_SN_gs_1.0_{0.0,-1.0}.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import load, save, kappa_tag

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

R = 1.0
d = load("trap_dynamics")
t_end = float(d["times"][-1])
for kappa in (0.0, 1.0):
    i = int(np.flatnonzero((d["radii"] == R) & (d["kappas"] == kappa))[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(d["x"], d["rho"][i, 0], ":", color="purple", label=r"$t=0$")
    ax.semilogy(d["x"], d["rho"][i, -1], "-", color="magenta", label=rf"$t={t_end:g}$")
    for sign in (-1, 1):
        ax.axvline(sign * (R + cfg.C_LIGHT * t_end), color="red", ls="--", lw=1.5)
    ax.set_xlim(-50, 50)
    ax.set_ylim(1e-6, 10)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho(t,x)$")
    ax.legend(frameon=False, loc="upper right")
    save(fig, f"rho_x_SN_gs_1.0_{kappa_tag(kappa)}.png")
