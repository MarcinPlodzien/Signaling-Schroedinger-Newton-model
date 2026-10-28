"""Shared plotting settings and helpers for all figure scripts."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg  # noqa: E402

plt.rcParams.update({
    "font.size": 16,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "lines.linewidth": 2,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Ordinal blue ramp (light -> dark) for line plots indexed by an ordered parameter
BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b", "#081f3f"]
LINESTYLES = ["-", "--", "-.", ":", "-"]
MARKERS = ["o", "s", "^", "D", "v"]


def load(name):
    """Load data/<name>.npz."""
    path = cfg.DATA / f"{name}.npz"
    if not path.exists():
        sys.exit(f"missing {path}; run generate_all_data.sh first")
    return np.load(path)


def save(fig, filename):
    """Write the figure to figures/output/<filename> and close it."""
    cfg.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg.FIGURES / filename)
    plt.close(fig)
    print("wrote", cfg.FIGURES / filename)


def kappa_tag(kappa):
    """Coupling as it appears in the original figure file names (e.g. 1 -> '-1.0')."""
    return f"-{kappa:.1f}" if kappa else "0.0"


def density_map(x, t, rho, xlim, tmax, log=False, cone=None):
    """
    Colour map of rho(t, x) on the (x, t) plane.
    log  : logarithmic colour scale between 1e-6 and 1;
    cone : half-width R of the trap; draws the light cone |x| = R + t.
    """
    from matplotlib.colors import LogNorm
    dx = x[1] - x[0]
    sel_x = (x >= xlim[0] - dx) & (x <= xlim[1] + dx)   # one extra point: no blank margin
    sel_t = t <= tmax + 1e-9
    X, Z = x[sel_x], rho[np.ix_(sel_t, sel_x)]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    if log:
        Z = np.clip(Z, 1.01e-6, 0.99)                      # keep values inside the level range
        cs = ax.contourf(X, t[sel_t], Z, levels=np.logspace(-6, 0, 25), norm=LogNorm(), cmap="inferno")
        cb = fig.colorbar(cs, ticks=10.0 ** np.arange(-6, 1))
    else:
        cs = ax.contourf(X, t[sel_t], Z, levels=24, cmap="inferno")
        cb = fig.colorbar(cs)
    cb.set_label(r"$\rho(t,x)$")
    if cone is not None:
        tt = np.linspace(0, tmax, 50)
        for sign in (-1, 1):
            ax.plot(sign * (cone + cfg.C_LIGHT * tt), tt, "--", color="white", lw=1.5)
    ax.set_xlim(xlim)
    ax.set_ylim(0, tmax)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    return fig
