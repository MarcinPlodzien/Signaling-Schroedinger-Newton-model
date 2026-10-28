"""
Full time evolutions rho(t, x) used in the density plots.

  trap_dynamics.npz   ground state in the box released at t = 0:
                      R = 2, kappa in {0.1, 1, 3}   (main text, Figs. 2-3; SM ground-state plot uses kappa = 0, 1)
                      R = 1, kappa in {0, 1}        (SM, density at t = 30)
  gauss_dynamics.npz  Gaussian initial state of width SIGMA,
                      kappa in {0, 0.1, 0.5, 1, 3}  (SM density plots and M(t, R) maps)

The runs use a box DYNAMICS_BOX_FACTOR times larger than L (same dx) to keep fast
components from re-entering through the periodic boundary; the stored densities are
cropped to |x| <= L/2. Densities are stored at N_STORE + 1 equally spaced times in
single precision.

Usage:  python generate/gen_dynamics.py
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
import sn1d

TRAP_RUNS = [(2.0, 0.0), (2.0, 0.1), (2.0, 1.0), (2.0, 3.0), (1.0, 0.0), (1.0, 1.0)]   # (R, kappa)
GAUSS_KAPPAS = [0.0, 0.1, 0.5, 1.0, 3.0]

L_BIG = cfg.DYNAMICS_BOX_FACTOR * cfg.L
N_BIG = cfg.DYNAMICS_BOX_FACTOR * cfg.N
CROP = np.abs(sn1d.Grid(L_BIG, N_BIG, cfg.EPS).x) <= cfg.L / 2


def trap_run(args):
    R, kappa = args
    grid, _, times, rhos, _ = sn1d.run_trap(L_BIG, N_BIG, cfg.EPS, kappa, R, cfg.V0, cfg.T_MAX,
                                           cfg.C_DT, n_store=cfg.N_STORE)
    return times, rhos[:, CROP].astype(np.float32)


def gauss_run(kappa):
    grid = sn1d.Grid(L_BIG, N_BIG, cfg.EPS)
    psi = sn1d.gaussian(grid, cfg.SIGMA)
    times, rhos = sn1d.evolve(grid, psi, kappa, cfg.T_MAX, cfg.C_DT * grid.dx**2, cfg.N_STORE)
    return times, rhos[:, CROP].astype(np.float32)


def main():
    t0 = time.time()
    x = sn1d.Grid(L_BIG, N_BIG, cfg.EPS).x[CROP]
    dx = L_BIG / N_BIG
    cfg.DATA.mkdir(exist_ok=True)
    with Pool() as pool:
        trap = pool.map(trap_run, TRAP_RUNS)
        gauss = pool.map(gauss_run, GAUSS_KAPPAS)

    np.savez_compressed(cfg.DATA / "trap_dynamics.npz", x=x, dx=dx, times=trap[0][0],
                        radii=np.array([r for r, _ in TRAP_RUNS]),
                        kappas=np.array([k for _, k in TRAP_RUNS]),
                        rho=np.stack([r for _, r in trap]),
                        L_box=L_BIG, N_box=N_BIG, eps=cfg.EPS, V0=cfg.V0)
    np.savez_compressed(cfg.DATA / "gauss_dynamics.npz", x=x, dx=dx, times=gauss[0][0],
                        kappas=np.array(GAUSS_KAPPAS), rho=np.stack([r for _, r in gauss]),
                        sigma=cfg.SIGMA, L_box=L_BIG, N_box=N_BIG, eps=cfg.EPS)
    print(f"dynamics: {len(TRAP_RUNS) + len(GAUSS_KAPPAS)} runs in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
