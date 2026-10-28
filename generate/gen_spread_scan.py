"""
Spread Delta_x(t) of an initial Gaussian wavepacket as a function of the
self-coupling kappa (SM figure on self-localisation; critical coupling kappa_c).

Usage:  python generate/gen_spread_scan.py
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
import sn1d

KAPPAS = np.round(np.arange(0, 0.901, 0.01), 3)


def one_kappa(kappa):
    grid = sn1d.Grid(cfg.L, cfg.N, cfg.EPS)
    psi = sn1d.gaussian(grid, cfg.SIGMA)
    times, rhos = sn1d.evolve(grid, psi, kappa, cfg.T_MAX, cfg.C_DT * grid.dx**2, cfg.N_STORE)
    return times, sn1d.spread(grid, rhos)


def main():
    t0 = time.time()
    with Pool() as pool:
        res = pool.map(one_kappa, KAPPAS)
    cfg.DATA.mkdir(exist_ok=True)
    np.savez_compressed(cfg.DATA / "spread_scan.npz", kappas=KAPPAS, times=res[0][0],
                        spread=np.array([s for _, s in res]),
                        sigma=cfg.SIGMA, L=cfg.L, N=cfg.N, eps=cfg.EPS)
    print(f"spread_scan: {len(KAPPAS)} runs in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
