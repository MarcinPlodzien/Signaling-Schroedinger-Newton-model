"""
Time-step check: M~(kappa, R) with the paper's time step dt = C_DT * dx^2 and with
dt / 4, for a set of points including the most strongly bound states.

Usage:  python generate/gen_dt_check.py
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
import sn1d

POINTS = [(0.0, 1.0), (1.0, 1.0), (4.0, 1.0), (3.0, 1.5), (4.0, 1.5), (1.0, 2.0), (2.0, 2.0), (0.0, 5.0)]
FACTORS = [1.0, 0.25]


def one_point(args):
    (kappa, R), f = args
    dx = cfg.L / cfg.N
    *_, mon = sn1d.run_trap(cfg.L, cfg.N, cfg.EPS, kappa, R, cfg.V0, cfg.T_MAX, cfg.C_DT,
                            n_store=10, dt=f * cfg.C_DT * dx**2)
    return mon.M_max[0]


def main():
    t0 = time.time()
    jobs = [(p, f) for f in FACTORS for p in POINTS]
    with Pool() as pool:
        res = np.array(pool.map(one_point, jobs, chunksize=1)).reshape(len(FACTORS), len(POINTS))
    cfg.DATA.mkdir(exist_ok=True)
    np.savez_compressed(cfg.DATA / "dt_check.npz", points=np.array(POINTS), factors=np.array(FACTORS),
                        Mmax=res, L=cfg.L, N=cfg.N, eps=cfg.EPS)
    print(f"dt_check: {len(jobs)} runs in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
