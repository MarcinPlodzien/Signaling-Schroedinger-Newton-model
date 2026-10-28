"""
Dependence of the signalling quantities on the dimensionless speed of light
c~ = m c l0 / hbar.

The density does not depend on c~; only the light cone |x| <= R + c~ t does.
For every (kappa, R) one evolution is computed and, at every time step, M(t, R)
and Bob's distinguishability D_t are evaluated for all values of c~. Stored are
their maxima over t in [0, T_MAX].

For larger c~ the light cone crosses a grid cell within very few time steps and the
results depend noticeably on dx; such values are not included.

Usage:  python generate/gen_c_scan.py
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
import sn1d

C_VALUES = np.array([1.0, 2.0, 5.0, 10.0])
RADII = np.array([1.0, 2.0, 3.0])
KAPPAS = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 4.0])


def one_point(args):
    kappa, R = args
    *_, mon = sn1d.run_trap(cfg.L, cfg.N, cfg.EPS, kappa, R, cfg.V0, cfg.T_MAX, cfg.C_DT,
                            n_store=10, cs=C_VALUES)
    return np.concatenate([mon.M_max, mon.D_max])


def main():
    t0 = time.time()
    jobs = [(k, R) for k in KAPPAS for R in RADII]
    with Pool() as pool:
        res = np.array(pool.map(one_point, jobs, chunksize=1)).reshape(len(KAPPAS), len(RADII), 2, len(C_VALUES))
    cfg.DATA.mkdir(exist_ok=True)
    np.savez_compressed(cfg.DATA / "c_scan.npz", kappas=KAPPAS, radii=RADII, c_values=C_VALUES,
                        Mmax=res[:, :, 0], Dmax=res[:, :, 1],
                        L=cfg.L, N=cfg.N, eps=cfg.EPS, V0=cfg.V0, t_max=cfg.T_MAX)
    print(f"c_scan: {len(jobs)} points in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
