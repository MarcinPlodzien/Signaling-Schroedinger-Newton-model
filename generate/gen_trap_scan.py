"""
Scan of the trap-release protocol over the self-coupling kappa and the trap
half-width R.

For every (kappa, R) the ground state in the box potential is computed, the trap
is released, and the state is evolved up to T_MAX. Stored per point:
  Mmax  - maximum over all time steps of M(t, R) (light speed C_LIGHT),
  t_max - time at which the maximum is attained,
  E_kin, E_int - kinetic and interaction energy of the ground state,
  gs_residual - residual ||(H - mu) psi|| of the ground state.

Usage:
  python generate/gen_trap_scan.py                      # paper parameters
  python generate/gen_trap_scan.py --eps 0.4            # other softening
  python generate/gen_trap_scan.py --N 4096 --subset    # coarse (kappa, R) subset
  python generate/gen_trap_scan.py --eps 0.4 --small-radii   # R in {1, 2, 3} only
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg
import sn1d

KAPPAS_FULL = np.round(np.concatenate([np.arange(0, 1, 0.05), np.arange(1, 4.01, 0.25)]), 3)
RADII_FULL = np.arange(1, 5.01, 0.5)
KAPPAS_SUBSET = np.array([0, 1, 2, 3, 4.0])
RADII_SUBSET = np.array([1.0, 1.5, 2.0, 3.0])        # includes the half-integer radius 1.5


def one_point(args):
    L, N, eps, kappa, R = args
    grid, psi, _, _, mon = sn1d.run_trap(L, N, eps, kappa, R, cfg.V0, cfg.T_MAX, cfg.C_DT,
                                         n_store=10, cs=(cfg.C_LIGHT,))
    e_kin, _, e_int = grid.energies(psi, kappa, sn1d.box_potential(grid, R, cfg.V0))
    return mon.M_max[0], mon.t_M[0], e_kin, e_int, mon.gs_residual


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L", type=float, default=cfg.L)
    ap.add_argument("--N", type=int, default=cfg.N)
    ap.add_argument("--eps", type=float, default=cfg.EPS)
    ap.add_argument("--subset", action="store_true", help="use the coarse (kappa, R) subset")
    ap.add_argument("--small-radii", action="store_true", help="full kappa grid, R in {1, 2, 3} only")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    kappas = KAPPAS_SUBSET if a.subset else KAPPAS_FULL
    radii = RADII_SUBSET if a.subset else (np.array([1.0, 2.0, 3.0]) if a.small_radii else RADII_FULL)
    jobs = [(a.L, a.N, a.eps, k, R) for k in kappas for R in radii]

    t0 = time.time()
    with Pool(a.workers) as pool:
        res = np.array(pool.map(one_point, jobs, chunksize=1)).reshape(len(kappas), len(radii), 5)

    cfg.DATA.mkdir(exist_ok=True)
    name = (f"trap_scan_eps{a.eps:g}_N{a.N}" + (f"_L{a.L:g}" if a.L != cfg.L else "")
            + ("_subset" if a.subset else "") + ("_R123" if a.small_radii else ""))
    np.savez_compressed(cfg.DATA / f"{name}.npz", kappas=kappas, radii=radii,
                        Mmax=res[..., 0], t_at_max=res[..., 1], E_kin=res[..., 2], E_int=res[..., 3],
                        gs_residual=res[..., 4],
                        L=a.L, N=a.N, eps=a.eps, V0=cfg.V0, t_max=cfg.T_MAX, c=cfg.C_LIGHT)
    print(f"{name}: {len(jobs)} points in {time.time() - t0:.0f} s, max ground-state residual {res[..., 4].max():.1e}")


if __name__ == "__main__":
    main()
