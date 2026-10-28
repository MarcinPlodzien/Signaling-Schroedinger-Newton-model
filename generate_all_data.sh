#!/usr/bin/env bash
# Regenerate all data in data/ (about one hour on 16 cores).
set -euo pipefail
cd "$(dirname "$0")"

# Time evolutions for the density plots and the Gaussian-wavepacket figures
python generate/gen_dynamics.py
python generate/gen_spread_scan.py

# Dependence on the dimensionless speed of light
python generate/gen_c_scan.py

# Signalling measure over (kappa, R) with the paper parameters (N = 2048)
python generate/gen_trap_scan.py

# Robustness: grid (N = 1024 and a N = 4096 subset including R = 1.5)
python generate/gen_trap_scan.py --N 1024
python generate/gen_trap_scan.py --N 4096 --subset

# Robustness: softening parameter (R = 1, 2, 3)
python generate/gen_trap_scan.py --eps 0.1 --small-radii
python generate/gen_trap_scan.py --eps 0.2 --small-radii
python generate/gen_trap_scan.py --eps 0.4 --small-radii
python generate/gen_trap_scan.py --eps 0.8 --small-radii

# Robustness: box size at fixed dx (periodic boundary)
python generate/gen_trap_scan.py --L 424 --N 4096 --subset

# Robustness: time step (dt and dt/4)
python generate/gen_dt_check.py
