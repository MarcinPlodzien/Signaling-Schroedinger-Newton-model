#!/usr/bin/env bash
# Regenerate every figure from the stored data in data/ (about 1 min).
# Output: figures/output/, with the file names used in the manuscript.
set -euo pipefail
cd "$(dirname "$0")/figures"
for script in fig2_trap_density.py fig3_trap_density_log.py fig4_distinguishability_vs_c.py \
              fig5_Mmax_map.py figS_gauss_density.py figS_gauss_spread.py figS_gauss_M_t_R.py \
              figS_ground_state.py figS_density_t30.py figS_convergence.py; do
    python "$script"
done
