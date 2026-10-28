"""Numerical parameters used for all results in the paper (dimensionless units)."""
from pathlib import Path

L = 212.0        # box length
N = 2048         # grid points, dx = L / N = 0.1035
EPS = 0.15       # softening of the kernel 1/sqrt(x^2 + eps^2)
V0 = 20.0        # depth of the trapping box potential
T_MAX = 30.0     # final time
C_DT = 0.1       # real-time step dt = C_DT * dx^2
N_STORE = 300    # number of stored time steps (every 0.1 time units)
SIGMA = 1.0      # width of the Gaussian initial state
C_LIGHT = 1.0    # dimensionless speed of light used in the light-cone integrals

# The density plots are computed in a 4x larger box with the same dx and cropped to
# |x| <= L/2. On the periodic grid, fast components leaving the box re-enter on the
# other side and produce fine interference ripples in the far tails (|x| > 40 at
# t = 30); they do not affect M (identical to 5 digits), but would show in the plots.
DYNAMICS_BOX_FACTOR = 4

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures" / "output"
