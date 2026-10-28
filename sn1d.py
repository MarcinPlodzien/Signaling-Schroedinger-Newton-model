"""
Split-step Fourier solver for the one-dimensional Schroedinger--Newton (SN) equation
in dimensionless units,

    i d_t psi = [ -1/2 d_x^2 + V(t, x) - kappa * Phi_eps[|psi|^2](x) ] psi,
    Phi_eps[rho](x) = int dy U_eps(x - y) rho(y),     U_eps(x) = 1 / sqrt(x^2 + eps^2).

The softened kernel U_eps replaces the Newtonian 1/|x|, which is not locally
integrable in one dimension.

Discretisation
--------------
* periodic grid x_j = (j - N/2) dx, j = 0..N-1, box length L = N dx;
* kinetic term in Fourier space;
* self-potential: FFT convolution with the kernel periodised by the minimum-image
  convention, including the quadrature weight dx;
* ground state: imaginary-time split-step (normalised gradient flow) with a
  decreasing ladder of imaginary-time steps, refined by self-consistent iteration
  of the lowest eigenvector of H[rho] (Lanczos) to a residual below 1e-9;
* real-time evolution: second-order Strang splitting  K/2 - V - K/2;
* the box potential and all interval probabilities use partial-cell weights, so
  that results do not jump when an interval edge crosses a grid point;
* maxima over time are taken over every time step, not only over stored steps.

Wavefunctions are normalised as  sum_j |psi_j|^2 dx = 1.
"""
import numpy as np
from numpy import fft


class Grid:
    """Periodic grid together with the Fourier transform of the softened kernel."""

    def __init__(self, L, N, eps):
        self.L = float(L)
        self.N = int(N)
        self.eps = float(eps)
        self.dx = self.L / self.N
        self.x = (np.arange(self.N) - self.N // 2) * self.dx
        self.k = 2 * np.pi * fft.fftfreq(self.N, d=self.dx)

        r = np.abs(self.x)
        r = np.minimum(r, self.L - r)                     # minimum-image distance
        U = 1.0 / np.sqrt(r**2 + self.eps**2)
        self.Uk = self.dx * fft.fft(fft.ifftshift(U))     # dx = quadrature weight

    def potential(self, rho):
        """Phi_eps[rho](x) = int dy U_eps(x - y) rho(y)."""
        return fft.ifft(fft.fft(rho) * self.Uk).real

    def normalise(self, psi):
        return psi / np.sqrt(np.sum(np.abs(psi)**2) * self.dx)

    def energies(self, psi, kappa, V):
        """Kinetic, external-potential and interaction energy of psi."""
        rho = np.abs(psi)**2
        psik = fft.fft(psi) * self.dx
        e_kin = 0.5 * np.sum(self.k**2 * np.abs(psik)**2) / self.L
        e_ext = np.sum(V * rho) * self.dx
        e_int = -0.5 * kappa * np.sum(rho * self.potential(rho)) * self.dx
        return e_kin, e_ext, e_int


def cell_overlap(x, dx, a):
    """
    Length of the overlap of each grid cell [x_j - dx/2, x_j + dx/2] with the
    interval [-a, a]. Using cell fractions (instead of including whole grid points)
    removes the O(dx) error from where an interval edge falls on the grid.
    """
    lo = np.maximum(x - dx / 2, -a)
    hi = np.minimum(x + dx / 2, a)
    return np.clip(hi - lo, 0.0, dx)


def window_probability(grid, rho, a):
    """P(|x| <= a) with partial-cell weights at the edges of the interval."""
    return np.sum(rho * cell_overlap(grid.x, grid.dx, a))


def box_potential(grid, R, V0):
    """Square well of depth V0 on [-R, R], averaged over each grid cell."""
    return -V0 * cell_overlap(grid.x, grid.dx, R) / grid.dx


def gaussian(grid, sigma):
    """Normalised Gaussian with |psi|^2 of standard deviation sigma."""
    return grid.normalise(np.exp(-grid.x**2 / (4 * sigma**2)).astype(complex))


def ground_state(grid, kappa, V, psi0=None, dtaus=(0.05, 0.01, 0.002, 0.0005),
                 tol=1e-10, check_every=200, max_iter=400000):
    """
    Ground state of the SN equation with external potential V, by imaginary-time
    evolution. The splitting error is O(dtau), hence the decreasing ladder of
    steps; at each step size the iteration stops when the energy changes by less
    than `tol` per unit imaginary time.
    """
    psi = gaussian(grid, 1.0) if psi0 is None else grid.normalise(psi0.astype(complex))
    for dtau in dtaus:
        exp_kin = np.exp(-dtau * 0.5 * grid.k**2)
        e_old = np.inf
        for it in range(max_iter):
            psi = fft.ifft(exp_kin * fft.fft(psi))
            rho = np.abs(psi)**2
            psi = grid.normalise(psi * np.exp(-dtau * (V - kappa * grid.potential(rho))))
            if it % check_every == 0:
                e = sum(grid.energies(psi, kappa, V))
                if abs(e - e_old) / (check_every * dtau) < tol:
                    break
                e_old = e
    return psi


def refine_ground_state(grid, kappa, V, psi, tol=1e-9, max_iter=100):
    """
    Polish a ground-state guess by self-consistent field iteration: at each step the
    lowest eigenvector of the linear operator H[rho] = -1/2 d_x^2 + V - kappa Phi[rho]
    is computed with the Lanczos method (kinetic term applied by FFT), and rho is
    updated. Stops when the residual ||(H[rho] - mu) psi|| (L2 norm) drops below tol.

    This removes the O(dtau) bias of the imaginary-time fixed point, which otherwise
    makes the held-trap state slightly non-stationary.
    Raises RuntimeError if the tolerance is not reached. Returns (psi, residual).
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    kin = 0.5 * grid.k**2
    psi = grid.normalise(np.real(psi).astype(float))
    if psi[grid.N // 2] < 0:
        psi = -psi

    def residual(psi):
        v_eff = V - kappa * grid.potential(psi**2)
        h_psi = fft.ifft(kin * fft.fft(psi)).real + v_eff * psi
        mu = np.sum(psi * h_psi) * grid.dx
        return np.sqrt(np.sum((h_psi - mu * psi)**2) * grid.dx)

    res = residual(psi)
    for _ in range(max_iter):
        if res < tol:
            break
        v_eff = V - kappa * grid.potential(psi**2)
        op = LinearOperator((grid.N, grid.N), dtype=float,
                            matvec=lambda v, v_eff=v_eff: fft.ifft(kin * fft.fft(v)).real + v_eff * v)
        _, vec = eigsh(op, k=1, which="SA", v0=psi, tol=1e-13)
        new = grid.normalise(vec[:, 0])
        if new[grid.N // 2] < 0:
            new = -new
        psi = 0.5 * (psi + new)                  # damped update of the self-consistent state
        psi = grid.normalise(psi)
        res = residual(psi)
    if res >= tol:
        raise RuntimeError(f"ground-state refinement did not converge: residual {res:.2e} >= {tol:.0e}")
    return psi.astype(complex), res


def evolve(grid, psi, kappa, t_max, dt, n_store=300, monitor=None):
    """
    Real-time evolution with V = 0 (Strang splitting).

    The time step is reduced, if necessary, so that the densities are stored
    exactly every t_max / n_store. Returns the stored times (n_store + 1 values) and
    densities, shape (n_store + 1, N). If `monitor` is given, it is called as
    monitor(t, rho) after EVERY time step (and once at t = 0), so that maxima over
    time are not limited by the storage interval.
    """
    t_store = t_max / n_store
    every = int(np.ceil(t_store / dt))          # time steps per stored sample
    dt = t_store / every                        # dt <= requested, storage exactly every t_store
    steps = every * n_store
    half_kin = np.exp(-1j * 0.25 * grid.k**2 * dt)
    rho = np.abs(psi)**2
    times, rhos = [0.0], [rho]
    if monitor is not None:
        monitor(0.0, rho)
    for n in range(1, steps + 1):
        psi = fft.ifft(half_kin * fft.fft(psi))
        rho = np.abs(psi)**2
        psi = psi * np.exp(1j * dt * kappa * grid.potential(rho))
        psi = fft.ifft(half_kin * fft.fft(psi))
        rho = np.abs(psi)**2
        if monitor is not None:
            monitor(n * dt, rho)
        if n % every == 0:
            times.append(n * dt)
            rhos.append(rho)
    return np.array(times), np.array(rhos)


def right_tail_probability(grid, rho, a):
    """P(x > a) with partial-cell weights."""
    lo = np.maximum(grid.x - grid.dx / 2, a)
    return np.sum(rho * np.clip(grid.x + grid.dx / 2 - lo, 0.0, grid.dx))


class LightConeMonitor:
    """
    Running maxima over time of the signalling measure M(t, R) and of Bob's
    distinguishability D_t, for one trap radius R and several light speeds c.

    D_t refers to a single observer (Bob) on one side of the trap, who detects in
    x > R + c t:  D_t = |P_t(x > R + c t | released) - P_0(x > R + c t)|,
    using that the trapped state is stationary. For the symmetric states considered
    here D_t >= M(t, R) / 2. Pass `monitor.update` to `evolve`.
    """

    def __init__(self, grid, rho0, R, cs=(1.0,)):
        self.grid, self.rho0, self.R = grid, rho0, R
        self.cs = np.atleast_1d(np.asarray(cs, dtype=float))
        self.p0_trap = window_probability(grid, rho0, R)
        self.M_max = np.zeros(len(self.cs))
        self.D_max = np.zeros(len(self.cs))
        self.t_M = np.zeros(len(self.cs))

    def update(self, t, rho):
        for i, c in enumerate(self.cs):
            a = self.R + c * t
            M = max(0.0, self.p0_trap - window_probability(self.grid, rho, a))
            D = abs(right_tail_probability(self.grid, rho, a)
                    - right_tail_probability(self.grid, self.rho0, a))
            if M > self.M_max[i]:
                self.M_max[i], self.t_M[i] = M, t
            self.D_max[i] = max(self.D_max[i], D)


def leakage_M(grid, times, rhos, R, c=1.0):
    """
    Signalling measure  M(t, R) = max{0, P_0([-R, R]) - P_t([-R - c t, R + c t])}
    at the stored times: probability that escaped the future light cone of the
    trap region [-R, R].
    """
    p0 = window_probability(grid, rhos[0], R)
    pt = np.array([window_probability(grid, r, R + c * t) for t, r in zip(times, rhos)])
    return np.maximum(0.0, p0 - pt)


def spread(grid, rhos):
    """Position spread Delta_x(t) = sqrt(<x^2> - <x>^2) for each stored density."""
    m1 = rhos @ grid.x * grid.dx
    m2 = rhos @ grid.x**2 * grid.dx
    return np.sqrt(np.maximum(m2 - m1**2, 0.0))


def run_trap(L, N, eps, kappa, R, V0, t_max, c_dt, n_store=300, cs=(1.0,), dt=None):
    """
    Trap-release protocol: ground state in the box of half-width R, released at
    t = 0 and evolved freely up to t_max.

    The ground state is obtained by imaginary-time evolution followed by
    self-consistent refinement (residual below 1e-9).
    Returns (grid, ground state, times, densities, monitor); `monitor` holds the
    running maxima of M and D over all time steps for the light speeds `cs`, and
    the ground-state residual in `monitor.gs_residual`.
    The default time step is c_dt * dx^2; pass `dt` to override it.
    """
    grid = Grid(L, N, eps)
    V = box_potential(grid, R, V0)
    psi = ground_state(grid, kappa, V)
    psi, mon_residual = refine_ground_state(grid, kappa, V, psi)
    mon = LightConeMonitor(grid, np.abs(psi)**2, R, cs)
    mon.gs_residual = mon_residual
    step = c_dt * grid.dx**2 if dt is None else dt
    times, rhos = evolve(grid, psi, kappa, t_max, step, n_store, monitor=mon.update)
    return grid, psi, times, rhos, mon
