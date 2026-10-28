# Superluminal signalling in the Schrödinger–Newton model: code and data

Code and data for

> J. Osęka-Lenart, M. Płodzień, M. Lewenstein, M. Eckstein,
> *Quantifying superluminal signalling in Schrödinger–Newton model*,
> [arXiv:2512.19260](https://arxiv.org/abs/2512.19260).

The repository contains a solver for the one-dimensional Schrödinger–Newton (SN)
equation, the scripts that produced all numerical data of the paper, the data
themselves, and one script per figure that redraws it from the stored data.

## Model

All quantities are dimensionless (length unit ℓ₀, time unit t₀ = mℓ₀²/ħ). The SN equation
in one dimension reads

    i ∂ₜψ = [ −½ ∂ₓ² + V(t,x) − κ ∫dy U_ε(x−y) |ψ(t,y)|² ] ψ,     U_ε(x) = 1/√(x² + ε²).

The Newtonian kernel 1/|x| is not locally integrable in one dimension and is replaced by the
softened kernel U_ε. The parameter ε plays the role of the transverse width of a
quasi-one-dimensional system (see the Supplementary Material of the paper).

**Trap-release protocol.** The initial state is the ground state in the box potential
V(x) = −V₀ for |x| ≤ R. At t = 0 the trap is switched off and the state evolves freely.
The signalling measure

    M(t,R) = max{ 0, P₀(|x| ≤ R) − Pₜ(|x| ≤ R + c̃ t) }

measures by how much the probability inside the future light cone of the trap falls below
its initial value; M̃(κ,R) is its maximum over t ∈ [0, 30]. Bob's distinguishability D_t is
the total-variation distance between the detection statistics of a detector on one side of
the trap, outside the light cone, with the trap released and kept; for the symmetric states
considered here D_t ≥ M(t,R)/2. The dimensionless speed of light c̃ = mcℓ₀/ħ only enters through the light cone.

## Numerical method

`sn1d.py` implements:

- a periodic grid with spectral treatment of the kinetic term;
- the self-potential as an FFT convolution with the kernel periodised by the minimum-image
  convention, including the quadrature weight Δx;
- the ground state by imaginary-time split-step evolution (normalised gradient flow), with a
  decreasing ladder of imaginary-time steps, followed by self-consistent refinement (lowest
  eigenvector of H[ρ] by the Lanczos method) to a residual below 10⁻⁹;
- real-time evolution by second-order Strang splitting;
- partial-cell weights for the box potential and for all interval probabilities, so that
  results vary continuously with R and with the light-cone radius;
- maxima over time taken after every time step.

Parameters used in the paper (`config.py`):

| quantity | value |
|---|---|
| box length L | 212 |
| grid points N | 2048 (Δx = 0.1035) |
| kernel softening ε | 0.15 |
| trap depth V₀ | 20 |
| time step Δt | 0.1 Δx² |
| final time | 30 |
| Gaussian initial width σ | 1 |
| speed of light c̃ | 1 (except Fig. 4) |

The density plots use a four times larger box with the same Δx and show |x| ≤ L/2. On the
periodic grid, fast components leaving the box re-enter on the other side and produce
ripples in the far tails of ρ; they do not change M.

## Repository layout

    sn1d.py                 solver (grid, kernel, ground state, time evolution, M and D)
    config.py               parameters of the paper and paths
    generate/               scripts producing the data in data/
    data/                   stored results (.npz)
    figures/                one script per figure, plus plot_style.py
    figures/output/         figures as used in the manuscript
    generate_all_data.sh    runs all data-generation scripts
    make_all_figures.sh     runs all figure scripts

## Usage

Requirements: Python ≥ 3.9, NumPy, SciPy, Matplotlib (`pip install -r requirements.txt`).

Redraw all figures from the stored data (about one minute):

    ./make_all_figures.sh

Regenerate all data (about one hour on 16 cores; the scripts use all available cores):

    ./generate_all_data.sh

Single scripts can be run on their own, e.g. `python generate/gen_trap_scan.py --eps 0.4`
or `python figures/fig5_Mmax_map.py`. Every script states in its header which data it
reads or writes.

## Figures

| Figure in the paper | Script | Data | Output file(s) |
|---|---|---|---|
| Main, Fig. 2: ρ(t,x), released ground state, R = 2 | `figures/fig2_trap_density.py` | `trap_dynamics.npz` | `rho_SN_gs_2.0_*.png` |
| Main, Fig. 3: same, log scale, light cone | `figures/fig3_trap_density_log.py` | `trap_dynamics.npz` | `rho_wideSN_gs_2.0_*.png` |
| Main, Fig. 4: max D_t vs κ for several c̃ | `figures/fig4_distinguishability_vs_c.py` | `c_scan.npz` | `fig_M_vs_c.png` |
| Main, Fig. 5: M̃(κ,R) | `figures/fig5_Mmax_map.py` | `trap_scan_eps0.15_N2048.npz` | `fig_M_max_g_R_color_long_range_gs.png` |
| SM: ρ(t,x), Gaussian initial state | `figures/figS_gauss_density.py` | `gauss_dynamics.npz` | `rho_SN_gauss_5.0_*.png`, `rho_wideSN_gauss_5.0_*.png` |
| SM: spread Δx(t,κ) of the Gaussian | `figures/figS_gauss_spread.py` | `spread_scan.npz` | `fig_dx_t_g_long_range_gauss.png` |
| SM: M(t,R) for the Gaussian | `figures/figS_gauss_M_t_R.py` | `gauss_dynamics.npz` | `fig_M_t_R_long_range_gauss_g_*.png` |
| SM: ground-state densities and E_int/E_kin | `figures/figS_ground_state.py` | `trap_dynamics.npz`, `trap_scan_eps0.15_N2048.npz` | `rho_INI_SN_gs_2.0_-1.0.png`, `fig_E_g_R_long_range_gs.png` |
| SM: density at t = 30, R = 1 | `figures/figS_density_t30.py` | `trap_dynamics.npz` | `rho_x_SN_gs_1.0_*.png` |
| SM: grid convergence and ε dependence | `figures/figS_convergence.py` | `trap_scan_*.npz` | `fig_convergence_dx.png`, `fig_convergence_eps.png` |

Output file names follow the manuscript sources, so the figures can be copied there directly.
In the names, `_2.0_`, `_1.0_`, `_5.0_` refer to R and the last field to −κ.

## Data files

All files are NumPy `.npz` archives; the parameters of each run are stored alongside the arrays.

| File | Content |
|---|---|
| `trap_dynamics.npz` | ρ(t,x) for the released ground state, (R, κ) ∈ {2}×{0, 0.1, 1, 3} ∪ {1}×{0, 1}; 301 times |
| `gauss_dynamics.npz` | ρ(t,x) for the Gaussian initial state, κ ∈ {0, 0.1, 0.5, 1, 3}; 301 times |
| `spread_scan.npz` | Δx(t) for the Gaussian, κ = 0…0.9 |
| `c_scan.npz` | max_t M and max_t D for c̃ ∈ {1, 2, 5, 10}, R ∈ {1, 2, 3} |
| `trap_scan_eps{ε}_N{N}[_L{L}][_subset][_R123].npz` | M̃(κ,R), time of the maximum, E_kin and E_int of the ground state; `_L424` is the box-size test, `_subset` a coarse (κ,R) grid including R = 1.5, `_R123` the full κ grid for R = 1, 2, 3 |
| `dt_check.npz` | M̃ at eight (κ, R) points with the paper's time step and with Δt/4 |

## Notes on accuracy

- Values of M̃ and D below 10⁻⁶ are treated as zero.
- The strongly self-bound ground states at large κ are narrow (width ≈ 0.25), which is why
  N = 2048 is used. With N = 1024 the leakage at large κ and small R is strongly
  underestimated; N = 2048 agrees with N = 4096 to 1.5 % (median) and to within 30 % at the
  most difficult point (κ = 4, R = 1.5, where M̃ ≈ 2 × 10⁻⁶).
- For c̃ ≳ 10 the light cone crosses a grid cell within a few time steps; the c̃ scan therefore
  stops at c̃ = 10 and is meant to show the trend with c̃, not precise values.

## Citation

If you use this code, please cite the paper above.

## License

MIT, see [LICENSE](LICENSE).
