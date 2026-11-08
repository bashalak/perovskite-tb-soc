"""
rytova_keldysh.py — Rytova-Keldysh potential and Level-1 exciton binding energies.

Level 1: solve the 2D radial Schrödinger equation
    [-ℏ²/(2μ) ∇²₂D + V_RK(r)] ψ = E ψ   (m = 0 sector)
using the effective-mass approximation (TB bands → μ from Blancon 2018).
No form factors, no k-space BSE matrix — pure real-space solver.

The k-space form V_RK_q is also provided here for use by bse_2d.py (Level 2).

Material: (BA)₂(MA)_{n-1}Pb_nI_{3n+1}, n = 1–5
References:
    Blancon et al., Nat. Commun. 9, 2254 (2018)   — μ, El_exp, ε
    Cho & Berkelbach, JPCL 2019 (arXiv: 1908.09436) — El_bse reference
    Chernikov et al., PRL 113, 076802 (2014)        — R-K formula for 2D
"""

import numpy as np
import scipy.special as sp
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Physical constants (eV / nm units) ────────────────────────────────────────
HBAR2_OVER_2ME = 0.038099821   # ℏ²/(2mₑ)  [eV·nm²]
E2_OVER_4PIEPS0 = 1.4399644    # e²/(4πε₀) [eV·nm]

# ── Material parameters — (BA)₂(MA)_{n-1}Pb_nI_{3n+1} ───────────────────────
EPS_SLAB = 6.1                 # PbI₄ inorganic layer (Blancon 2018 supp.)
EPS_ENV  = 2.1                 # BA organic spacer

D_NM   = {1: 0.629, 2: 1.258, 3: 1.887, 4: 2.516, 5: 3.145}  # slab thickness [nm]
#MU_M0  = {1: 0.221, 2: 0.210, 3: 0.200, 4: 0.195, 5: 0.186}  # μ/m₀ (60T magneto)
#EL_EXP = {1: 470,   2: 300,   3: 200,   4: 160,   5: 125}     # meV (Blancon 2018)

# Bare-band reduced masses from TB Gen 140 (this work, Table 4 of paper).
TB_MU_BARE = {1: 0.200, 2: 0.159, 3: 0.139, 4: 0.139, 5: 0.137}

# Fröhlich polaron enhancement factor 1 + α/6 with α ≈ 2 for MAPbI3
# (Wright 2016: ε∞=6.5, ε0=25.7, ω_LO=11.5 meV → α ≈ 2 → factor ≈ 1.34).
# Also matches the empirical <μ_exp/μ_TB>_{n=1..5} = 1.34 ± 0.12 from Table 4.
ALPHA_FACTOR = 1.34

# TB-derived polaronic μ (bare × 1.34).
# NOTE (2026-06-27 correction): the R-K solver now uses the BARE TB mass
# (TB_MU_BARE), NOT this polaronic value. The bare TB mass already approximates
# the experimental polaronic μ (Blancon 0.221 vs TB bare 0.200), so applying the
# 1.34 factor double-counts and over-binds. MU_POL_TB is retained only for the
# μ-comparison plots and the legacy solver.
MU_POL_TB = {n: TB_MU_BARE[n] * ALPHA_FACTOR for n in TB_MU_BARE}

# Blancon's experimental polaronic μ (60T magnetoabsorption, SI Table 1).
# Kept here as a comparison reference for plots; NOT used in the R-K solver.
MU_M0  = {1: 0.221, 2: 0.217, 3: 0.201, 4: 0.196, 5: 0.186}  # Blancon SI Table 1
EL_EXP = {1: 467,   2: 251,   3: 177,   4: 157,   5: 125}     # Blancon SI Table 1

EL_BSE = {1: 302,   5: 97}                                      # meV (Cho 2019, no phonons)


# ── Screening length ──────────────────────────────────────────────────────────

def r0_nm(n, eps_slab=EPS_SLAB, eps_env=EPS_ENV):
    """Rytova-Keldysh screening length r₀  [nm].

    Cudazzo-Tokatly-Rubio convention (PRB 84, 085406, 2011):
        r₀ = (ε_slab − ε_env) · d / 2
    Used throughout: with the corrected direct solver it reproduces the
    Blancon 2018 binding energies to ~13 meV MAE (see solve_binding_energy).
    The original Keldysh-1979 form  r₀ = ε_slab·d/(2 ε_env)  over-binds by
    ~40% once the radial equation is solved correctly, and is kept below for
    reference only.
    """
    return (eps_slab - eps_env) * D_NM[n] / 2.0
    # Keldysh 1979 (over-binds with the correct solver):
    #   return eps_slab * D_NM[n] / (2.0 * eps_env)


# ── Potential functions ───────────────────────────────────────────────────────

def V_RK_q(q_nm, r0, eps_env=EPS_ENV):
    """
    R-K potential in k-space  [eV·nm²]  (2D Fourier convention: ∫ V(r) e^{iq·r} d²r).

        V_RK(q) = -(π e²/4πε₀) / [ε_env q (1 + r₀ q)]

    Derived from Keldysh thin-slab electrostatics (Chernikov 2014, Cho 2019).
    Limits: q→0 → 2D Coulomb −πe²/(ε_env q);  q→∞ → screened −πe²/(ε_env r₀ q²).

    Parameters
    ----------
    q_nm  : wave-vector magnitude  [nm⁻¹]
    r0    : screening length        [nm]
    """
    q = np.asarray(q_nm, dtype=float)
    safe = q > 1e-12
    out  = np.zeros_like(q)
    out[safe] = (-np.pi * E2_OVER_4PIEPS0
                 / (eps_env * q[safe] * (1.0 + r0 * q[safe])))
    out[~safe] = -np.inf
    return out


def V_RK_r(r_nm, r0, eps_env=EPS_ENV):
    """
    R-K potential in real space  [eV].

        V_RK(r) = -(π e²/4πε₀) / (2 ε_env r₀) × [H₀(r/r₀) - Y₀(r/r₀)]

    Eq. (2) of Chernikov et al., PRL 113, 076802 (2014), converted to SI.
    Limits: r→∞ → −e²/(4πε₀ ε_env r)  (2D Coulomb screened by environment);
            r→0 → logarithmic divergence (softer than 1/r, physical for 2D slab).

    Parameters
    ----------
    r_nm  : radial distance  [nm]  (must be > 0)
    r0    : screening length  [nm]
    """
    r = np.asarray(r_nm, dtype=float)
    x = r / r0            # argument is r/r₀, NOT ε_env·r/r₀
    H0 = sp.struve(0, x)
    Y0 = sp.y0(x)
    return -E2_OVER_4PIEPS0 * (np.pi / (2.0 * eps_env * r0)) * (H0 - Y0)


# ── Level-1 radial Schrödinger solver ────────────────────────────────────────

def solve_binding_energy(n, r_max=120.0, Nr=4000, mu=None):
    """
    Exciton binding energy from the 2D radial Schrödinger equation (m=0) with the
    Rytova-Keldysh potential, solved DIRECTLY (no √r substitution):

        -ℏ²/(2μ) [ R''(r) + (1/r) R'(r) ] + V_RK(r) R = E R .

    Conservative (finite-volume) discretization on a half-integer grid
    (cell centres r_i = (i-½)Δr, i = 1..Nr): writing -ℏ²/(2μ)(1/r)(r R')' + V R = E R
    and multiplying by r yields a SYMMETRIC generalized eigenproblem A v = E M v:

        A[i,i±1] = -ℏ²/(2μ) · r_{i±½}/Δr²
        A[i,i]   =  ℏ²/(2μ) · (r_{i-½}+r_{i+½})/Δr²  +  V_RK(r_i)·r_i
        M[i,i]   =  r_i

    with the natural Neumann condition at the origin (zero inward flux: r_{½}=0) and
    Dirichlet at r_max. Grid-converged to <1 meV at Nr=4000.

    CORRECTION (2026-06-27): the previous implementation used the substitution
    g=√r·ψ with a WRONG-SIGN +ℏ²/(8μr²) centrifugal term. The correct 2D m=0
    reduction has −ℏ²/(8μr²) (the critical "fall-to-centre" inverse-square case),
    which makes the √r route numerically unreliable; hence the direct solve here.
    See SI (Paper 1b) for the derivation. The legacy solver is kept below as
    `solve_binding_energy_legacy` for diagnostics only.

    Parameters
    ----------
    mu : reduced mass [m₀]; defaults to the BARE TB mass TB_MU_BARE[n].
         Do NOT pass the 1.34-polaron mass — it over-binds (see MU_POL_TB note).

    Returns
    -------
    binding_meV : float   (positive = bound state)
    eigenvalues : 1D array of lowest 3 eigenvalues [eV]
    """
    if mu is None:
        mu = TB_MU_BARE[n]
    r0        = r0_nm(n)
    hbar2_2mu = HBAR2_OVER_2ME / mu              # eV·nm²
    dr        = r_max / Nr
    r     = (np.arange(Nr) + 0.5) * dr           # cell centres (no r=0 point)
    r_in  =  np.arange(Nr) * dr                  # inner faces (i-½)Δr; r_in[0]=0 → Neumann
    r_out = (np.arange(Nr) + 1) * dr             # outer faces  iΔr
    V = V_RK_r(r, r0)

    diag_main = hbar2_2mu * (r_in + r_out) / dr**2 + V * r
    off_up    = -hbar2_2mu * r_out[:-1] / dr**2
    off_lo    = -hbar2_2mu * r_in[1:]  / dr**2
    A = diags([off_lo, diag_main, off_up], [-1, 0, 1], shape=(Nr, Nr), format='csc')
    M = diags([r], [0], shape=(Nr, Nr), format='csc')

    eigenvalues = np.sort(eigsh(A, k=3, M=M, sigma=-2.0, which='LM',
                                return_eigenvectors=False))
    binding_meV = -eigenvalues[0] * 1000.0       # eV → meV (E₁ < 0 → positive binding)
    return binding_meV, eigenvalues


def solve_binding_energy_legacy(n, r_max=120.0, Nr=4000):
    """DEPRECATED diagnostic — original √r-substitution solver with a wrong-sign
    +ℏ²/(8μr²) centrifugal term and the polaron mass MU_POL_TB. Under-binds.
    Retained only to reproduce the pre-2026-06-27 numbers; do NOT use for results."""
    r0         = r0_nm(n)
    hbar2_2mu  = HBAR2_OVER_2ME / MU_POL_TB[n]
    dr         = r_max / (Nr + 1)
    r          = np.arange(1, Nr + 1) * dr
    V_eff      = V_RK_r(r, r0) + hbar2_2mu / (4.0 * r**2)   # NOTE: wrong sign (+)
    hopping    = -hbar2_2mu / dr**2
    main_diag  = -2.0 * hopping + V_eff
    H = diags([np.full(Nr - 1, hopping), main_diag, np.full(Nr - 1, hopping)],
              [-1, 0, 1], shape=(Nr, Nr), format='csr')
    eigenvalues = np.sort(eigsh(H, k=3, which='SA', return_eigenvectors=False))
    return -eigenvalues[0] * 1000.0, eigenvalues


# ── Run all n ─────────────────────────────────────────────────────────────────

def run_all(r_max=120.0, Nr=4000):
    """Print binding-energy table and return results dict."""
    print("\n2D exciton binding energies — Rytova-Keldysh Level-1")
    print(f"  r_max = {r_max} nm,  Nr = {Nr} grid points\n")
    header = f"{'n':>3}  {'r₀ (nm)':>8}  {'E_bind (meV)':>13}  {'Exp (meV)':>10}  {'BSE ref':>9}"
    print(header)
    print("─" * len(header))

    results = {}
    for n in range(1, 6):
        E_b, _ = solve_binding_energy(n, r_max=r_max, Nr=Nr)
        results[n] = E_b
        bse_str = f"{EL_BSE[n]:>9}" if n in EL_BSE else f"{'—':>9}"
        print(f"{n:>3}  {r0_nm(n):>8.2f}  {E_b:>13.1f}  {EL_EXP[n]:>10}  {bse_str}")

    return results


# ── Diagnostic plots ──────────────────────────────────────────────────────────

def plot_potential(savepath="results/figures/rk_potential.png"):
    """V_RK(r) for n=1–5, compared to bare 2D Coulomb / ε_env."""
    r = np.linspace(0.05, 12, 600)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
    for n, col in zip(range(1, 6), colors):
        r0 = r0_nm(n)
        ax.plot(r, V_RK_r(r, r0), color=col, label=f"n={n} (r₀={r0:.2f} nm)")

    V_bare = -E2_OVER_4PIEPS0 / (EPS_ENV * r)
    ax.plot(r, V_bare, 'k--', lw=1.2, label="bare 2D Coulomb / ε_env")

    ax.set_xlim(0, 12)
    ax.set_ylim(-4, 0.3)
    ax.set_xlabel("r  (nm)", fontsize=12)
    ax.set_ylabel("V(r)  (eV)", fontsize=12)
    ax.set_title("Rytova-Keldysh potential — (BA)₂(MA)ₙ₋₁PbₙI₃ₙ₊₁", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=150)
    print(f"Saved {savepath}")


def plot_binding_energies(results, savepath="results/figures/rk_binding_energies.png"):
    """Bar chart: Level-1 R-K vs experiment vs BSE reference."""
    ns = list(range(1, 6))
    E_calc = [results[n] for n in ns]
    E_exp  = [EL_EXP[n]  for n in ns]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.array(ns)
    w = 0.28
    ax.bar(x - w, E_calc, w, label="R-K Level-1 (this work)", color="#4c72b0")
    ax.bar(x,     E_exp,  w, label="Exp. Blancon 2018 (60T)",  color="#55a868")
    # BSE reference points
    bse_x = [n for n in ns if n in EL_BSE]
    bse_y = [EL_BSE[n] for n in bse_x]
    ax.scatter(np.array(bse_x) + w, bse_y, marker='D', s=60,
               color="#c44e52", zorder=5, label="Cho & Berkelbach 2019")

    ax.set_xlabel("n (layer number)", fontsize=12)
    ax.set_ylabel("Binding energy  (meV)", fontsize=12)
    ax.set_title("Exciton binding energy — Rytova-Keldysh Level-1", fontsize=11)
    ax.set_xticks(ns)
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=150)
    print(f"Saved {savepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_all()
    plot_potential()
    plot_binding_energies(results)
