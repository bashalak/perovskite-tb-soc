"""
absorption_elliott.py — Phase 5 Level-2: optical absorption spectrum.

Uses BSE output (exciton energies + oscillator strengths from bse_2d.py)
to build the optical absorption spectrum via the Elliott formula:

    alpha(E) ∝  sum_s  f_s * Lorentzian(E - Omega_s, Gamma)     [exciton peaks]
             +  theta(E - E_gap) * rho_2D * S_2D(E, E_B)         [continuum]

where:
    Omega_s = E_gap + omega_s    exciton resonance energy [eV]
    f_s                          BSE oscillator strength
    rho_2D  = const              2D joint density of states (parabolic bands)
    S_2D(E) = 2pi*xi / (1 - exp(-2pi*xi)),  xi = sqrt(E_B / (E - E_gap))
              2D Sommerfeld factor (Elliott 1957, adapted for 2D)
    E_B     = E_bind_meV / 1000  effective binding energy from BSE [eV]

Run from src/python/:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 absorption_elliott.py [Nk]   (default Nk=21)
"""

import numpy as np
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bse_2d import solve_bse, r0_nm, EL_EXP


def lorentzian(E, E0, Gamma):
    """Normalized Lorentzian L(E; E0, Gamma) = (Gamma/pi) / ((E-E0)^2 + Gamma^2)."""
    return (Gamma / np.pi) / ((E - E0) ** 2 + Gamma ** 2)


def sommerfeld_2d(E, E_gap, E_B):
    """
    2D Sommerfeld (Coulomb enhancement) factor for free-carrier absorption.

        S(E) = 2*pi*xi / (1 - exp(-2*pi*xi)),   xi = sqrt(E_B / (E - E_gap))

    Returns 1 below the gap (no continuum there).
    """
    out = np.ones_like(E)
    above = E > E_gap
    if E_B <= 0:
        return out
    xi = np.sqrt(np.maximum(E_B / (E[above] - E_gap), 0.0))
    twopix = 2.0 * np.pi * xi
    # avoid overflow: for large xi, S → 2*pi*xi
    safe = twopix < 700.0
    S    = np.where(safe, twopix / (1.0 - np.exp(-twopix)), twopix)
    out[above] = S
    return out


def absorption_spectrum(bse_result, Gamma_meV=20.0, n_E=2000):
    """
    Compute absorption spectrum alpha(E) from BSE result.

    Parameters
    ----------
    bse_result : dict returned by bse_2d.solve_bse
    Gamma_meV  : Lorentzian broadening [meV]  (default 20 meV)
    n_E        : number of energy points

    Returns
    -------
    E     : (n_E,) energy array [eV]
    alpha : (n_E,) absorption (arbitrary units, normalised to continuum ~ 1)
    """
    E_gap  = bse_result["E_gap"]
    omega  = bse_result["omega"]          # relative to E_gap [eV]; < 0 = bound
    osc    = bse_result["osc"]            # oscillator strengths
    E_B    = bse_result["E_bind_meV"] / 1000.0   # [eV]
    Gamma  = Gamma_meV / 1000.0          # [eV]

    # Energy window: 4*E_B below gap to 3*E_B above gap
    E_lo = E_gap - 4.0 * E_B
    E_hi = E_gap + 3.0 * E_B
    E    = np.linspace(E_lo, E_hi, n_E)

    # Exciton peaks (bound states: omega < 0)
    alpha_exc = np.zeros(n_E)
    for s, (om, f) in enumerate(zip(omega, osc)):
        if om < 0.5 * E_B:  # only clearly bound states
            E_s = E_gap + om
            alpha_exc += f * lorentzian(E, E_s, Gamma)

    # Normalise exciton contribution to 1 at its maximum
    peak = alpha_exc.max()
    if peak > 0:
        alpha_exc /= peak

    # Free-carrier continuum with 2D Sommerfeld enhancement
    alpha_cont  = np.where(E >= E_gap, 1.0, 0.0) * sommerfeld_2d(E, E_gap, E_B)
    # Scale continuum so it matches ~ 1 far above gap
    cont_far    = alpha_cont[E > E_gap + 0.5 * E_B]
    if cont_far.size > 0 and cont_far.max() > 0:
        alpha_cont /= cont_far.max()

    return E, alpha_exc, alpha_cont


def plot_absorption(n, Nk=21, Gamma_meV=20.0,
                    savepath=None):
    """Compute and plot absorption spectrum for slab n."""
    print(f"  n={n}: solving BSE (Nk={Nk}) ...", flush=True)
    res = solve_bse(n, Nk=Nk)

    E, alpha_exc, alpha_cont = absorption_spectrum(res, Gamma_meV=Gamma_meV)

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(E * 1000 - res["E_gap"] * 1000, alpha_cont,
            color="0.65", lw=1.5, label="Continuum + Sommerfeld")
    ax.plot(E * 1000 - res["E_gap"] * 1000, alpha_exc,
            color="#4c72b0", lw=2.0, label="Exciton peaks (BSE)")
    ax.plot(E * 1000 - res["E_gap"] * 1000, alpha_exc + alpha_cont,
            color="#c44e52", lw=1.5, ls="--", label="Total")

    # Mark 1s exciton position
    om0  = res["omega"][0] * 1000   # meV
    ax.axvline(om0, color="#4c72b0", lw=0.8, ls=":")
    ax.axvline(0,   color="black",   lw=0.8, ls=":")

    ax.set_xlabel("E − E$_{gap}$  (meV)", fontsize=12)
    ax.set_ylabel("Absorption  (arb. units)", fontsize=12)
    ax.set_title(f"Optical absorption — (BA)₂(MA)$_{{n-1}}$Pb$_n$I$_{{3n+1}}$, n={n}  "
                 f"[BSE Level-2, Γ={Gamma_meV} meV]", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    if savepath is None:
        savepath = f"results/figures/absorption_bse_n{n}.png"
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=150)
    print(f"  Saved {savepath}")
    plt.close(fig)
    return res


def plot_all(Nk=21, Gamma_meV=20.0, savepath=None):
    """
    Single figure with absorption spectra for n=1–5 stacked vertically.
    """
    if savepath is None:
        _here = os.path.dirname(os.path.abspath(__file__))
        savepath = os.path.join(_here, "..", "..", "results", "figures",
                                "absorption_bse_all.png")
    fig, axes = plt.subplots(5, 1, figsize=(7, 12), sharex=False)
    fig.suptitle(
        f"BSE optical absorption — (BA)₂(MA)$_{{n-1}}$Pb$_n$I$_{{3n+1}}$  "
        f"[Γ={Gamma_meV} meV]",
        fontsize=11, y=1.01
    )

    for i, n in enumerate(range(1, 6)):
        res = solve_bse(n, Nk=Nk)
        E, alpha_exc, alpha_cont = absorption_spectrum(res, Gamma_meV=Gamma_meV)
        E_rel = (E - res["E_gap"]) * 1000   # meV relative to gap

        ax = axes[i]
        ax.fill_between(E_rel, alpha_cont, alpha=0.25, color="0.5")
        ax.plot(E_rel, alpha_exc + alpha_cont, color="#c44e52", lw=1.5)
        ax.plot(E_rel, alpha_exc, color="#4c72b0", lw=1.5, ls="--")

        ax.axvline(res["omega"][0] * 1000, color="#4c72b0", lw=0.8, ls=":")
        ax.axvline(0, color="black", lw=0.8, ls=":")

        ax.set_ylabel(f"n={n}", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        E_b_str = f"E_B={res['E_bind_meV']:.0f} meV"
        ax.text(0.97, 0.85, E_b_str, transform=ax.transAxes,
                ha="right", fontsize=9)

    axes[-1].set_xlabel("E − E$_{gap}$  (meV)", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    print(f"Saved {savepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Nk = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    plot_all(Nk=Nk)
