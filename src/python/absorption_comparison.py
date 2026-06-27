"""
absorption_comparison.py — Phase 5: experimental vs BSE Elliott absorption.

Loads digitized OD spectra from Blancon et al. Nat. Commun. 9, 2254 (2018),
Fig. 3d for (BA)₂(MA)_{n-1}Pb_n I_{3n+1}, n = 1–5, and overlays them against
the BSE Elliott spectrum computed by absorption_elliott.py / bse_2d.py.

Normalisation protocol (stacked figure → per-curve baseline):
  1. Subtract baseline = mean OD of first 6 data points (below any feature).
  2. Find 1s exciton peak in a per-n energy window; normalise peak → 1.
  3. Centre x-axis on E_1s for both theory and experiment: ΔE (meV).

Output: results/figures/absorption_comparison_exp_vs_bse.png

Run from src/python/:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 absorption_comparison.py [Nk]   (default Nk=21)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bse_2d import solve_bse
from absorption_elliott import absorption_spectrum

# ── Per-n energy windows to locate the 1s exciton peak in experiment ──────────
# These bracket the known 1s peak and exclude the Rydberg / continuum region.
PEAK_WINDOW_EV = {
    1: (2.45, 2.68),
    2: (2.13, 2.25),
    3: (1.97, 2.08),
    4: (1.87, 1.96),
    5: (1.82, 1.89),
}

# Path to experimental data files (relative to this script)
_HERE   = os.path.dirname(os.path.abspath(__file__))
_DAT    = os.path.join(_HERE, "..", "..", "results", "data")
_FIGS   = os.path.join(_HERE, "..", "..", "results", "figures")


def load_exp(n):
    """
    Load, baseline-subtract and normalise experimental OD for slab n.

    Returns
    -------
    E_rel   : energy relative to 1s peak [meV]
    od_norm : normalised OD (1s peak = 1)
    E_1s    : 1s exciton energy [eV]
    """
    path = os.path.join(_DAT, f"absorption_exp_n{n}.dat")
    data = np.loadtxt(path, comments="#")
    E   = data[:, 0]
    od  = data[:, 1]

    # Subtract per-curve baseline using first 6 points (pre-exciton background)
    baseline = np.mean(od[:6])
    od = od - baseline
    od = np.maximum(od, 0.0)   # clip any negative noise after subtraction

    # Find 1s peak in search window
    lo, hi = PEAK_WINDOW_EV[n]
    mask = (E >= lo) & (E <= hi)
    if not np.any(mask):
        raise ValueError(f"n={n}: no data in peak window [{lo}, {hi}] eV")
    idx_peak = np.argmax(od[mask])
    # map back to full array index
    full_idx = np.where(mask)[0][idx_peak]
    E_1s     = E[full_idx]
    peak_od  = od[full_idx]

    od_norm = od / peak_od          # 1s peak → 1
    E_rel   = (E - E_1s) * 1000.0  # meV, centred at 0

    return E_rel, od_norm, E_1s


def bse_spectrum_centred(res, Gamma_meV=20.0, n_E=2000):
    """
    Return BSE Elliott spectrum centred on the 1s exciton peak.

    Returns
    -------
    E_rel   : energy relative to BSE 1s peak [meV]
    alpha   : total absorption (exciton + continuum), peak ≈ 1
    """
    E_gap = res["E_gap"]
    E_1s  = E_gap + res["omega"][0]   # 1s exciton energy [eV]

    E, alpha_exc, alpha_cont = absorption_spectrum(res, Gamma_meV=Gamma_meV,
                                                   n_E=n_E)
    alpha = alpha_exc + alpha_cont

    # Normalise to 1s exciton peak (alpha_exc maximum, same normalisation as exp)
    peak = alpha_exc.max()
    if peak > 0:
        alpha = alpha / peak

    E_rel = (E - E_1s) * 1000.0   # meV
    return E_rel, alpha, E_1s


def plot_comparison(Nk=21, Gamma_meV=20.0,
                    savepath=None):
    """
    5-panel figure: experiment vs BSE Elliott for n = 1–5.
    Both spectra are centred on their respective 1s exciton peaks.
    """
    if savepath is None:
        savepath = os.path.join(_FIGS, "absorption_comparison_exp_vs_bse.png")

    fig, axes = plt.subplots(5, 1, figsize=(7.5, 12), sharex=False)
    fig.suptitle(
        "Optical absorption:  experiment (Blancon 2018)  vs  BSE Elliott (this work)\n"
        r"$({\rm BA})_2({\rm MA})_{n-1}{\rm Pb}_n{\rm I}_{3n+1}$",
        fontsize=11, y=1.00
    )

    # x-axis display range [meV] around 1s peak
    X_LO, X_HI = -500, 1500

    for i, n in enumerate(range(1, 6)):
        ax = axes[i]

        # ── Experimental spectrum ──────────────────────────────────────────────
        try:
            E_rel_exp, od_norm, E_1s_exp = load_exp(n)
            mask_exp = (E_rel_exp >= X_LO) & (E_rel_exp <= X_HI)
            ax.fill_between(E_rel_exp[mask_exp], od_norm[mask_exp],
                            alpha=0.35, color="#4c72b0", label="Exp. (Blancon 2018)")
            ax.plot(E_rel_exp[mask_exp], od_norm[mask_exp],
                    color="#4c72b0", lw=1.2)
            exp_loaded = True
        except FileNotFoundError:
            exp_loaded = False
            E_1s_exp   = float("nan")

        # ── BSE Elliott theory ─────────────────────────────────────────────────
        print(f"  n={n}: running BSE (Nk={Nk}) ...", flush=True)
        res                       = solve_bse(n, Nk=Nk)
        E_rel_th, alpha, E_1s_th  = bse_spectrum_centred(res, Gamma_meV=Gamma_meV)
        mask_th = (E_rel_th >= X_LO) & (E_rel_th <= X_HI)
        ax.plot(E_rel_th[mask_th], alpha[mask_th],
                color="#c44e52", lw=1.8, ls="--", label="BSE Elliott (this work)")

        # ── Vertical markers ───────────────────────────────────────────────────
        ax.axvline(0.0, color="gray", lw=0.7, ls=":")    # 1s peak at 0 for both
        # Band gap position in BSE (E_gap - E_1s_bse = E_bind > 0)
        E_B_meV = res["E_bind_meV"]
        ax.axvline(E_B_meV, color="#c44e52", lw=0.7, ls=":", alpha=0.6)

        # ── Annotations ───────────────────────────────────────────────────────
        label_n = f"n = {n}"
        ax.text(0.02, 0.88, label_n, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

        info = f"$E_{{\\rm B}}^{{\\rm BSE}}$ = {E_B_meV:.0f} meV"
        if exp_loaded:
            info += f"\n$E_{{1s}}^{{\\rm exp}}$ = {E_1s_exp:.3f} eV"
        ax.text(0.98, 0.88, info, transform=ax.transAxes,
                fontsize=8.5, ha="right", va="top", linespacing=1.5)

        ax.set_xlim(X_LO, X_HI)
        ax.set_ylim(bottom=0)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel("Norm. absorption", fontsize=9)

        if i == 0:
            ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 0.76))

    axes[-1].set_xlabel(r"$E - E_{1s}$  (meV)", fontsize=12)

    fig.text(0.5, -0.01,
             "Note: 'A/B/C' features (interband transitions) and Rydberg 2s/3s states\n"
             "above $E_{\\rm gap}$ are not reproduced by the Elliott formula.",
             ha="center", fontsize=8, color="0.45", linespacing=1.4)

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=150, bbox_inches="tight")
    print(f"\nSaved {savepath}")
    plt.close(fig)

    return savepath


def print_summary(Nk=21):
    """Print table: n, E_1s_exp, E_1s_bse, E_B_bse, shift (BSE – exp)."""
    print(f"\n{'n':>3}  {'E_1s_exp(eV)':>13}  {'E_1s_BSE(eV)':>13}  "
          f"{'E_B_BSE(meV)':>13}  {'shift(meV)':>11}")
    print("─" * 60)
    for n in range(1, 6):
        try:
            _, _, E_1s_exp = load_exp(n)
        except FileNotFoundError:
            E_1s_exp = float("nan")
        res     = solve_bse(n, Nk=Nk)
        E_1s_th = res["E_gap"] + res["omega"][0]
        shift   = (E_1s_th - E_1s_exp) * 1000
        print(f"{n:>3}  {E_1s_exp:>13.4f}  {E_1s_th:>13.4f}  "
              f"{res['E_bind_meV']:>13.1f}  {shift:>+11.0f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Nk = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    print(f"absorption_comparison.py  Nk={Nk}\n")
    plot_comparison(Nk=Nk)
    print_summary(Nk=Nk)
