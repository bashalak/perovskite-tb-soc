"""
plot_2n_scaling.py — Figure 3 of Paper 1: the 2/n scaling-rule signature plot.

Shows that the 2D Ruddlesden-Popper (BA)2(MA)_{n-1}Pb_nI_{3n+1} quasiparticle
gap Eg(n) is linear in 2/n across n=1..5 for three methods:

    Eg(n) = Eg_∞ + ΔE_surf × (2/n)

A single intercept (Eg_∞) + single slope (ΔE_surf) reproduces the n-dependence
of all five slabs.  In our TB model this corresponds to ONE set of three
surface-correction parameters (E_s1,ap, E_p1,ap, E_p0,surf) accounting for
all 5 gaps — the "compression" claim of the paper.

Three series overlaid:
    • TB Gen 140 (this work)        — filled red circles
    • DFT-1/2 (Traoré 2022)         — empty blue squares
    • Experiment (Blancon 2018 E_el)— black triangles
Each fitted with weighted least squares; slope + intercept annotated.

A horizontal dashed line marks the 3D Pnma TB Eg(Γ) = 1.457 eV — visualising
that the 2D-slab n→∞ extrapolation does NOT equal the 3D Pnma gap, because
the RP slabs have BA spacers and different tilting (~0.5 eV residual).

Output: report/figures/fig_2n_scaling.{png,pdf}

Run:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 plot_2n_scaling.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')

# ── data (from report/tables/table3_2D_observables.md, all verified) ──────────
N = np.array([1, 2, 3, 4, 5], dtype=float)
TWO_OVER_N = 2.0 / N

# 2D RP Eg(n) — three series
EG_TB        = np.array([2.840, 2.457, 2.205, 2.181, 2.110])   # this work
EG_DFT12     = np.array([2.84,  2.45,  2.29,  2.32,  2.01 ])   # Traoré 2022
EG_EXP       = np.array([3.016, 2.432, 2.206, 2.078, 1.971])   # Blancon 2018 E_elec

# 3D Pnma TB Eg(Γ) — horizontal reference
EG_3D_TB     = 1.457
EG_3D_DFT12  = 1.52      # Traoré target

# Experimental Pnma references for the 2D n→∞ extrapolation comparison.
# Band-edge states in Pb-I perovskites are dominated by the Pb-I octahedral
# framework with negligible A-site contribution [EVEN2013, BOYER2016];
# Cs vs MA A-site shifts Eg by only ~70 meV at fixed phase.
#
# Two reference values shown in the figure to bracket the bulk Pnma range:
EG_3D_MAPBI3_PNMA_EXP = 1.65   # MAPbI3 Pnma, low-T PL  [Traoré 2022 ref 70]
                                # — same A-site as 2D RP slabs, slightly LOWER
                                #   due to stronger MA-driven octahedral tilting
EG_3D_CSPBI3_PNMA_EXP = 1.72   # β-CsPbI3 Pnma, exp  [Marronnier 2018, ACS Nano
                                # 12, 3477; Pedesseau co-author]
                                # — different A-site, closer to the 2D slab
                                #   pseudocubic tilting; falls within 40 meV
                                #   of the Blancon Eg_∞ extrapolation

# DPI / output
DPI = 220


def lin_fit(x, y):
    """Plain least-squares y = a + b*x. Returns (intercept a, slope b, R²)."""
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    y_hat = a + b*x
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot
    return a, b, r2


def main():
    out_dir = os.path.join(_ROOT, 'report', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    # Linear fits on Eg vs 2/n
    a_tb,    b_tb,    r2_tb    = lin_fit(TWO_OVER_N, EG_TB)
    a_dft,   b_dft,   r2_dft   = lin_fit(TWO_OVER_N, EG_DFT12)
    a_exp,   b_exp,   r2_exp   = lin_fit(TWO_OVER_N, EG_EXP)

    print('Linear fits Eg = a + b · (2/n):')
    print(f'  TB Gen 140:   a = {a_tb:.3f}   b = {b_tb:.3f}   R² = {r2_tb:.4f}')
    print(f'  DFT-1/2:      a = {a_dft:.3f}   b = {b_dft:.3f}   R² = {r2_dft:.4f}')
    print(f'  Experiment:   a = {a_exp:.3f}   b = {b_exp:.3f}   R² = {r2_exp:.4f}')

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.3))

    # smooth fit lines from 2/n = 0 to slightly beyond n=1 (=2)
    x_fit = np.linspace(0.0, 2.1, 100)

    # ── 3D Pnma reference lines (3 of them) — labels at right edge ──────────
    # All labelled so it's clear what the n→∞ extrapolation should compare to.

    # 1) TB Gen 140 3D Pnma — bottom (1.46)
    ax.axhline(EG_3D_TB, color='#888', linestyle=':', lw=1.1, alpha=0.85, zorder=1)
    ax.text(2.10, EG_3D_TB - 0.05,
            f'3D Pnma TB = {EG_3D_TB:.2f} eV',
            fontsize=7.8, color='#444', ha='right', va='top')

    # 2) MAPbI3 Pnma exp (low T)  — same A-site as 2D RP slabs (1.65)
    ax.axhline(EG_3D_MAPBI3_PNMA_EXP, color='#b0701e', linestyle=(0, (3, 2, 1, 2)),
               lw=1.2, alpha=0.85, zorder=1)
    ax.text(2.10, EG_3D_MAPBI3_PNMA_EXP - 0.05,
            f'MAPbI$_3$ Pnma exp = {EG_3D_MAPBI3_PNMA_EXP:.2f} eV  '
            r'[Traoré 2022, ref 70]',
            fontsize=7.8, color='#7a4a10', ha='right', va='top')

    # 3) β-CsPbI3 Pnma exp — different A-site but similar tilting (1.72)
    ax.axhline(EG_3D_CSPBI3_PNMA_EXP, color='#108040', linestyle='-.', lw=1.3,
               alpha=0.9, zorder=1)
    ax.text(2.10, EG_3D_CSPBI3_PNMA_EXP + 0.02,
            f'$\\beta$-CsPbI$_3$ Pnma exp = {EG_3D_CSPBI3_PNMA_EXP:.2f} eV  '
            r'[Marronnier 2018]',
            fontsize=7.8, color='#085030', ha='right', va='bottom')

    # ── TB ────────────────────────────────────────────────────────────────────
    ax.plot(x_fit, a_tb + b_tb*x_fit,
            color='#d6342b', lw=1.4, alpha=0.65, zorder=2)
    ax.plot(TWO_OVER_N, EG_TB, 'o',
            ms=10, mfc='#d6342b', mec='black', mew=0.8,
            label='TB Gen 140 (this work)', zorder=4)

    # ── DFT-1/2 ───────────────────────────────────────────────────────────────
    ax.plot(x_fit, a_dft + b_dft*x_fit,
            color='#2c5fb8', lw=1.4, alpha=0.65, zorder=2)
    ax.plot(TWO_OVER_N, EG_DFT12, 's',
            ms=10, mfc='none', mec='#2c5fb8', mew=1.6,
            label='DFT-1/2 (Traoré 2022)', zorder=4)

    # ── Experiment ────────────────────────────────────────────────────────────
    ax.plot(x_fit, a_exp + b_exp*x_fit,
            color='black', lw=1.4, alpha=0.55, ls='--', zorder=2)
    ax.plot(TWO_OVER_N, EG_EXP, '^',
            ms=10, mfc='black', mec='black', mew=0.8,
            label='Experiment $E_g^\\mathrm{elec}$ (Blancon 2018)', zorder=4)

    # ── axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(-0.05, 2.15)
    ax.set_ylim(1.35, 3.25)
    ax.set_xlabel(r'$2/n$', fontsize=14)
    ax.set_ylabel(r'$E_g(\Gamma)$  (eV)', fontsize=14)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha=0.3)

    # secondary top x-axis showing n values
    n_ticks  = np.array([1, 2, 3, 4, 5, 10, 100])
    n_xpos   = 2.0 / n_ticks
    sec = ax.secondary_xaxis('top')
    sec.set_xticks(n_xpos)
    sec.set_xticklabels([f'{int(v)}' if v < 100 else r'$\infty$' for v in n_ticks])
    sec.set_xlabel('layer thickness  $n$', fontsize=13)
    sec.tick_params(axis='x', labelsize=10)
    # mark n→∞ explicitly at x=0
    ax.axvline(0.0, color='#bbb', lw=0.6, ls='-', alpha=0.5, zorder=0)

    # ── annotation box: fit parameters ────────────────────────────────────────
    txt = (
        r'$E_g(n) = E_g^\infty + \Delta E_\mathrm{surf} \cdot (2/n)$' + '\n'
        '\n'
        f'TB Gen 140:    $E_g^\\infty$ = {a_tb:.3f},  '
        f'$\\Delta E_\\mathrm{{surf}}$ = {b_tb:.3f},  $R^2$ = {r2_tb:.4f}\n'
        f'DFT-1/2:        $E_g^\\infty$ = {a_dft:.3f},  '
        f'$\\Delta E_\\mathrm{{surf}}$ = {b_dft:.3f},  $R^2$ = {r2_dft:.4f}\n'
        f'Experiment:   $E_g^\\infty$ = {a_exp:.3f},  '
        f'$\\Delta E_\\mathrm{{surf}}$ = {b_exp:.3f},  $R^2$ = {r2_exp:.4f}'
    )
    ax.text(0.02, 0.98, txt, transform=ax.transAxes,
            fontsize=9.0, va='top', ha='left',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.55',
                      fc='#fffce0', ec='#888', lw=0.6, alpha=0.55))

    # Legend at center-right, transparent so lines/markers underneath stay visible.
    ax.legend(loc='center right', fontsize=10.5, framealpha=0.55,
              edgecolor='#888', fancybox=True, bbox_to_anchor=(0.98, 0.55))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'fig_2n_scaling.{ext}')
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        print(f'  → {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
