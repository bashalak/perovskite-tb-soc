"""
plot_exciton_chain.py — Figure 5 of Paper 1: exciton binding validation chain.

Three-panel figure showing the chain of predictions:

  Panel A — Exciton binding energy E_b(n) — TB+R-K vs Blancon vs Cho 2019 BSE
  Panel B — Reduced effective mass μ(n)   — TB bare, TB+Fröhlich vs Blancon
  Panel C — Optical gap E_opt(n)          — E_TB - E_b vs Blancon optical

Headline numbers (in annotations):
  • E_b RMS error vs Blancon
  • Polaron enhancement factor ⟨μ_pol/μ_bare⟩
  • E_opt RMS error vs Blancon

Output: report/figures/fig_exciton_chain.{png,pdf}

Run:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 plot_exciton_chain.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')

# Load Phase 7A results
data_path = os.path.join(_ROOT, 'results', 'data', 'phase7a_excitons.npz')
if not os.path.exists(data_path):
    raise FileNotFoundError(f'Run phase7a_excitons.py first to generate {data_path}')
d = np.load(data_path)

# Cho & Berkelbach 2019 BSE reference (from rytova_keldysh.py)
CHO_E_B_meV = {1: 302, 5: 97}

# Load our own BSE result (Option C, if available)
bse_path = os.path.join(_ROOT, 'results', 'data', 'phase7a_bse.npz')
HAS_BSE = os.path.exists(bse_path)
if HAS_BSE:
    bse_data = np.load(bse_path)
    E_b_BSE = bse_data['E_b_BSE_meV']
else:
    E_b_BSE = None

DPI = 220


def main():
    n_vals      = d['n_vals']
    E_b_pred    = d['E_b_RK_meV']
    E_b_exp     = d['E_b_exp_meV']
    mu_TB       = d['mu_TB']
    mu_pol_pred = d['mu_polaron_pred_1st']
    mu_pol_2nd  = d['mu_polaron_pred_2nd']
    mu_pol_exp  = d['mu_polaron_exp']
    eg_TB       = d['eg_TB']
    E_opt_pred  = d['E_opt_pred']
    E_opt_exp   = d['E_opt_exp']
    alpha       = d['alpha_Frohlich']

    # Cho BSE points (only n=1 and n=5)
    cho_n = np.array(list(CHO_E_B_meV.keys()))
    cho_E = np.array([CHO_E_B_meV[int(k)] for k in cho_n])

    # Statistics
    rms_eb  = np.sqrt(np.mean((E_b_pred - E_b_exp)**2))
    rms_mu  = np.sqrt(np.mean((mu_pol_pred - mu_pol_exp)**2))
    rms_opt = np.sqrt(np.mean((E_opt_pred - E_opt_exp)**2))
    enh_ratio = np.mean(mu_pol_pred / mu_TB)
    enh_ratio_exp = np.mean(mu_pol_exp / mu_TB)

    # ── figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

    # ── Panel A: Exciton binding energy ────────────────────────────────────
    ax = axes[0]
    x = n_vals.astype(float)
    if HAS_BSE:
        w = 0.20
        ax.bar(x - 1.5*w, E_b_BSE, w,
               label='TB Gen 140 + BSE  (this work)',
               color='#8172b3', edgecolor='black', lw=0.5)
        ax.bar(x - 0.5*w, E_b_pred, w,
               label='TB Gen 140 + R-K  (this work)',
               color='#4c72b0', edgecolor='black', lw=0.5)
        ax.bar(x + 0.5*w, E_b_exp, w,
               label='Blancon 2018 (60 T)',
               color='#55a868', edgecolor='black', lw=0.5)
        # Cho BSE reference (markers)
        ax.scatter(cho_n + 1.5*w, cho_E, marker='D', s=80,
                   color='#c44e52', edgecolors='black', linewidths=0.8,
                   zorder=5, label='Cho-Berkelbach 2019 BSE')
    else:
        w = 0.27
        ax.bar(x - w/2, E_b_pred, w,
               label='TB Gen 140 + R-K  (this work)',
               color='#4c72b0', edgecolor='black', lw=0.5)
        ax.bar(x + w/2, E_b_exp, w,
               label='Blancon 2018 (60 T)',
               color='#55a868', edgecolor='black', lw=0.5)
        ax.scatter(cho_n, cho_E, marker='D', s=80,
                   color='#c44e52', edgecolors='black', linewidths=0.8,
                   zorder=5, label='Cho-Berkelbach 2019 BSE')

    ax.set_xlabel('layer number  $n$', fontsize=13)
    ax.set_ylabel(r'exciton binding  $E_b$  (meV)', fontsize=13)
    ax.set_title('(A)  Exciton binding energy', fontsize=12)
    ax.set_xticks(x)
    ax.set_ylim(0, max(E_b_exp.max(), E_b_pred.max()) * 1.15)
    ax.legend(fontsize=9.5, framealpha=0.6, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    # annotate RMS error — placed in lower-left to avoid overlap with the legend
    ax.text(0.04, 0.04, f'RMS error = {rms_eb:.0f} meV',
            transform=ax.transAxes, fontsize=10, va='bottom', ha='left',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      fc='#fffce0', ec='#888', lw=0.6, alpha=0.6))

    # ── Panel B: μ_2D — bare vs polaronic ──────────────────────────────────
    ax = axes[1]
    ax.plot(n_vals, mu_TB, 'o-', ms=10,
            mfc='#d6342b', mec='black', mew=0.8,
            color='#d6342b', lw=1.6,
            label='TB bare-band $\\mu$ (this work)')
    ax.plot(n_vals, mu_pol_pred, 's-', ms=10,
            mfc='none', mec='#d6342b', mew=1.6,
            color='#d6342b', lw=1.6, ls='--',
            label=r'predicted polaronic $\mu$ ($1+\alpha/6$)')
    ax.plot(n_vals, mu_pol_exp, '^-', ms=10,
            mfc='black', mec='black', mew=0.8,
            color='black', lw=1.6, ls=':',
            label=r'experiment $\mu_\mathrm{polaron}$ (Blancon)')

    ax.set_xlabel('layer number  $n$', fontsize=13)
    ax.set_ylabel(r'reduced mass  $\mu$  ($m_e$)', fontsize=13)
    ax.set_title('(B)  Mass: bare → polaronic → experiment', fontsize=12)
    ax.set_xticks(n_vals)
    ax.set_ylim(0.10, 0.27)
    ax.legend(fontsize=9.5, framealpha=0.6, loc='upper right')
    ax.grid(True, alpha=0.3)

    # stat box placed in lower-left to avoid overlap with the legend
    ax.text(0.04, 0.04,
            f'⟨α⟩ = {alpha.mean():.2f}\n'
            f'⟨μ_pol/μ_bare⟩ pred = {enh_ratio:.2f}\n'
            f'⟨μ_pol/μ_bare⟩ exp  = {enh_ratio_exp:.2f}',
            transform=ax.transAxes, fontsize=9, va='bottom', ha='left',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      fc='#fffce0', ec='#888', lw=0.6, alpha=0.6))

    # ── Panel C: Optical gap ───────────────────────────────────────────────
    ax = axes[2]
    ax.plot(n_vals, eg_TB, 'o:', ms=8,
            mfc='#aaaaaa', mec='black', mew=0.6,
            color='#777777', lw=1.2,
            label=r'$E_g^\mathrm{TB}$ (QP gap, no exciton)')
    ax.plot(n_vals, E_opt_pred, 's-', ms=10,
            mfc='#4c72b0', mec='black', mew=0.8,
            color='#4c72b0', lw=1.6,
            label=r'$E_\mathrm{opt}^\mathrm{pred}=E_g^\mathrm{TB}-E_b$')
    ax.plot(n_vals, E_opt_exp, '^-', ms=10,
            mfc='black', mec='black', mew=0.8,
            color='black', lw=1.6, ls='--',
            label=r'experiment $E_\mathrm{opt}$ (Blancon)')

    # arrows showing the E_b shift on a representative point (n=1)
    ax.annotate('', xy=(n_vals[0], E_opt_pred[0]),
                xytext=(n_vals[0], eg_TB[0]),
                arrowprops=dict(arrowstyle='->', color='#4c72b0', lw=1.8))
    ax.text(n_vals[0]+0.08, (eg_TB[0]+E_opt_pred[0])/2,
            f'$-E_b$\n({E_b_pred[0]:.0f} meV)',
            fontsize=9, color='#2c5fb8', va='center')

    ax.set_xlabel('layer number  $n$', fontsize=13)
    ax.set_ylabel('energy  (eV)', fontsize=13)
    ax.set_title('(C)  Quasiparticle gap → Optical gap', fontsize=12)
    ax.set_xticks(n_vals)
    ax.set_ylim(1.7, 3.05)
    ax.legend(fontsize=9.5, framealpha=0.6, loc='upper right')
    ax.grid(True, alpha=0.3)

    # stat box placed in lower-left to avoid overlap with the legend
    ax.text(0.04, 0.04,
            f'RMS error\n'
            f'$E_\\mathrm{{opt}}$ = {rms_opt:.3f} eV',
            transform=ax.transAxes, fontsize=10, va='bottom', ha='left',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      fc='#fffce0', ec='#888', lw=0.6, alpha=0.6))

    # Super-title
    fig.suptitle(r'TB Gen 140 → R-K → polaron correction → Blancon 2018 experimental observables',
                 fontsize=13, y=1.005)

    plt.tight_layout()
    out_dir = os.path.join(_ROOT, 'report', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'fig_exciton_chain.{ext}')
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        print(f'  → {out}')
    plt.close(fig)

    print(f'\nSummary statistics:')
    print(f'  E_b RMS error      = {rms_eb:.1f} meV')
    print(f'  μ_polaron RMS error = {rms_mu:.4f} m_e')
    print(f'  E_opt RMS error    = {rms_opt:.3f} eV')
    print(f'  ⟨α⟩ Fröhlich        = {alpha.mean():.3f}')
    print(f'  ⟨μ_pol/μ_bare⟩ pred = {enh_ratio:.3f}')
    print(f'  ⟨μ_pol/μ_bare⟩ exp  = {enh_ratio_exp:.3f}')


if __name__ == '__main__':
    main()
