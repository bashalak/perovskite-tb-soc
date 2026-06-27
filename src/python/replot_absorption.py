"""
replot_absorption.py — reload pre-computed .dat files and redraw figure
with DFT dielectric absorption overlaid.

Run from src/python/:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 replot_absorption.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE   = os.path.dirname(os.path.abspath(__file__))
DAT     = os.path.join(_HERE, '..', '..', 'results', 'data')
FIG     = os.path.join(_HERE, '..', '..', 'results', 'figures')

OMEGA_MIN, OMEGA_MAX = 1.4, 4.0

# ── Load TB absorption ────────────────────────────────────────────────────────
e3d, a3d = np.loadtxt(f'{DAT}/absorption_3D_bulk.dat', unpack=True)
results  = {n: np.loadtxt(f'{DAT}/absorption_2D_n{n}.dat', unpack=True)
            for n in range(1, 6)}

# ── DFT dielectric → α = 2E·κ / ℏc,  κ = Im(sqrt(ε_r + i ε_i)) ─────────────
E_i, eps_i = np.loadtxt(f'{DAT}/dft_epsi_george.dat', unpack=True)
E_r, eps_r = np.loadtxt(f'{DAT}/dft_epsr_george.dat', unpack=True)

E_dft  = np.linspace(max(E_i.min(), E_r.min()), min(E_i.max(), E_r.max()), 800)
eps_i  = np.maximum(np.interp(E_dft, E_i, eps_i), 0.0)
eps_r  = np.interp(E_dft, E_r, eps_r)
kappa  = np.sqrt(np.maximum((np.sqrt(eps_r**2 + eps_i**2) - eps_r) / 2.0, 0.0))
a_dft  = (2.0 * E_dft / 1.9732e-5) * kappa          # cm-1

mask   = (E_dft >= OMEGA_MIN) & (E_dft <= OMEGA_MAX)
a_dft  = a_dft * a3d[(e3d >= OMEGA_MIN) & (e3d <= OMEGA_MAX)].max() / a_dft[mask].max()

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(e3d, a3d, 'k--', lw=2.0, label='3D bulk (TB)', zorder=10)
try:
    d = np.loadtxt(f'{DAT}/simon1.txt')
    ax.plot(d[:, 0], d[:, 1], 'r--', lw=1.5, label='Fortran Soline model', alpha=0.8)
except FileNotFoundError:
    pass
ax.plot(E_dft[mask], a_dft[mask], color='darkorange', lw=2.0, ls='-.',
        label='DFT  α=2Eκ/ℏc (George)', zorder=9)
for i, n in enumerate(range(1, 6)):
    e, a = results[n]
    ax.plot(e, a, color=COLORS[i], lw=1.5, label=f'2D n={n}')

ax.set_xlabel('Photon energy (eV)', fontsize=13)
ax.set_ylabel('α (cm⁻¹)', fontsize=13)
ax.set_title('Optical absorption: 3D bulk and 2D slabs (SP³ TB+SOC) + DFT', fontsize=11)
ax.set_xlim(OMEGA_MIN, OMEGA_MAX)
ax.set_yscale('log')
ax.set_ylim(bottom=1)
ax.legend(fontsize=10, framealpha=0.85)
ax.grid(True, alpha=0.25)
fig.tight_layout()

os.makedirs(FIG, exist_ok=True)
out = f'{FIG}/absorption_3D_and_2D.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
