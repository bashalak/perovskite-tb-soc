"""
plot_cspbi3_bands.py — Band-structure plot of γ-CsPbI3 Pnma with Gen 140.

Test A of Phase 7B. Uses the same Pnma k-path as our MAPbI3 figure:
Γ–X–S–Y–Γ–Z–U–R–T–Z. Compares to literature reference points (DFT-1/2,
DFTB, GW, exp) as horizontal markers at Γ.

Output: report/figures/fig_cspbi3_bands.{png,pdf}
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
sys.path.insert(0, _HERE)

import ham_ortho_harrison as ham
from ase.io import read as ase_read
from ase.neighborlist import neighbor_list as ase_neighbor_list

# ── Gen 140 (canonical) ───────────────────────────────────────────────────────
GEN140 = dict(zip(ham.PARAM_KEYS, [
    -2.79, +3.29, -10.37, -4.06,
    -0.67, -1.20, -0.22, -1.75, -1.53,
    +1.30, +0.90,
    -0.76, -0.22, -0.14, +0.10,
    -0.05, +0.33, +0.19, -0.49,
    0.0, 0.0, 0.0, 0.0,
    -9.38, -2.49, +3.18,
    +2.36,
]))

LABEL = '3D_cspbi3'
# b-long Pnma convention (b ↔ c permuted from MP-1120768) so the band-structure
# plot is directly comparable to MAPbI3 (Whitfield b-long setting). Physical
# crystal is identical; only axis labels change.
POSCAR_PATH = os.path.join(_ROOT, 'data',
                           'CsPbI3_pnma_mp1120768_blong.vasp')
JSON_PATH = os.path.join(_ROOT, 'results', 'data', f'geometry_{LABEL}.json')


# ── Patch ham_ortho_harrison to use our POSCAR ────────────────────────────────
def _patched_finder(geom, cutoff_PbPb=7.5, cutoff_II=6.5):
    key = (geom['label'], round(cutoff_PbPb, 3), round(cutoff_II, 3))
    if key in ham._BOND_CACHE:
        return ham._BOND_CACHE[key]
    v = geom.get('vasp_path')
    if v and not os.path.isabs(v):
        v = os.path.join(_ROOT, v)
    if v is None:
        v = POSCAR_PATH
    atoms = ase_read(v)
    pb_set = {a.index for a in atoms if a.symbol == 'Pb'}
    i_set  = {a.index for a in atoms if a.symbol == 'I'}
    ia, ja, Darr = ase_neighbor_list('ijD', atoms,
                                      max(cutoff_PbPb, cutoff_II))
    pb_pb_bonds = [(int(i), int(j), D) for i, j, D in zip(ia, ja, Darr)
                   if i in pb_set and j in pb_set and i < j
                   and np.linalg.norm(D) < cutoff_PbPb]
    ii_bonds = [(int(i), int(j), D) for i, j, D in zip(ia, ja, Darr)
                if i in i_set and j in i_set and i < j
                and np.linalg.norm(D) < cutoff_II]
    ham._BOND_CACHE[key] = (pb_pb_bonds, ii_bonds)
    return pb_pb_bonds, ii_bonds


ham._find_second_neighbour_bonds = _patched_finder


# ── Reciprocal lattice ────────────────────────────────────────────────────────

def recip_cell(label):
    geom = ham.load_geometry(label)
    cell = np.array(geom['lattice'])
    V = np.dot(cell[0], np.cross(cell[1], cell[2]))
    b1 = 2 * np.pi * np.cross(cell[1], cell[2]) / V
    b2 = 2 * np.pi * np.cross(cell[2], cell[0]) / V
    b3 = 2 * np.pi * np.cross(cell[0], cell[1]) / V
    return np.array([b1, b2, b3])


def build_path(pts, nk):
    kpts, kd, ticks = [], [], [0.0]
    cum = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        dk = b - a
        L = np.linalg.norm(dk)
        for t in np.linspace(0, 1, nk, endpoint=False):
            kpts.append(a + t * dk)
            kd.append(cum + t * L)
        cum += L
        ticks.append(cum)
    kpts.append(pts[-1])
    kd.append(cum)
    return np.array(kpts), np.array(kd), ticks


def kpath_3D(B, nk=60):
    def f(c):
        return c[0] * B[0] + c[1] * B[1] + c[2] * B[2]
    hsp = {'G': [0, 0, 0], 'X': [.5, 0, 0], 'S': [.5, .5, 0],
           'Y': [0, .5, 0], 'Z': [0, 0, .5], 'U': [.5, 0, .5],
           'R': [.5, .5, .5], 'T': [0, .5, .5]}
    seq = ['G', 'X', 'S', 'Y', 'G', 'Z', 'U', 'R', 'T', 'Z']
    lab = [r'$\Gamma$', 'X', 'S', 'Y', r'$\Gamma$', 'Z', 'U', 'R', 'T', 'Z']
    kp, kd, tc = build_path([f(hsp[p]) for p in seq], nk)
    return kp, kd, tc, lab


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Make sure JSON exists
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(
            f"Run phase7b_testA_cspbi3.py first to generate {JSON_PATH}")

    geom = ham.load_geometry(LABEL)
    nf = 2 * geom['n_Pb'] + 8 * geom['n_I']
    print(f"CsPbI3 Pnma: n_Pb={geom['n_Pb']}, n_I={geom['n_I']}, n_fill={nf}")

    # Build k-path
    B = recip_cell(LABEL)
    kp, kd, ticks, tlabs = kpath_3D(B, nk=60)
    print(f"k-path: {len(kp)} points, length {kd[-1]:.3f} Å⁻¹")

    # Compute bands
    print("Computing bands ...", flush=True)
    # Pass '3D' (not LABEL='3D_cspbi3') so ham uses the 3D-bulk branch and
    # IIz parameters are NOT applied to z-axis I-I bonds. LABEL is kept on
    # geom['label'] for cache differentiation, not code-path selection.
    bands = ham.eigenvalues_batch('3D', kp, GEN140, geom=geom)
    # Shift VBM(Γ) to zero
    gi = int(np.argmin(np.linalg.norm(kp, axis=1)))
    vbm = bands[gi, nf - 1]
    bands -= vbm
    Eg_G = float(bands[gi, nf] - bands[gi, nf - 1])
    print(f"Eg(Γ) = {Eg_G:.4f} eV (with VBM shifted to 0)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # All bands
    for b in range(bands.shape[1]):
        ax.plot(kd, bands[:, b], color='#3060c0', lw=0.7, alpha=0.55, zorder=2)

    # VBM (red) and CBM (green) emphasized
    ax.plot(kd, bands[:, nf - 1], 'r-', lw=2.0, label='VBM', zorder=4)
    ax.plot(kd, bands[:, nf],     'g-', lw=2.0, label='CBM', zorder=4)

    # Reference markers at the MIDDLE Γ (4th tick, index 4 in the path
    # G-X-S-Y-G-Z-U-R-T-Z). Placing them here avoids cropping at the edge
    # and makes the comparison sit in the middle of the figure.
    # 3D Gen 140 was fit primarily to Jiang ss-GW — that's the natural
    # comparison. Marronnier (exp) and Jiang DFTB are independent cross-checks.
    # Traoré DFT-1/2 (2.79) is omitted: it's a known overshoot for Cs phases,
    # and was NOT a fit target for the 3D bulk part of the model.
    refs = {
        'Jiang ss-GW (1.57)  ← fit level':  1.57,
        'Jiang DFTB (1.70)':                1.70,
        'Marronnier exp 90K (1.72)':        1.72,
    }
    colors = {'Jiang ss-GW (1.57)  ← fit level':  '#8172b3',
              'Jiang DFTB (1.70)':                '#108040',
              'Marronnier exp 90K (1.72)':        'black'}
    markers = {'Jiang ss-GW (1.57)  ← fit level':  's',
               'Jiang DFTB (1.70)':                'D',
               'Marronnier exp 90K (1.72)':        '^'}
    k_G_mid = ticks[4]   # the second Γ in the path
    # Place all markers ON Γ (their y-values differ enough not to overlap).
    for lbl, val in refs.items():
        ax.plot(k_G_mid, val, markers[lbl], ms=11,
                mfc=colors[lbl], mec='black', mew=0.8,
                label=lbl, zorder=6)

    # Axes
    ax.axhline(0, color='red', lw=0.7, ls='--', alpha=0.7)
    for tp in ticks[1:-1]:
        ax.axvline(tp, color='k', lw=0.5, alpha=0.4)
    ax.set_xlim(kd[0], kd[-1])
    ax.set_ylim(-3.0, 4.5)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabs, fontsize=12)
    ax.set_ylabel(r'$E - E_\mathrm{VBM}$ (eV)', fontsize=12)
    ax.set_title(rf'$\gamma$-CsPbI$_3$ Pnma — TB Gen 140 (zero refit) '
                 f'  Eg(Γ) = {Eg_G:.3f} eV', fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis='y', labelsize=10)

    # Annotation — anchor on ss-GW (the fit level of theory for 3D bulk)
    txt = (f'A-cation transferability test (Phase 7B Test A)\n'
           f'All 21 Gen 140 parameters frozen.\n'
           f'Only geometry changes (MA → Cs).\n\n'
           f'TB Gen 140:        {Eg_G:.3f} eV\n'
           f'Jiang ss-GW:       1.57 eV  (Δ = {Eg_G - 1.57:+.3f})  ← fit level\n'
           f'Exp (Marronnier):  1.72 eV  (Δ = {Eg_G - 1.72:+.3f})')
    ax.text(0.02, 0.97, txt, transform=ax.transAxes,
            fontsize=9, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', fc='#fffce0',
                      ec='#888', lw=0.6, alpha=0.7))

    ax.legend(loc='upper right', fontsize=9, framealpha=0.6,
              edgecolor='#888', fancybox=True)

    plt.tight_layout()
    out_dir = os.path.join(_ROOT, 'report', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'fig_cspbi3_bands.{ext}')
        fig.savefig(out, dpi=220, bbox_inches='tight')
        print(f'  → {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
