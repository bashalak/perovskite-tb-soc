"""
plot_paper_figures.py — Publication-quality individual band-structure figures.

Generates **six standalone figures** (3D bulk + 2D slabs n=1..5) for the paper,
each annotated with:
  • Eg at Γ
  • Direction-resolved effective masses (3D: mv/mc per axis; 2D: μ along both
    in-plane directions plus average)
  • For the 3D panel: ss-GW reference bands (Jiang priv. comm.) overlaid

Reuses the helpers from plot_bands_interactive_harrison.py (k-path construction,
GW loader, finite-difference mass extraction) but produces non-interactive,
high-resolution PNG + PDF output.

Output: report/figures/
  fig_bands_3D.{png,pdf}
  fig_bands_n1.{png,pdf}
  fig_bands_n2.{png,pdf}
  fig_bands_n3.{png,pdf}
  fig_bands_n4.{png,pdf}
  fig_bands_n5.{png,pdf}

Parameters used: **Gen 140** (canonical, physically clean, n_harrison=2.36).

Run:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 plot_paper_figures.py [--show]
"""
import os, sys, argparse, time

import numpy as np
import matplotlib
matplotlib.use('Agg')          # default: file-only output
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
sys.path.insert(0, _HERE)

import ham_ortho_harrison as ham_ortho

# ── Gen 140 parameters (canonical) ────────────────────────────────────────────
GEN140 = dict(zip(ham_ortho.PARAM_KEYS, [
    # on-site
    -2.79, +3.29, -10.37, -4.06,
    # Pb-I 1st neighbour SK (Harrison-scaled)
    -0.67, -1.20, -0.22, -1.75, -1.53,
    # SOC (fixed, Boyer-Richard 2016)
    +1.30, +0.90,
    # I-I in-plane
    -0.76, -0.22, -0.14, +0.10,
    # I-I apical
    -0.05, +0.33, +0.19, -0.49,
    # Pb-Pb (frozen)
    0.0, 0.0, 0.0, 0.0,
    # surface corrections
    -9.38, -2.49, +3.18,
    # Harrison exponent
    +2.36,
]))

LABELS    = ['3D', 'n1', 'n2', 'n3', 'n4', 'n5']
HBAR2_M   = 7.6200              # ℏ²/m_e [eV·Å²]  →  m*/m_e = HBAR2_M / (d²E/dk²)
DK_MASS   = 0.05                # Å⁻¹ finite-difference step
NK_3D     = 60                  # k-points per segment, 3D
NK_2D     = 80                  # k-points per segment, 2D
DPI       = 220                 # for PNG output


# ── geometry / k-path helpers (mirrored from plot_bands_interactive_harrison) ──

def _vasp_path_for(label):
    if label == '3D':
        return os.path.join(_ROOT, 'data', 'pos3D2Dtilted', 'POSCAR')
    return os.path.join(_ROOT, 'data', 'pos3D2Dpnma_xyz',
                        f'pnma_xyz_{label}.vasp')


def _recip_cell(label):
    from ase.io import read
    cell = read(_vasp_path_for(label)).cell.array
    V  = np.dot(cell[0], np.cross(cell[1], cell[2]))
    b1 = 2*np.pi * np.cross(cell[1], cell[2]) / V
    b2 = 2*np.pi * np.cross(cell[2], cell[0]) / V
    b3 = 2*np.pi * np.cross(cell[0], cell[1]) / V
    return np.array([b1, b2, b3])


def _build_path(pts, nk):
    kpts, kd, ticks = [], [], [0.0]
    cum = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        dk = b - a; L = np.linalg.norm(dk)
        for t in np.linspace(0, 1, nk, endpoint=False):
            kpts.append(a + t*dk); kd.append(cum + t*L)
        cum += L; ticks.append(cum)
    kpts.append(pts[-1]); kd.append(cum)
    return np.array(kpts), np.array(kd), ticks


def _kpath_3D(B, nk=NK_3D):
    def f(c): return c[0]*B[0] + c[1]*B[1] + c[2]*B[2]
    hsp = {'G':[0,0,0], 'X':[.5,0,0], 'S':[.5,.5,0], 'Y':[0,.5,0],
           'Z':[0,0,.5], 'U':[.5,0,.5], 'R':[.5,.5,.5], 'T':[0,.5,.5]}
    seq = ['G','X','S','Y','G','Z','U','R','T','Z']
    lab = [r'$\Gamma$','X','S','Y',r'$\Gamma$','Z','U','R','T','Z']
    kp, kd, tc = _build_path([f(hsp[p]) for p in seq], nk)
    return kp, kd, tc, lab


def _kpath_2D(label, geom, nk=NK_2D):
    from ase.io import read
    cell = read(_vasp_path_for(label)).cell.array
    b1_2d, b2_2d = ham_ortho.in_plane_reciprocal_vectors(cell, geom['slab_normal'])
    X = 0.5*b1_2d; Y = 0.5*b2_2d; S = X + Y; G = np.zeros(3)
    kp, kd, tc = _build_path([Y, G, X, S], nk)
    return kp, kd, tc, ['Y', r'$\Gamma$', 'X', 'S']


def _n_fill(geom):
    return geom['n_Pb'] * 2 + geom['n_I'] * 8


def _load_gw(ticks_3D):
    """Load Jiang's ss-GW Pnma bands and align k-axis to TB's 3D path length."""
    path = os.path.join(_ROOT, 'data', 'ss-gw.band.dat')
    if not os.path.exists(path):
        return [], None
    bands, kc, ec = [], [], []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                if kc:
                    bands.append((np.array(kc), np.array(ec)))
                    kc = []; ec = []
            else:
                v = s.split()
                kc.append(float(v[0])); ec.append(float(v[1]))
    if kc:
        bands.append((np.array(kc), np.array(ec)))
    eg    = np.sort([b[1][0] for b in bands])
    gi    = int(np.argmax(np.diff(eg)))
    vbm   = eg[gi]
    scale = ticks_3D[-1] / bands[0][0][-1]
    lines = [(b[0]*scale, b[1]-vbm) for b in bands
             if np.any(np.abs(b[1]-vbm) < 4.5)]
    print(f'  ss-GW loaded: VBM={vbm:.3f}, {len(lines)} bands within window')
    return lines, vbm


# ── effective-mass k batches ──────────────────────────────────────────────────

def _mass_kpts_3D(B):
    ex = B[0] / np.linalg.norm(B[0])
    ey = B[1] / np.linalg.norm(B[1])
    ez = B[2] / np.linalg.norm(B[2])
    G  = np.zeros(3)
    return np.array([
        G,
        G + DK_MASS*ex, G - DK_MASS*ex,
        G + DK_MASS*ey, G - DK_MASS*ey,
        G + DK_MASS*ez, G - DK_MASS*ez,
    ])


def _mass_kpts_2D(label, geom):
    from ase.io import read
    cell = read(_vasp_path_for(label)).cell.array
    b1_2d, b2_2d = ham_ortho.in_plane_reciprocal_vectors(cell, geom['slab_normal'])
    e1 = b1_2d / np.linalg.norm(b1_2d)
    e2 = b2_2d / np.linalg.norm(b2_2d)
    G  = np.zeros(3)
    return np.array([
        G,
        G + DK_MASS*e1, G - DK_MASS*e1,
        G + DK_MASS*e2, G - DK_MASS*e2,
    ])


def _masses_3D(bands_mass, nf):
    """Returns (mv_xx, mv_yy, mv_zz), (mc_xx, mc_yy, mc_zz) with signs."""
    ivb, icb = nf - 1, nf
    e0v = bands_mass[0, ivb]
    e0c = bands_mass[0, icb]
    d2v = [(bands_mass[1, ivb] + bands_mass[2, ivb] - 2*e0v) / DK_MASS**2,
           (bands_mass[3, ivb] + bands_mass[4, ivb] - 2*e0v) / DK_MASS**2,
           (bands_mass[5, ivb] + bands_mass[6, ivb] - 2*e0v) / DK_MASS**2]
    d2c = [(bands_mass[1, icb] + bands_mass[2, icb] - 2*e0c) / DK_MASS**2,
           (bands_mass[3, icb] + bands_mass[4, icb] - 2*e0c) / DK_MASS**2,
           (bands_mass[5, icb] + bands_mass[6, icb] - 2*e0c) / DK_MASS**2]
    mv = [HBAR2_M / d for d in d2v]
    mc = [HBAR2_M / d for d in d2c]
    return mv, mc


def _masses_2D(bands_mass, nf):
    """Returns (μ_dir1, μ_dir2, μ_avg, mh1, mh2, me1, me2)."""
    ivb, icb = nf - 1, nf
    e0v = bands_mass[0, ivb]
    e0c = bands_mass[0, icb]
    d2v1 = (bands_mass[1, ivb] + bands_mass[2, ivb] - 2*e0v) / DK_MASS**2
    d2v2 = (bands_mass[3, ivb] + bands_mass[4, ivb] - 2*e0v) / DK_MASS**2
    d2c1 = (bands_mass[1, icb] + bands_mass[2, icb] - 2*e0c) / DK_MASS**2
    d2c2 = (bands_mass[3, icb] + bands_mass[4, icb] - 2*e0c) / DK_MASS**2
    mh1 = abs(HBAR2_M / d2v1); mh2 = abs(HBAR2_M / d2v2)
    me1 = abs(HBAR2_M / d2c1); me2 = abs(HBAR2_M / d2c2)
    mu1 = mh1*me1 / (mh1 + me1)
    mu2 = mh2*me2 / (mh2 + me2)
    return mu1, mu2, 0.5*(mu1 + mu2), mh1, mh2, me1, me2


# ── plotting ──────────────────────────────────────────────────────────────────

def _format_ax(ax, kd, ticks, tlabs, ymin=-3.0, ymax=4.5):
    ax.axhline(0,            color='red', lw=0.7, ls='--', alpha=0.7)
    for tp in ticks[1:-1]:
        ax.axvline(tp,       color='k',   lw=0.5, alpha=0.4)
    ax.set_xlim(kd[0], kd[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tlabs, fontsize=12)
    ax.set_ylabel(r'$E - E_\mathrm{VBM}$ (eV)', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, alpha=0.25)


def _plot_3D(out_dir, kp, kd, ticks, tlabs, bands, nf, gw_lines):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # ss-GW overlay (drawn first, behind TB)
    if gw_lines:
        for i, (k_arr, e_arr) in enumerate(gw_lines):
            lbl = 'ss-GW (Jiang)' if i == 0 else None
            ax.plot(k_arr, e_arr, color='#808080', lw=0.8, alpha=0.6,
                    label=lbl, zorder=1)

    # all TB bands (faint blue)
    for b in range(bands.shape[1]):
        ax.plot(kd, bands[:, b], color='#3060c0', lw=0.7,
                alpha=0.55, zorder=2)

    # VBM (red) / CBM (green) emphasised
    ax.plot(kd, bands[:, nf-1], 'r-', lw=2.0, label='TB VBM', zorder=4)
    ax.plot(kd, bands[:, nf],   'g-', lw=2.0, label='TB CBM', zorder=4)

    # gap at Γ
    gi    = int(np.argmin(np.linalg.norm(kp, axis=1)))
    gap_g = float(bands[gi, nf] - bands[gi, nf-1])

    # 3D effective masses (computed in main)
    mv, mc = _3D_MASSES
    title  = f'3D Pnma MAPbI$_3$  —  $E_g(\\Gamma)$ = {gap_g:.3f} eV'
    ax.set_title(title, fontsize=13)
    _format_ax(ax, kd, ticks, tlabs)

    # annotation box: masses
    box  = (f'$m_v$ [xx,yy,zz] = [{mv[0]:+.3f}, {mv[1]:+.3f}, {mv[2]:+.3f}]\n'
            f'$m_c$ [xx,yy,zz] = [{mc[0]:+.3f}, {mc[1]:+.3f}, {mc[2]:+.3f}]\n'
            r'($m_v^\mathrm{target}=-0.32$,  $m_c^\mathrm{target}=+0.28$)')
    ax.text(0.02, 0.02, box, transform=ax.transAxes,
            fontsize=9, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.5',
                      fc='#fffce0', ec='#888', lw=0.6, alpha=0.9))

    ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
              edgecolor='#888', fancybox=True)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'fig_bands_3D.{ext}')
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        print(f'  → {out}')
    plt.close(fig)


def _plot_2D(out_dir, lbl, kp, kd, ticks, tlabs, bands, nf,
             mu1, mu2, mu_avg, mh1, mh2, me1, me2):
    fig, ax = plt.subplots(figsize=(6.5, 4.6))

    for b in range(bands.shape[1]):
        ax.plot(kd, bands[:, b], color='#3060c0', lw=0.7,
                alpha=0.55, zorder=2)

    ax.plot(kd, bands[:, nf-1], 'r-', lw=2.0, label='VBM', zorder=4)
    ax.plot(kd, bands[:, nf],   'g-', lw=2.0, label='CBM', zorder=4)

    gi    = int(np.argmin(np.linalg.norm(kp, axis=1)))
    gap_g = float(bands[gi, nf] - bands[gi, nf-1])

    n_val = lbl.replace('n', '')
    title = (rf'2D (BA)$_2$(MA)$_{{n-1}}$Pb$_n$I$_{{3n+1}}$,  $n={n_val}$  '
             f'—  $E_g(\\Gamma)$ = {gap_g:.3f} eV')
    ax.set_title(title, fontsize=12)
    _format_ax(ax, kd, ticks, tlabs)

    box = (f'$\\mu_1$ = {mu1:.3f},  $\\mu_2$ = {mu2:.3f}\n'
           f'$\\mu_\\mathrm{{avg}}$ = {mu_avg:.3f}\n'
           f'$m_h$ = ({mh1:.3f}, {mh2:.3f})\n'
           f'$m_e$ = ({me1:.3f}, {me2:.3f})')
    ax.text(0.02, 0.02, box, transform=ax.transAxes,
            fontsize=9, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.5',
                      fc='#e0f5ff', ec='#888', lw=0.6, alpha=0.9))

    ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
              edgecolor='#888', fancybox=True)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        out = os.path.join(out_dir, f'fig_bands_{lbl}.{ext}')
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        print(f'  → {out}')
    plt.close(fig)


# ── main driver ───────────────────────────────────────────────────────────────

_3D_MASSES = None     # populated in main(); used by _plot_3D

def main():
    global _3D_MASSES
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true',
                        help='also display windows (requires Tk backend)')
    args = parser.parse_args()

    if args.show:
        matplotlib.use('TkAgg', force=True)

    out_dir = os.path.join(_ROOT, 'report', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output directory: {out_dir}')

    print('\nLoading geometries…')
    geoms = {lbl: ham_ortho.load_geometry(lbl) for lbl in LABELS}
    nf    = {lbl: _n_fill(geoms[lbl])           for lbl in LABELS}

    print('\nBuilding k-paths…')
    paths = {}
    for lbl in LABELS:
        B = _recip_cell(lbl)
        if lbl == '3D':
            kp, kd, tc, tl = _kpath_3D(B)
            kmass = _mass_kpts_3D(B)
        else:
            kp, kd, tc, tl = _kpath_2D(lbl, geoms[lbl])
            kmass = _mass_kpts_2D(lbl, geoms[lbl])
        paths[lbl] = dict(kp=kp, kd=kd, ticks=tc, tlabs=tl, kmass=kmass)
        print(f'  {lbl}: {len(kp)} k-points, path length {kd[-1]:.3f} Å⁻¹')

    print('\nLoading ss-GW reference (Jiang, priv. comm.)…')
    gw_lines, _ = _load_gw(paths['3D']['ticks'])

    print('\nComputing TB bands + masses with Gen 140…')
    t0 = time.perf_counter()
    for lbl in LABELS:
        geom = geoms[lbl]
        p    = paths[lbl]

        # bands along k-path
        bands = ham_ortho.eigenvalues_batch(lbl, p['kp'], GEN140, geom=geom)
        gi  = int(np.argmin(np.linalg.norm(p['kp'], axis=1)))
        vbm = bands[gi, nf[lbl] - 1]
        bands -= vbm

        # effective masses
        bands_m = ham_ortho.eigenvalues_batch(lbl, p['kmass'], GEN140, geom=geom)

        if lbl == '3D':
            mv, mc = _masses_3D(bands_m, nf[lbl])
            _3D_MASSES = (mv, mc)
            print(f'  {lbl}: Eg(Γ)={bands[gi, nf[lbl]]:.3f}  '
                  f'mv=[{mv[0]:+.3f},{mv[1]:+.3f},{mv[2]:+.3f}]  '
                  f'mc=[{mc[0]:+.3f},{mc[1]:+.3f},{mc[2]:+.3f}]')
            _plot_3D(out_dir, p['kp'], p['kd'], p['ticks'], p['tlabs'],
                     bands, nf[lbl], gw_lines)
        else:
            mu1, mu2, mu_avg, mh1, mh2, me1, me2 = _masses_2D(bands_m, nf[lbl])
            print(f'  {lbl}: Eg(Γ)={bands[gi, nf[lbl]]:.3f}  '
                  f'μ=({mu1:.3f}, {mu2:.3f}), avg={mu_avg:.3f}')
            _plot_2D(out_dir, lbl, p['kp'], p['kd'], p['ticks'], p['tlabs'],
                     bands, nf[lbl],
                     mu1, mu2, mu_avg, mh1, mh2, me1, me2)

    dt = time.perf_counter() - t0
    print(f'\nDone in {dt:.1f} s.')

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
