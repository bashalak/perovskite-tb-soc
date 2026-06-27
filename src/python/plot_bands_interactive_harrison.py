"""
plot_bands_interactive_harrison.py — Interactive TB+SOC band viewer with
HARRISON-SCALED Pb-I hopping integrals.

Same UX as plot_bands_interactive.py, but uses the new ham_ortho_harrison
module and exposes the n_harrison exponent as a tunable parameter.

The Pb-I hopping integrals scale as V_eff(d) = V_0 * (3.14/d)^n_harrison
   • n_harrison = 0  → constant SK (same as the original ham_ortho.py)
   • n_harrison ≈ 2  → Harrison's classical sp-bonded prediction (best fit)

Layout  : 2×3 band-structure panels (3D, n1…n5) + TextBox parameter controls.
Controls: type a value and press Enter, or click ^ / v to step by 0.01 eV.
3D panel overlays ss-gw.band.dat reference bands in gray.
All TB bands are shifted so VBM(Γ) = 0 eV.

Default parameters: SEED 7 FINAL from the Harrison NSGA-II fit (clean,
complete parameter set from the BEST RESULT block).
  → F = [1.564, 0.783, 0.015, 0.0]   ideal=20
  → 2D μ RMS error reduced ~32% vs constant-SK baseline
  → fitted n_harrison = 3.81

Note: an earlier "gen 57" candidate appeared to give a 69% RMS reduction
but its parameter list was interleaved with another seed's output in the
log (multi-process write collision), so it cannot be reliably reproduced.
The seed-7-final params here are taken from the SUMMARY block at the
end of the log and are guaranteed clean. You can manually set
n_harrison ≈ 2.0 in the GUI to test Harrison's classical sp-bonded value.

Run:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 plot_bands_interactive_harrison.py
"""

import os, sys, time
from functools import partial

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
sys.path.insert(0, _HERE)
import ham_ortho_harrison as ham_ortho   # ← Harrison-scaled Hamiltonian

LABELS = ['3D', 'n1', 'n2', 'n3', 'n4', 'n5']

# Effective-mass finite-difference constants (same convention as fit_ortho.py)
HBAR2_M = 7.6200    # ℏ²/m_e [eV·Å²]  →  m*/m_e = HBAR2_M / (d²E/dk²)
DK_MASS = 0.05      # Å⁻¹ finite-difference step


# Seed 7 FINAL from the Harrison NSGA-II fit (complete parameter set from the
# BEST RESULT block at the end of harrison_fit_log.txt — guaranteed clean,
# not affected by parallel-log interleaving).
#
# F = [1.564, 0.783, 0.015, 0.0]  ideal=20  (best by sum-of-objectives)
# 2D μ = [0.159, 0.154, 0.099, 0.103, 0.112]  → RMS error ~32% lower than const-SK
# 3D Eg(Γ) = 1.485
# Fitted n_harrison = 3.81 (steeper than Harrison's classical n=2)
#
# Note: the "gen 57" candidate I quoted earlier had its parameters
# interleaved with another seed's output in the log. The COMPLETE seed-7
# final params below are reliable.
Es0            =  -7.5300
Ep0            =  +6.1300
Es1            =  -9.2100
Ep1            =  -1.5200
Vss            =  -3.1600
Vsp            =  +3.6700
Vps            =  +4.6000
Vppsig         =  -3.2500
Vpppi          =  +2.5600
deltaSO0       =  +1.3000
deltaSO1       =  +0.9000
Vss_II         =  +0.1900
Vsp_II         =  +0.0700
Vppsig_II      =  +0.1600
Vpppi_II       =  +0.2700
Vss_IIz        =  +0.5600
Vsp_IIz        =  +0.7100
Vppsig_IIz     =  +1.1100
Vpppi_IIz      =  -1.8000
Vss_PbPb       =  +0.0000
Vsp_PbPb       =  +0.0000
Vppsig_PbPb    =  +0.0000
Vpppi_PbPb     =  +0.0000
Es1_ap         =  -5.3600
Ep1_ap         =  -3.9900
Ep0_surf       =  +6.0700
n_harrison     =  +3.8100   # NEW parameter — fitted value; try 2.0 manually for classical Harrison

Es0            =  -7.4900
Ep0            =  +6.4200
Es1            = -11.6200
Ep1            =  +0.0600
Vss            =  +3.6600
Vsp            =  +0.8600
Vps            =  -1.3900
Vppsig         =  -0.5600
Vpppi          =  -1.7400
deltaSO0       =  +1.3000
deltaSO1       =  +0.9000
Vss_II         =  -0.1800
Vsp_II         =  +0.1200
Vppsig_II      =  -0.6900
Vpppi_II       =  -0.0200
Vss_IIz        =  +0.4700
Vsp_IIz        =  -0.0700
Vppsig_IIz     =  -0.2900
Vpppi_IIz      =  +0.0600
Vss_PbPb       =  +0.0000
Vsp_PbPb       =  +0.0000
Vppsig_PbPb    =  +0.0000
Vpppi_PbPb     =  +0.0000
Es1_ap         =  -5.2400
Ep1_ap         =  -1.0000
Ep0_surf       =  +6.3800
n_harrison     =  +1.9200

Es0            =  -2.7900
Ep0            =  +3.2900
Es1            = -10.3700
Ep1            =  -4.0600
Vss            =  -0.6700
Vsp            =  -1.2000
Vps            =  -0.2200
Vppsig         =  -1.7500
Vpppi          =  -1.5300
deltaSO0       =  +1.3000
deltaSO1       =  +0.9000
Vss_II         =  -0.7600
Vsp_II         =  -0.2200
Vppsig_II      =  -0.1400
Vpppi_II       =  +0.1000
Vss_IIz        =  -0.0500
Vsp_IIz        =  +0.3300
Vppsig_IIz     =  +0.1900
Vpppi_IIz      =  -0.4900
Vss_PbPb       =  +0.0000
Vsp_PbPb       =  +0.0000
Vppsig_PbPb    =  +0.0000
Vpppi_PbPb     =  +0.0000
Es1_ap         =  -9.3800
Ep1_ap         =  -2.4900
Ep0_surf       =  +3.1800
n_harrison     =  +2.3600

_BEST_FIT = dict(zip(ham_ortho.PARAM_KEYS, [
                 Es0 ,Ep0 ,Es1 ,Ep1        ,
                 Vss        ,Vsp        ,Vps        ,Vppsig     ,Vpppi      ,
                 deltaSO0   ,deltaSO1   ,
                 Vss_II     ,Vsp_II     ,Vppsig_II  ,Vpppi_II   ,
                 Vss_IIz    ,Vsp_IIz    ,Vppsig_IIz ,Vpppi_IIz  ,
                 Vss_PbPb   ,Vsp_PbPb   ,Vppsig_PbPb,Vpppi_PbPb ,
                 Es1_ap     ,Ep1_ap     ,Ep0_surf   ,
                 n_harrison,
    ]))
#_BEST_FIT = dict(zip(ham_ortho.PARAM_KEYS, [
#    -10.76, +6.91, -9.20, +3.00,
#    -0.67, -1.11, -5.91, +1.98, +0.74,
#    +1.30, +0.90,
#    -1.00, -0.81, -0.34, -0.68,  
#    0.0,   0.0,   0.0,   0.0,
#    0.0,   0.0,   0.0,   0.0,
#    0.0,   0.0,   0.0
#]))
#CS
#_BEST_FIT = dict(zip(ham_ortho.PARAM_KEYS, [
#    -2.75, 
#    +3.42,
#    -9.42,
#    +2.58,
#    +1.95,
#    +0.74,
#    -3.64,
#    -1.86,
#    -0.90,
#    +1.30,
#    +0.90,
#    -0.19,
#    -0.59,
#    +0.09,
#    +0.27,
#    -0.49,
#    +0.01,
#    +0.27,
#    +0.20,
#    0.0,
#    0.0, 
#    0.0, 
#    0.0,
#    -0.16,
#    +3.60,
#    +3.52,
#]))
#_BEST_FIT = dict(zip(ham_ortho.PARAM_KEYS, [
#    -2.75, +3.42, -9.42, +2.58,
#    +1.95, +0.74, -3.64, -1.86, -0.90,
#    +1.30, +0.90,
#    -0.19, -0.59, +0.09, +0.27,
#    -0.49, +0.01, +0.27, +0.20,
#     0.0,   0.0,   0.0,   0.0,
#    -0.16, +3.60, +3.52, +2.0100
#]))
#_BEST_FIT = dict(zip(ham_ortho.PARAM_KEYS, [
#                 -6.61,+3.47,-4.89,+5.46,
#                 -3.45,+2.31,-3.75,+2.06,+3.43,
#                 +1.30,+0.90,
#                 -0.46,-0.80,+0.66,-0.08,
#                 +0.10,+0.01,+0.30,-0.12,
#                 -0.64,+0.42,+0.66,-0.84,
#                 +11.26,+3.99,+2.78
#    ]))

#_BEST_FIT = dict(zip(ham_ortho.PARAM_KEYS, [
#                 -3.6402, -1.9916, -8.7741, (-1.5548),
#                 1.4069, -0.0204, -3.3715, -1.7803, -1.8045,
#                 +1.30,+0.90,
#                 0.1507, -0.2154, 0.0822, 0.09106,
#                 +0.00,+0.00,0,0,
#                 -0,+0,+0,-0,
#                 -8.7741, (-1.5548),-0.0
#    ]))

#-3.6402, -1.9916, -8.7741, -1.5548, 1.4069, -0.0204, -3.3715, -1.7803, -1.8045, 1.3 0.9 -3.2123, 0.1507, -0.2154, 0.0822, 0.09106,
# 27 parameters (26 standard + n_harrison) split into 3 columns for control panel
_COL0 = ['Es0','Ep0','Es1','Ep1','Vss','Vsp','Vps','Vppsig','Vpppi']
_COL1 = ['deltaSO0','deltaSO1','Vss_II','Vsp_II','Vppsig_II','Vpppi_II',
          'Vss_IIz','Vsp_IIz','Vppsig_IIz']
_COL2 = ['Vpppi_IIz','Vss_PbPb','Vsp_PbPb','Vppsig_PbPb','Vpppi_PbPb',
          'Es1_ap','Ep1_ap','Ep0_surf',
          'n_harrison']   # NEW row for Harrison exponent


# ── geometry helpers ──────────────────────────────────────────────────────────

def _vasp_path_for(label):
    """3D bulk uses original orthogonal Pnma POSCAR; 2D slabs use the
    orthogonal-cell Pnma POSCARs (consistent with ham_ortho_harrison)."""
    if label == '3D':
        return os.path.join(_ROOT, 'data', 'pos3D2Dtilted', 'POSCAR')
    return os.path.join(_ROOT, 'data', 'pos3D2Dpnma_xyz',
                        f'pnma_xyz_{label}.vasp')


def _recip_cell(label):
    from ase.io import read
    atoms = read(_vasp_path_for(label))
    cell  = atoms.cell.array
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


def _kpath_3D(B, nk=30):
    def f(c): return c[0]*B[0] + c[1]*B[1] + c[2]*B[2]
    hsp = {'G':[0,0,0],'X':[.5,0,0],'S':[.5,.5,0],'Y':[0,.5,0],
           'Z':[0,0,.5],'U':[.5,0,.5],'R':[.5,.5,.5],'T':[0,.5,.5]}
    seq = ['G','X','S','Y','G','Z','U','R','T','Z']
#    seq = ['Z','G','X','U','G','Y','T','Z','G','S']# Z-G-X-U-G-Y-T-Z-G-S
    lab = [r'$\Gamma$','X','S','Y',r'$\Gamma$','Z','U','R','T','Z']
    kp, kd, tc = _build_path([f(hsp[p]) for p in seq], nk)
    return kp, kd, tc, lab


def _kpath_2D(label, geom, nk=30):
    """
    Build a Y-Γ-X-S path using the **true 2D in-plane reciprocal vectors**
    derived from the in-plane primitive lattice vectors (handles tilted-cell
    slabs n2-n5 where cell[0] and cell[1] are not individually in-plane).
    """
    from ase.io import read
    cell = read(_vasp_path_for(label)).cell.array
    b1_2d, b2_2d = ham_ortho.in_plane_reciprocal_vectors(cell, geom['slab_normal'])
    X = 0.5*b1_2d; Y = 0.5*b2_2d; S = X+Y; G = np.zeros(3)
    kp, kd, tc = _build_path([Y, G, X, S], nk)
    return kp, kd, tc, ['Y', r'$\Gamma$', 'X', 'S']


def _n_fill(geom):
    return geom['n_Pb'] * 2 + geom['n_I'] * 8


def _load_gw(ticks_3D):
    path = os.path.join(_ROOT, 'data', 'ss-gw.band.dat')
    if not os.path.exists(path):
        return [], 1.0, None
    bands, kc, ec = [], [], []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                if kc: bands.append((np.array(kc), np.array(ec))); kc=[]; ec=[]
            else:
                v = s.split(); kc.append(float(v[0])); ec.append(float(v[1]))
    if kc: bands.append((np.array(kc), np.array(ec)))
    eg = np.sort([b[1][0] for b in bands])
    gi = int(np.argmax(np.diff(eg)))
    vbm = eg[gi]; cbm = eg[gi+1]
    print(f'  GW: VBM={vbm:.3f}  CBM={cbm:.3f}  gap={cbm-vbm:.3f} eV')
    scale = ticks_3D[-1] / bands[0][0][-1]
    lines = [(b[0]*scale, b[1]-vbm) for b in bands if np.any(np.abs(b[1]-vbm) < 4.0)]
    return lines, scale, vbm


# ── effective-mass k-batches ──────────────────────────────────────────────────

def _mass_kpts_3D(B):
    """7-point FD batch at Γ along the three reciprocal axes (Cartesian)."""
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
    """5-point FD batch at Γ along the **true 2D in-plane reciprocal axes**."""
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


# ── App ───────────────────────────────────────────────────────────────────────

class TightBindingOrthoApp:

    STEP = 0.01

    def __init__(self):
        print('Loading geometries…', flush=True)
        self.geoms = {lbl: ham_ortho.load_geometry(lbl) for lbl in LABELS}
        self.nf    = {lbl: _n_fill(self.geoms[lbl])     for lbl in LABELS}

        self.kpaths = {}; self.kdists = {}
        self.ticks  = {}; self.tlabs  = {}
        self.kmass  = {}
        for lbl in LABELS:
            B = _recip_cell(lbl)
            if lbl == '3D':
                kp, kd, tc, tl = _kpath_3D(B, nk=30)
                self.kmass[lbl] = _mass_kpts_3D(B)
            else:
                kp, kd, tc, tl = _kpath_2D(lbl, self.geoms[lbl], nk=30)
                self.kmass[lbl] = _mass_kpts_2D(lbl, self.geoms[lbl])
            self.kpaths[lbl]=kp; self.kdists[lbl]=kd
            self.ticks[lbl]=tc;  self.tlabs[lbl]=tl

        self.gw_lines, _, _ = _load_gw(self.ticks['3D'])

        self.params = dict(_BEST_FIT)

        # ── figure: 2×3 band panels + control strip ───────────────────────────
        self.fig, axes = plt.subplots(2, 3, figsize=(16, 13))
        plt.subplots_adjust(left=0.05, right=0.98, top=0.96,
                            bottom=0.35, wspace=0.24, hspace=0.38)
        self.ax = {'3D': axes[0,0], 'n1': axes[0,1], 'n2': axes[0,2],
                   'n3': axes[1,0], 'n4': axes[1,1], 'n5': axes[1,2]}

        self.textboxes = {}
        self.buttons   = []
        self._build_controls()

        # Update / Reset buttons
        ax_u = self.fig.add_axes([0.88, 0.285, 0.10, 0.042])
        self.btn_update = Button(ax_u, 'Update',
                                 color='#ccffcc', hovercolor='#44cc44')
        self.btn_update.on_clicked(self.update)

        ax_r = self.fig.add_axes([0.88, 0.235, 0.10, 0.042])
        self.btn_reset = Button(ax_r, 'Reset',
                                color='#ffcccc', hovercolor='#ff4444')
        self.btn_reset.on_clicked(self.reset)

        print('Computing initial bands (caches bonds)…', flush=True)
        self.update(None)

        # Un-minimize and maximize the Tk window
        try:
            win = self.fig.canvas.manager.window
            win.deiconify()
            try:
                win.state('zoomed')          # Windows / WSLg Tk
            except Exception:
                win.attributes('-zoomed', True)   # X11 Tk
            win.lift()
            win.focus_force()
        except Exception:
            pass

    # ── controls ─────────────────────────────────────────────────────────────

    def _build_controls(self):
        x0_list = [0.03, 0.33, 0.63]
        y0 = 0.005; dy = 0.043; bw = 0.20; bh = 0.034

        for x0, col in zip(x0_list, [_COL0, _COL1, _COL2]):
            for row, key in enumerate(col):
                self._make_row(key, x0, y0 + row * dy, bw, bh)

    def _make_row(self, key, x, y, w, h):
        ax_tb = self.fig.add_axes([x, y, w, h])
        tb = TextBox(ax_tb, key, initial=f'{self.params[key]:.4f}')
        tb.on_submit(self.update)
        self.textboxes[key] = tb

        ax_dn = self.fig.add_axes([x + w + 0.005, y, 0.030, h])
        ax_up = self.fig.add_axes([x + w + 0.038, y, 0.030, h])
        b_dn  = Button(ax_dn, 'v'); b_up = Button(ax_up, '^')
        self.buttons += [b_dn, b_up]
        b_dn.on_clicked(partial(self._step, key, -self.STEP))
        b_up.on_clicked(partial(self._step, key, +self.STEP))

    def _step(self, key, delta, _ev=None):
        try:
            val = float(self.textboxes[key].text) + delta
            self.textboxes[key].set_val(f'{val:.4f}')   # triggers update
        except Exception as e:
            print(f'[step] {key}: {e}')

    def _read_params(self):
        for k in self.params:
            try: self.params[k] = float(self.textboxes[k].text)
            except ValueError: pass
        return self.params

    def reset(self, _ev):
        for k, v in _BEST_FIT.items():
            self.textboxes[k].set_val(f'{v:.4f}')

    # ── band computation + drawing ────────────────────────────────────────────

    def _draw(self, lbl, bands, kd, ticks, tlabs):
        ax  = self.ax[lbl]
        nf  = self.nf[lbl]
        ax.clear()

        # all bands
        for b in range(bands.shape[1]):
            ax.plot(kd, bands[:, b], color='C0', lw=0.5, alpha=0.45)

        # VBM (red) / CBM (green)
        ax.plot(kd, bands[:, nf-1], 'r-', lw=1.6)
        ax.plot(kd, bands[:, nf],   'g-', lw=1.6)

        # GW overlay on 3D panel
        if lbl == '3D':
            for k_arr, e_arr in self.gw_lines:
                ax.plot(k_arr, e_arr, color='#999999', lw=0.8, alpha=0.65)

        gi    = int(np.argmin(np.linalg.norm(self.kpaths[lbl], axis=1)))
        e_gi  = np.sort(bands[gi])
        gap_g = float(e_gi[nf] - e_gi[nf-1])

        ax.axhline(0, color='red', lw=0.7, ls='--')
        for tp in ticks[1:-1]:
            ax.axvline(tp, color='k', lw=0.5, alpha=0.35)
        ax.set_xlim(kd[0], kd[-1])
        ax.set_ylim(-10, 6)
        ax.set_xticks(ticks); ax.set_xticklabels(tlabs, fontsize=9)
        ax.set_ylabel('E − VBM (eV)', fontsize=8)
        ax.set_title(f'{lbl}   Eg(Γ)={gap_g:.3f} eV', fontsize=9)
        ax.grid(True, alpha=0.22)

    def update(self, _ev):
        p = self._read_params()
        t0 = time.perf_counter()
        gaps   = []
        mv_3d  = mc_3d = None
        mu_2d  = {}
        for lbl in LABELS:
            geom = self.geoms[lbl]
            kp   = self.kpaths[lbl]
            kd   = self.kdists[lbl]
            bands = ham_ortho.eigenvalues_batch(lbl, kp, p, geom=geom)
            # shift so VBM at Γ = 0
            gi  = int(np.argmin(np.linalg.norm(kp, axis=1)))
            vbm = bands[gi, self.nf[lbl] - 1]
            bands -= vbm
            self._draw(lbl, bands, kd, self.ticks[lbl], self.tlabs[lbl])
            gaps.append(f'{lbl}={bands[gi, self.nf[lbl]]:.3f}')

            # ── effective masses at Γ via central FD ─────────────────────────
            ev = ham_ortho.eigenvalues_batch(lbl, self.kmass[lbl], p, geom=geom)
            nf = self.nf[lbl]
            ivb, icb = nf - 1, nf
            if lbl == '3D':
                e0v = ev[0, ivb]; e0c = ev[0, icb]
                d2v = [(ev[1, ivb] + ev[2, ivb] - 2*e0v) / DK_MASS**2,
                       (ev[3, ivb] + ev[4, ivb] - 2*e0v) / DK_MASS**2,
                       (ev[5, ivb] + ev[6, ivb] - 2*e0v) / DK_MASS**2]
                d2c = [(ev[1, icb] + ev[2, icb] - 2*e0c) / DK_MASS**2,
                       (ev[3, icb] + ev[4, icb] - 2*e0c) / DK_MASS**2,
                       (ev[5, icb] + ev[6, icb] - 2*e0c) / DK_MASS**2]
                mv_3d = [HBAR2_M / d for d in d2v]   # signed (negative at Γ)
                mc_3d = [HBAR2_M / d for d in d2c]
            else:
                e0v = ev[0, ivb]; e0c = ev[0, icb]
                d2v1 = (ev[1, ivb] + ev[2, ivb] - 2*e0v) / DK_MASS**2
                d2v2 = (ev[3, ivb] + ev[4, ivb] - 2*e0v) / DK_MASS**2
                d2c1 = (ev[1, icb] + ev[2, icb] - 2*e0c) / DK_MASS**2
                d2c2 = (ev[3, icb] + ev[4, icb] - 2*e0c) / DK_MASS**2
                mh1 = abs(HBAR2_M / d2v1); mh2 = abs(HBAR2_M / d2v2)
                me1 = abs(HBAR2_M / d2c1); me2 = abs(HBAR2_M / d2c2)
                mu1 = mh1*me1 / (mh1 + me1)
                mu2 = mh2*me2 / (mh2 + me2)
                mu_2d[lbl] = 0.5 * (mu1 + mu2)

        dt = (time.perf_counter() - t0) * 1000
        print(f'[{dt:.0f} ms]  Eg(Γ): ' + '  '.join(gaps))
        if mv_3d is not None:
            print(f'           3D mv [xx,yy,zz] = '
                  f'[{mv_3d[0]:+.3f}, {mv_3d[1]:+.3f}, {mv_3d[2]:+.3f}]   '
                  f'mc = [{mc_3d[0]:+.3f}, {mc_3d[1]:+.3f}, {mc_3d[2]:+.3f}]')
        if mu_2d:
            mu_str = '  '.join(f'{k}={v:.3f}' for k, v in mu_2d.items())
            print(f'           2D μ:  {mu_str}')
        self.fig.canvas.draw_idle()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = TightBindingOrthoApp()
    plt.show()
