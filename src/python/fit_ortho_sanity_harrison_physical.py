"""
fit_ortho_sanity.py — fit_ortho.py + full physical-sanity constraint block.

Difference from fit_ortho.py:
  • Adds 3D sanity:  gap > 0, mv < 0 (xx,yy,zz), mc > 0, dVBM ≤ 0, dCBM ≥ 0
  • Adds 2D sanity:  gap > 0, mv_2D < 0 (e1,e2), mc_2D > 0
  • Adds confinement monotonicity:  μ(n+1) ≤ μ(n)
  • Returns and uses signed 2D band curvatures from objectivesfunctions

Run:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 fit_ortho_sanity.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time
import numpy as np
import multiprocessing as mp

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.mutation.gauss import GaussianMutation
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.repair import Repair
from pymoo.core.callback import Callback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..')
import sys; sys.path.insert(0, _HERE)
import ham_ortho_harrison as ham_ortho


# ================================================================
# CONFIGURATION
# ================================================================

# 'joint' | '3D' | '2D'
FIT_MODE = 'joint'

# Toggle the new sanity constraint block on/off for experimentation
ENABLE_SANITY = True

SEEDS       = [815, 314, 7,  1234]   # short test with 3 best seeds 42, 7, 1234, 99, 314, 500, 815, 271
N_GEN       = 200
POP_SIZE    = 150
MAX_IDEAL   = 30
MIN_GEN     = 80
OUTPUT_FILE = 'best_ortho_sanity_harrison_physical.npz'

#WARM_START = np.array([
#    -5.99, +6.85, -14.55, -0.07,
#    -1.20, -1.64, +4.93, -0.20, -0.21,
#    +1.30, +0.90,
#    +0.19, -0.93, -0.28, -0.83,
#    -0.00, +0.00, +0.00, +0.00,
#     0.00,  0.00,  0.00,  0.00,
#    -0.00, +0.00, +0.00,
#])
# Warm start = seed-815 best from the previous fit + Harrison exponent = 2.0
# (Harrison's original prediction for sp-bonded materials)
#WARM_START = np.array([
#    -7.53, +6.13, -9.33, -1.52,
#    -3.06, +3.61, +4.62, -3.31, +2.53,
#    +1.30, +0.90,
#    +0.22, +0.07, +0.17, +0.27,
#    +0.56, +0.71, +1.11, -1.93,
#     0.00,  0.00,  0.00,  0.00,
#    -5.17, -5.40, +5.84,
#    +2.00,                          # n_harrison: Harrison's original n=2 starting guess
#])

# WARM START = Gen 48 (best physically-clean candidate from harrison-pnma fit)
# Es0>Es1 ✓, Vss<-0.5 ✓, |Vppσ|>|Vppπ| ✓, surface shifts ≤ 3 eV ✓
# n_harrison = 2.38 — within the [2,4] range for soft sp-bonded materials
WARM_START = np.array([
    -2.79, +3.41, -12.22, -4.07,                # Es0, Ep0, Es1, Ep1
    -1.24, -1.13, -0.27, -1.78, -1.53,           # Vss, Vsp, Vps, Vppσ, Vppπ
    +1.30, +0.90,                                # δSO0, δSO1
    -0.17, -0.25, -0.14, +0.29,                  # Vss_II, Vsp_II, Vppσ_II, Vppπ_II
    -1.99, -0.53, +0.33, -0.82,                  # Vss_IIz, Vsp_IIz, Vppσ_IIz, Vppπ_IIz
     0.0,   0.0,   0.0,   0.0,                  # PbPb (frozen at 0)
    -10.35, -1.74, +2.63,                        # Es1_ap, Ep1_ap, Ep0_surf
    +2.38,                                       # n_harrison
])
#WARM_START = np.array([
#     -10.76, +6.91, -9.20, +3.00,
#     -0.67, -1.11, -5.91, +1.98, +0.74,
#     +1.30, +0.90,
#     -1.00, -0.81, -0.34, -0.68,
#     0.0,   0.0,   0.0,   0.0,
#     0.0,   0.0,   0.0,   0.0,
#     -0.16, +3.60, +3.52
#])


PARAM_NAMES = ham_ortho.PARAM_KEYS
N_VAR = len(PARAM_NAMES)

OBJ_NAMES = ['energy_err', 'mass_err', 'order_err', 'disp_err']
N_OBJ = 4

# ── Targets ────────────────────────────────────────────────────────
shift = 0.0
EG_3D_GA = 1.5725 + shift-0.0525 #traore: 1.5200, cubic: 1.62 
EG_3D_Y  = 2.5457 + shift
EG_3D_Z  = 3.4796 + shift
EG_3D_X  = 3.48  + shift
EG_3D_S  = 4.83  + shift
EG_3D_U  = 2.87  + shift
EG_3D_R  = 3.81  + shift
EG_3D_T  = 4.84  + shift

#TARGET_MV = np.array([-0.209, -0.234, -0.207])
#TARGET_MC = np.array([ 0.236,  0.199,  0.245])

TARGET_MV = np.array([-0.32, -0.32, -0.32]) #traore
TARGET_MC = np.array([ 0.28, 0.28, 0.28]) #traore

#TARGET_EG_2D = np.array([3.016, 2.432, 2.206, 2.078, 1.971])
#TARGET_MU_2D = np.array([0.221, 0.217, 0.201, 0.196, 0.186])

TARGET_EG_2D = np.array([2.84, 2.45, 2.29, 2.32, 2.01]) #traore
TARGET_MU_2D = np.array([0.18, 0.17, 0.14, 0.14, 0.14]) #traore

W_EG_3D = np.array([4.0, 2.5, 2.0, 2.0, 1.0, 2.5, 2.0, 1.0])

TARGET_DVBM = np.array([-0.339, -0.800, -0.797, -1.103, -0.516, -0.897, -1.158])
TARGET_DCBM = np.array([+0.634, +1.107, +1.111, +2.150, +0.776, +1.339, +2.112])
W_DISP_3D   = 1.5

# ── Constraint windows ─────────────────────────────────────────────
EG_3D_GA_WIN = [1.47+shift-0.0525, 1.67+shift-0.0525]
EG_3D_Y_WIN  = [2.30+shift, 2.80+shift]
EG_3D_Z_WIN  = [3.20+shift, 3.80+shift]
EG_3D_X_WIN  = [3.0+shift, 4.0+shift]
EG_3D_S_WIN  = [4.0+shift, 5.5+shift]
EG_3D_U_WIN  = [2.5+shift, 3.5+shift]
EG_3D_R_WIN  = [3.0+shift, 4.5+shift]
EG_3D_T_WIN  = [4.0+shift, 6.0+shift]

#MV_WIN = np.array([[-0.26, -0.17], [-0.28, -0.19], [-0.26, -0.17]])
#MC_WIN = np.array([[ 0.19,  0.28], [ 0.16,  0.24], [ 0.20,  0.30]])

MV_WIN = np.array([[-0.36, -0.28], [-0.36, -0.28], [-0.36, -0.28]]) #traore
MC_WIN = np.array([[ 0.25,  0.31], [0.25,  0.31], [0.25,  0.31]]) #traore

#EG_2D_WIN = np.array([
#    [2.60, 3.25], [2.30, 2.50], [2.10, 2.30], [1.90, 2.15], [1.80, 2.00],
#])

EG_2D_WIN = np.array([
    [2.60, 3.00], [2.35, 2.55], [2.20, 2.40], [2.2, 2.40], [1.90, 2.10],
]) #traore

#MU_2D_WIN = np.array([
#    [0.22, 0.23], [0.21, 0.22], [0.20, 0.21], [0.19, 0.20], [0.18, 0.19],
#])
MU_2D_WIN = np.array([
    [0.15, 0.20], [0.14, 0.19], [0.11, 0.16], [0.11, 0.16], [0.11, 0.16],
])#traore
# ── N_CONSTR — adapt per mode and sanity flag ──────────────────────
# Physics-window counts (same as fit_ortho.py)
_N_CONSTR_3D_BASE = 28      # gaps Γ/Y/Z + mv/mc windows + X/S/U/R/T gaps
_N_CONSTR_2D_BASE = 24      # 2D gaps + 2D μ windows + ordering

# Sanity counts (new)
_N_SANITY_3D = 1 + 3 + 3 + 7 + 7   # gap>0 + mv<0(3) + mc>0(3) + dvbm≤0(7) + dcbm≥0(7) = 21
_N_SANITY_2D = 5 + 10 + 10         # 2D gap>0(5) + mv_2d<0(5×2) + mc_2d>0(5×2) = 25
_N_MU_MONO   = 4                    # μ(n+1) ≤ μ(n) for n=1..4
_N_PARAM_SANITY = 8                 # orbital ordering + |Vss| + 6 NEW physical SK constraints

# ── Parameter-basin constraints ──────────────────────────────────────────
# Force the optimizer out of unphysical basins:
#   - "spectator-s" (small |Vss| with wrong Es ordering)
#   - "π > σ" (which violates standard SK theory)
#   - extreme surface corrections (> 3 eV chemical shift is unphysical)
# All constraints have the form g(x) ≤ 0 if satisfied.
ORB_ORDER_MARGIN = 1.0   # eV — require Es0 ≥ Es1 + 1   (Pb 6s clearly above I 5s)
VSS_MIN_ABS      = 0.5   # eV — require |Vss| ≥ 0.5     (no Pb 6s spectator basin)
VSS_MAX_POS      = -0.5  # require Vss ≤ -0.5 (negative sign, bonding convention)
VPPSIG_MIN_ABS   = 1.0   # require |Vppσ| ≥ 1.0 (substantial σ-bond)
SURF_SHIFT_MAX   = 3.0   # |Es1_ap - Es1|, |Ep1_ap - Ep1|, |Ep0_surf - Ep0| ≤ 3 eV


def _param_sanity_constraints(x):
    """Parameter-basin sanity. Convention: g ≤ 0 means satisfied.
       Layout (PARAM_KEYS index):
         x[0] = Es0     x[1] = Ep0     x[2] = Es1     x[3] = Ep1
         x[4] = Vss     x[5] = Vsp     x[6] = Vps
         x[7] = Vppsig  x[8] = Vpppi
         x[23]= Es1_ap  x[24]= Ep1_ap  x[25]= Ep0_surf
    """
    es0, es1   = x[0], x[2]
    ep0, ep1   = x[1], x[3]
    vss        = x[4]
    vppsig     = x[7]
    vpppi      = x[8]
    es1_ap     = x[23]
    ep1_ap     = x[24]
    ep0_surf   = x[25]

    g_order   = es1 - es0 + ORB_ORDER_MARGIN     # Es0 ≥ Es1 + 1
    g_vss_mag = VSS_MIN_ABS - abs(vss)           # |Vss| ≥ 0.5
    g_vss_neg = vss - VSS_MAX_POS                # Vss ≤ -0.5 (negative)
    g_sigma   = abs(vpppi) - abs(vppsig)         # |Vppσ| ≥ |Vppπ|
    g_sig_mag = VPPSIG_MIN_ABS - abs(vppsig)     # |Vppσ| ≥ 1.0
    g_es1_ap  = abs(es1_ap - es1) - SURF_SHIFT_MAX   # surface I 5s shift ≤ 3
    g_ep1_ap  = abs(ep1_ap - ep1) - SURF_SHIFT_MAX   # surface I 5p shift ≤ 3
    g_ep0_sf  = abs(ep0_surf - ep0) - SURF_SHIFT_MAX # surface Pb 6p shift ≤ 3

    return [g_order, g_vss_mag, g_vss_neg, g_sigma, g_sig_mag,
            g_es1_ap, g_ep1_ap, g_ep0_sf]


def _compute_n_constr(mode, sanity_on):
    n = 0
    if mode in ('joint', '3D'):
        n += _N_CONSTR_3D_BASE
        if sanity_on: n += _N_SANITY_3D
    if mode in ('joint', '2D'):
        n += _N_CONSTR_2D_BASE
        if sanity_on: n += _N_SANITY_2D + _N_MU_MONO
    if sanity_on:
        n += _N_PARAM_SANITY               # orbital ordering + Vss coupling
    return n

N_CONSTR = _compute_n_constr(FIT_MODE, ENABLE_SANITY)

# ── Parameter bounds ───────────────────────────────────────────────
_XL_FULL = np.array([
    -15.0, -7.0, -15.0, -7.0,
    -4.0,  -5.0, -6.0,  -7.0, -5.0,
    +1.30, +0.90,
    -1.0,  -1.0, -1.0,  -1.0,
    -0.5,  -0.5, -0.5,  -0.5,
    -0.0,  -0.0, -0.0,  -0.0,
    -15.0,  -9.0, +0.0,
])
_XU_FULL = np.array([
    +0.0,  +7.0, +0.0,  +7.0,
    +4.0,  +5.0, +6.0,  +7.0, +5.0,
    +1.30, +0.90,
    +1.0,  +1.0, +1.0,  +1.0,
    +0.5,  +0.5, +0.5,  +0.5,
    +0.0,  +0.0, +0.0,  +0.0,
    +1.0,  +9.0, +12.0,
    +2.50,                              # n_harrison upper (typical 0..4)
])

# Extend lower bound array with -1.0 for n_harrison (allow slightly negative for safety)
_XL_FULL = np.append(_XL_FULL, [1.80])  # n_harrison lower

_IDX_3D = list(range(0, 15)) + list(range(19, 23))
_IDX_2D = list(range(15, 19)) + list(range(23, 26))
# n_harrison (index 26) is a JOINT parameter affecting all bonds
_IDX_JOINT_EXTRA = [26]
assert sorted(_IDX_3D + _IDX_2D + _IDX_JOINT_EXTRA) == list(range(27))

def _build_bounds(mode):
    xl, xu = _XL_FULL.copy(), _XU_FULL.copy()
    if   mode == 'joint': frozen = []
    elif mode == '3D':    frozen = _IDX_2D
    elif mode == '2D':    frozen = _IDX_3D
    else: raise ValueError(mode)
    for i in frozen:
        xl[i] = WARM_START[i]; xu[i] = WARM_START[i]
    return xl, xu

XL, XU = _build_bounds(FIT_MODE)

# ================================================================
# END CONFIGURATION
# ================================================================

HBAR2_M  = 7.6200
DK_MASS  = 0.05


# ── reciprocal lattice helpers ────────────────────────────────────────────────

def _recip_cell(label):
    from ase.io import read
    # 3D bulk: original orthogonal Pnma POSCAR (unchanged).
    # 2D slabs: new orthogonal-cell Pnma POSCARs (BFS-unwrapped, organics dropped).
    if label == '3D':
        path = os.path.join(_ROOT, 'data', 'pos3D2Dtilted', 'POSCAR')
    else:
        path = os.path.join(_ROOT, 'data', 'pos3D2Dpnma_xyz', f'pnma_xyz_{label}.vasp')
    atoms = read(path)
    cell  = atoms.cell.array
    V  = np.dot(cell[0], np.cross(cell[1], cell[2]))
    b1 = 2*np.pi * np.cross(cell[1], cell[2]) / V
    b2 = 2*np.pi * np.cross(cell[2], cell[0]) / V
    b3 = 2*np.pi * np.cross(cell[0], cell[1]) / V
    return np.array([b1, b2, b3])


def _in_plane_units(label, B):
    from ase.io import read
    # 2D slabs read from new orthogonal-cell Pnma POSCARs
    path = os.path.join(_ROOT, 'data', 'pos3D2Dpnma_xyz', f'pnma_xyz_{label}.vasp')
    cell = read(path).cell.array
    geom = ham_ortho.load_geometry(label)
    b1_2d, b2_2d = ham_ortho.in_plane_reciprocal_vectors(cell, geom['slab_normal'])
    return b1_2d / np.linalg.norm(b1_2d), b2_2d / np.linalg.norm(b2_2d)


def _build_kpoints():
    B3D = _recip_cell('3D')
    def f(c): return c[0]*B3D[0] + c[1]*B3D[1] + c[2]*B3D[2]
    KGA = f([0, 0, 0]);   KY = f([0, 0.5, 0]);   KZ = f([0, 0, 0.5])
    KX  = f([0.5, 0, 0]); KS = f([0.5, 0.5, 0]); KU = f([0.5, 0, 0.5])
    KR  = f([0.5, 0.5, 0.5]); KT = f([0, 0.5, 0.5])

    ex = B3D[0] / np.linalg.norm(B3D[0])
    ey = B3D[1] / np.linalg.norm(B3D[1])
    ez = B3D[2] / np.linalg.norm(B3D[2])

    kfd_3d = np.array([
        KGA,
        KGA + DK_MASS*ex, KGA - DK_MASS*ex,
        KGA + DK_MASS*ey, KGA - DK_MASS*ey,
        KGA + DK_MASS*ez, KGA - DK_MASS*ez,
        KY, KZ, KX, KS, KU, KR, KT,
    ])

    kfd_2d = {}
    for lbl in ['n1', 'n2', 'n3', 'n4', 'n5']:
        B_slab  = _recip_cell(lbl)
        e1, e2  = _in_plane_units(lbl, B_slab)
        GA      = np.zeros(3)
        kfd_2d[lbl] = np.array([
            GA,
            GA + DK_MASS*e1, GA - DK_MASS*e1,
            GA + DK_MASS*e2, GA - DK_MASS*e2,
        ])
    return kfd_3d, kfd_2d

_KFD_3D, _KFD_2D = _build_kpoints()


# ── physics ───────────────────────────────────────────────────────────────────

def objectivesfunctions(x):
    """
    Returns
    -------
    eg_ga, eg_y, eg_z              : 3D direct gaps at Γ, Y, Z [eV]
    mv, mc                         : ndarray(3,)  3D hole/electron masses [m₀]
    eg_2d                          : ndarray(5,) 2D gaps [eV]
    mu_2d                          : ndarray(5,) 2D in-plane reduced masses [m₀]
    eg_x, eg_s, eg_u, eg_r, eg_t   : 3D direct gaps at X, S, U, R, T [eV]
    dvbm                           : ndarray(7,) VBM(k)−VBM(Γ) at [Y,Z,X,S,U,R,T]
    dcbm                           : ndarray(7,) CBM(k)−CBM(Γ) at same points
    d2v_3d                         : ndarray(3,) signed 3D VBM curvatures [eV·Å²]
                                       (negative for a real VBM at Γ)
    d2c_3d                         : ndarray(3,) signed 3D CBM curvatures [eV·Å²]
                                       (positive for a real CBM at Γ)
    d2v_2d                         : ndarray(5,2) signed 2D VBM curvatures
                                       (rows=slabs, cols=e1,e2 directions)
    d2c_2d                         : ndarray(5,2) signed 2D CBM curvatures
    """
    params  = dict(zip(PARAM_NAMES, x))

    # ── 3D ────────────────────────────────────────────────────────────
    geom_3d = ham_ortho.load_geometry('3D')
    nf_3d    = geom_3d['n_Pb']*2 + geom_3d['n_I']*8
    ivb, icb = nf_3d - 1, nf_3d
    ev3 = ham_ortho.eigenvalues_batch('3D', _KFD_3D, params, geom=geom_3d)

    vbm_ga = ev3[0, ivb]; cbm_ga = ev3[0, icb]
    eg_ga  = cbm_ga - vbm_ga
    eg_y   = ev3[7,  icb] - ev3[7,  ivb]
    eg_z   = ev3[8,  icb] - ev3[8,  ivb]
    eg_x   = ev3[9,  icb] - ev3[9,  ivb]
    eg_s   = ev3[10, icb] - ev3[10, ivb]
    eg_u   = ev3[11, icb] - ev3[11, ivb]
    eg_r   = ev3[12, icb] - ev3[12, ivb]
    eg_t   = ev3[13, icb] - ev3[13, ivb]

    dvbm = ev3[7:14, ivb] - vbm_ga
    dcbm = ev3[7:14, icb] - cbm_ga

    # Signed 3D band curvatures at Γ along x, y, z
    def _curv(band_idx):
        e0  = ev3[0, band_idx]
        return np.array([
            (ev3[1, band_idx] + ev3[2, band_idx] - 2*e0) / DK_MASS**2,
            (ev3[3, band_idx] + ev3[4, band_idx] - 2*e0) / DK_MASS**2,
            (ev3[5, band_idx] + ev3[6, band_idx] - 2*e0) / DK_MASS**2,
        ])
    d2v_3d = _curv(ivb)   # should be ≤ 0 in xx,yy,zz at a valid VBM
    d2c_3d = _curv(icb)   # should be ≥ 0 in xx,yy,zz at a valid CBM
    mv = HBAR2_M / d2v_3d   # signed (negative at Γ)
    mc = HBAR2_M / d2c_3d   # positive

    # ── 2D ────────────────────────────────────────────────────────────
    labels_2d = ['n1', 'n2', 'n3', 'n4', 'n5']
    eg_2d  = np.zeros(5)
    mu_2d  = np.zeros(5)
    d2v_2d = np.zeros((5, 2))   # rows=slab, cols=e1/e2
    d2c_2d = np.zeros((5, 2))

    for i, lbl in enumerate(labels_2d):
        geom       = ham_ortho.load_geometry(lbl)
        nf         = geom['n_Pb']*2 + geom['n_I']*8
        ivb2, icb2 = nf-1, nf
        ev2 = ham_ortho.eigenvalues_batch(lbl, _KFD_2D[lbl], params, geom=geom)
        eg_2d[i] = ev2[:, icb2].min() - ev2[:, ivb2].max()

        e0v = ev2[0, ivb2]; e0c = ev2[0, icb2]
        d2v_2d[i, 0] = (ev2[1, ivb2] + ev2[2, ivb2] - 2*e0v) / DK_MASS**2
        d2v_2d[i, 1] = (ev2[3, ivb2] + ev2[4, ivb2] - 2*e0v) / DK_MASS**2
        d2c_2d[i, 0] = (ev2[1, icb2] + ev2[2, icb2] - 2*e0c) / DK_MASS**2
        d2c_2d[i, 1] = (ev2[3, icb2] + ev2[4, icb2] - 2*e0c) / DK_MASS**2

        mh1 = abs(HBAR2_M / d2v_2d[i, 0])
        mh2 = abs(HBAR2_M / d2v_2d[i, 1])
        me1 = abs(HBAR2_M / d2c_2d[i, 0])
        me2 = abs(HBAR2_M / d2c_2d[i, 1])
        mu1 = mh1*me1 / (mh1 + me1)
        mu2 = mh2*me2 / (mh2 + me2)
        mu_2d[i] = (mu1 + mu2) / 2.0

    return (eg_ga, eg_y, eg_z,
            mv, mc, eg_2d, mu_2d,
            eg_x, eg_s, eg_u, eg_r, eg_t,
            dvbm, dcbm,
            d2v_3d, d2c_3d, d2v_2d, d2c_2d)


def _build_G(eg_ga, eg_y, eg_z, mv, mc, eg_2d, mu_2d,
             eg_x, eg_s, eg_u, eg_r, eg_t,
             dvbm, dcbm,
             d2v_3d, d2c_3d, d2v_2d, d2c_2d):
    """
    Constraint vector G. Convention: G[i] ≤ 0 means satisfied.
    Layout per mode:
      3D:  windows (28) + sanity (21 if ENABLE_SANITY)
      2D:  windows (24) + sanity (25) + μ-monotonicity (4) if ENABLE_SANITY
      joint: union of both.
    """
    g = []

    if FIT_MODE in ('joint', '3D'):
        # ── 3D physics windows (28) ─────────────────────────────────
        for val, (lo, hi) in [(eg_ga, EG_3D_GA_WIN),
                              (eg_y,  EG_3D_Y_WIN),
                              (eg_z,  EG_3D_Z_WIN)]:
            g.append(val - hi); g.append(lo - val)
        for val, (lo, hi) in zip(mv, MV_WIN):
            g.append(val - hi); g.append(lo - val)
        for val, (lo, hi) in zip(mc, MC_WIN):
            g.append(val - hi); g.append(lo - val)
        for val, (lo, hi) in [(eg_x, EG_3D_X_WIN), (eg_s, EG_3D_S_WIN),
                              (eg_u, EG_3D_U_WIN), (eg_r, EG_3D_R_WIN),
                              (eg_t, EG_3D_T_WIN)]:
            g.append(val - hi); g.append(lo - val)

        if ENABLE_SANITY:
            # ── 3D sanity (21) ──────────────────────────────────────
            g.append(-eg_ga)                       # gap > 0  → −gap ≤ 0
            for d in d2v_3d: g.append(d)          # d²E_v/dk² < 0  →  d ≤ 0 violated if positive
            for d in d2c_3d: g.append(-d)         # d²E_c/dk² > 0  →  −d ≤ 0
            for d in dvbm:   g.append(d)          # VBM(k) ≤ VBM(Γ)  →  d ≤ 0
            for d in dcbm:   g.append(-d)         # CBM(k) ≥ CBM(Γ)  →  −d ≤ 0

    if FIT_MODE in ('joint', '2D'):
        # ── 2D physics windows (24) ─────────────────────────────────
        for i in range(5):
            g.append(eg_2d[i] - EG_2D_WIN[i, 1])
            g.append(EG_2D_WIN[i, 0] - eg_2d[i])
        for i in range(5):
            g.append(mu_2d[i] - MU_2D_WIN[i, 1])
            g.append(MU_2D_WIN[i, 0] - mu_2d[i])
        for i in range(4):
            g.append(eg_2d[i+1] - eg_2d[i])

        if ENABLE_SANITY:
            # ── 2D sanity (25) ──────────────────────────────────────
            for i in range(5):
                g.append(-eg_2d[i])                      # 2D gap > 0
            for i in range(5):
                g.append(d2v_2d[i, 0])                   # mv_e1 < 0
                g.append(d2v_2d[i, 1])                   # mv_e2 < 0
            for i in range(5):
                g.append(-d2c_2d[i, 0])                  # mc_e1 > 0
                g.append(-d2c_2d[i, 1])                  # mc_e2 > 0
            # ── confinement μ monotonicity (4) ──────────────────────
            for i in range(4):
                g.append(mu_2d[i+1] - mu_2d[i])

    # NOTE: parameter-basin constraints (orbital ordering, Vss coupling) are
    # appended by _evaluate after this function returns. So the expected size
    # of `g` here is N_CONSTR - _N_PARAM_SANITY (when sanity is on).
    expected = N_CONSTR - (_N_PARAM_SANITY if ENABLE_SANITY else 0)
    assert len(g) == expected, f"_build_G: {len(g)} ≠ {expected}"
    return g


# ── pymoo problem ─────────────────────────────────────────────────────────────

class RoundRepair(Repair):
    def _do(self, problem, X, **kwargs):
        return np.round(X, 2)


class TightBindingOptimization(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(n_var=N_VAR, n_obj=N_OBJ, n_constr=N_CONSTR,
                         xl=XL, xu=XU, **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.round(x, 2)
        try:
            (eg_ga, eg_y, eg_z,
             mv, mc, eg_2d, mu_2d,
             eg_x, eg_s, eg_u, eg_r, eg_t,
             dvbm, dcbm,
             d2v_3d, d2c_3d, d2v_2d, d2c_2d) = objectivesfunctions(x)
        except Exception:
            out['F'] = [1e6] * N_OBJ
            out['G'] = [1e6] * N_CONSTR
            return

        _e3d_terms = np.array([
            (eg_ga - EG_3D_GA)**2, (eg_y - EG_3D_Y)**2, (eg_z - EG_3D_Z)**2,
            (eg_x  - EG_3D_X)**2,  (eg_s - EG_3D_S)**2, (eg_u - EG_3D_U)**2,
            (eg_r  - EG_3D_R)**2,  (eg_t - EG_3D_T)**2,
        ])
        e3d = float(np.sum(W_EG_3D * _e3d_terms))
        disp3d = float(W_DISP_3D * (np.sum((dvbm - TARGET_DVBM)**2) +
                                    np.sum((dcbm - TARGET_DCBM)**2)))
        e3d = e3d + disp3d
        e2d = float(np.sum((eg_2d - TARGET_EG_2D)**2))
        m3d = (np.sum(((mv - TARGET_MV) * 10)**2) +
               np.sum(((mc - TARGET_MC) * 10)**2))
        m2d = float(np.sum(((mu_2d - TARGET_MU_2D) * 10)**2))
        o2d = float(10 * sum(max(0.0, eg_2d[i+1] - eg_2d[i])**2 for i in range(4)))

        if FIT_MODE == 'joint':
            energy_err = float(e3d + e2d); mass_err = float(m3d + m2d); order_err = o2d
        elif FIT_MODE == '3D':
            energy_err = float(e3d);       mass_err = float(m3d);       order_err = 0.0
        elif FIT_MODE == '2D':
            energy_err = e2d;              mass_err = m2d;              order_err = o2d

        disp_err = 0.0
        out['F'] = [energy_err, mass_err, order_err, disp_err]
        G = _build_G(eg_ga, eg_y, eg_z, mv, mc, eg_2d, mu_2d,
                     eg_x, eg_s, eg_u, eg_r, eg_t,
                     dvbm, dcbm,
                     d2v_3d, d2c_3d, d2v_2d, d2c_2d)
        if ENABLE_SANITY:
            G = G + _param_sanity_constraints(x)
        out['G'] = G


# ── callback ──────────────────────────────────────────────────────────────────

class PrintIdealCallback(Callback):
    def __init__(self, max_ideal_improvements=50, min_gen=50, require_feasible=True):
        super().__init__()
        self.prev_ideal = None
        self.ideal_count = 0
        self.max_ideal = max_ideal_improvements
        self.min_gen = min_gen
        self.require_feasible = require_feasible
        self.best_x = {i: None for i in range(N_OBJ)}
        self.best_f = {i: None for i in range(N_OBJ)}

    def notify(self, algorithm):
        F  = algorithm.pop.get('F')
        X  = algorithm.pop.get('X')
        CV = algorithm.pop.get('CV')
        if F is None or len(F) == 0:
            return
        gen = algorithm.n_gen

        if self.require_feasible and CV is not None:
            mask = (CV.flatten() <= 1e-12)
            if not mask.any():
                return
            F_eff = F[mask]; X_eff = X[mask]
        else:
            F_eff = F; X_eff = X

        cur_ideal = F_eff.min(axis=0)
        if self.prev_ideal is None:
            self.prev_ideal = cur_ideal.copy()
            return
        if not bool(np.any(cur_ideal < self.prev_ideal - 1e-12)):
            return

        self.ideal_count += 1
        for i_obj in range(N_OBJ):
            if cur_ideal[i_obj] >= self.prev_ideal[i_obj] - 1e-12:
                continue
            best_idx = int(np.argmin(F_eff[:, i_obj]))
            self.best_x[i_obj] = X_eff[best_idx].copy()
            self.best_f[i_obj] = F_eff[best_idx].copy()
            print(f'\n[Gen {gen}] *** new ideal for {OBJ_NAMES[i_obj]}: '
                  f'{F_eff[best_idx, i_obj]:.6f} ***  '
                  f'(improvement {self.ideal_count}/{self.max_ideal})')
            print(f'  F = [{", ".join(f"{v:.6f}" for v in F_eff[best_idx])}]')
            self._print_physics(X_eff[best_idx])
        self.prev_ideal = cur_ideal.copy()

        if self.ideal_count >= self.max_ideal and gen >= self.min_gen:
            print(f'\n[CALLBACK] {self.max_ideal} improvements at gen {gen} '
                  '— signalling termination.')
            try:    algorithm.termination.force_termination = True
            except AttributeError: pass
            try:    algorithm.has_terminated = True
            except AttributeError: pass
            if hasattr(algorithm.termination, 'n_max_gen'):
                algorithm.termination.n_max_gen = algorithm.n_gen

    @staticmethod
    def _print_physics(x):
        try:
            (eg_ga, eg_y, eg_z,
             mv, mc, eg_2d, mu_2d,
             eg_x, eg_s, eg_u, eg_r, eg_t,
             dvbm, dcbm,
             d2v_3d, d2c_3d, d2v_2d, d2c_2d) = objectivesfunctions(x)
            fmt4 = lambda v: f'{float(v):.4f}'
            arr4 = lambda a: [round(float(v), 4) for v in a]
            print(f'  eg  [Γ,Y,Z]    = [{fmt4(eg_ga)}, {fmt4(eg_y)}, {fmt4(eg_z)}]')
            print(f'      target     = [{EG_3D_GA:.4f}, {EG_3D_Y:.4f}, {EG_3D_Z:.4f}]')
            print(f'  eg  [X,S,U,R,T]= [{fmt4(eg_x)}, {fmt4(eg_s)}, {fmt4(eg_u)}, {fmt4(eg_r)}, {fmt4(eg_t)}]')
            print(f'      target     = [{EG_3D_X:.4f}, {EG_3D_S:.4f}, {EG_3D_U:.4f}, {EG_3D_R:.4f}, {EG_3D_T:.4f}]')
            print(f'  dVBM [Y,Z,X,S,U,R,T] = {arr4(dvbm)}')
            print(f'       target          = {arr4(TARGET_DVBM)}')
            print(f'  dCBM [Y,Z,X,S,U,R,T] = {arr4(dcbm)}')
            print(f'       target          = {arr4(TARGET_DCBM)}')
            print(f'  mv   [xx,yy,zz]= {[float(v) for v in np.round(mv, 4)]}')
            print(f'       target    = {TARGET_MV.tolist()}')
            print(f'  mc   [xx,yy,zz]= {[float(v) for v in np.round(mc, 4)]}')
            print(f'       target    = {TARGET_MC.tolist()}')
            print(f'  eg2D [n1..n5]  = {[float(v) for v in np.round(eg_2d, 4)]}')
            print(f'       target    = {TARGET_EG_2D.tolist()}')
            print(f'  mu2D [n1..n5]  = {[float(v) for v in np.round(mu_2d, 4)]}')
            print(f'       target    = {TARGET_MU_2D.tolist()}')
        except Exception as exc:
            print(f'  [warn] physics recompute failed: {exc}')
        print('  parameters:')
        for name, val in zip(PARAM_NAMES, x):
            print(f'    {name:14s} = {val:+8.4f}')


# ── single-seed worker ────────────────────────────────────────────────────────

def run_one_seed(args):
    seed, n_gen, pop_size, max_ideal, min_gen = args
    print(f'[seed={seed}] starting', flush=True)
    t0 = time.time()
    problem = TightBindingOptimization()

    np.random.seed(seed)
    sampling = np.random.uniform(problem.xl, problem.xu, (pop_size, N_VAR))
    if WARM_START is not None:
        sampling[0] = np.clip(WARM_START, problem.xl, problem.xu)

    algorithm = NSGA2(
        pop_size=pop_size,
        repair=RoundRepair(),
        n_offsprings=40,
        sampling=sampling,
        crossover=SimulatedBinaryCrossover(eta=20, prob=0.9),
        mutation=GaussianMutation(sigma=0.1, prob=0.2),
        eliminate_duplicates=True,
    )

    cb = PrintIdealCallback(
        max_ideal_improvements=max_ideal,
        min_gen=min_gen,
        require_feasible=False,
    )

    try:
        res = minimize(
            problem, algorithm, get_termination('n_gen', n_gen),
            seed=seed, save_history=False, verbose=False, callback=cb,
        )
    except Exception as e:
        print(f'[seed={seed}] FAILED: {e}', flush=True)
        return {'seed': seed, 'elapsed': time.time() - t0,
                'best_x': None, 'best_f': None, 'ideal_count': 0, 'error': str(e)}

    elapsed = time.time() - t0
    best_x, best_f, best_score = None, None, np.inf
    for i_obj in range(N_OBJ):
        if cb.best_x[i_obj] is not None:
            f = cb.best_f[i_obj]
            score = f[0] + f[1]
            if score < best_score:
                best_score = score; best_x = cb.best_x[i_obj]; best_f = cb.best_f[i_obj]

    if best_x is None and res.X is not None:
        X_all = res.X if res.X.ndim == 2 else res.X[np.newaxis, :]
        F_all = res.F if res.F.ndim == 2 else res.F[np.newaxis, :]
        best_idx = int(np.argmin(F_all[:, 0] + F_all[:, 1]))
        best_x = X_all[best_idx]; best_f = F_all[best_idx]

    obs = None
    if best_x is not None:
        try:    obs = objectivesfunctions(best_x)
        except Exception as e: print(f'[seed={seed}] verify failed: {e}', flush=True)

    print(f'[seed={seed}] done in {elapsed:.1f}s  '
          f'F={None if best_f is None else np.round(best_f, 4).tolist()}  '
          f'ideal={cb.ideal_count}', flush=True)

    return {'seed': seed, 'elapsed': elapsed,
            'best_x': best_x, 'best_f': best_f,
            'obs': obs, 'ideal_count': cb.ideal_count, 'error': None}


# ── main driver ───────────────────────────────────────────────────────────────

def run_fit():
    """Run the NSGA-II optimisation with the current module-level config.

    Any caller can override module constants (W_EG_3D, WARM_START, SEEDS,
    N_GEN, POP_SIZE, OUTPUT_FILE, …) BEFORE calling this function to
    customise the run."""
    print('=' * 70)
    print(f'fit_ortho_sanity.py: {len(SEEDS)} seeds × {N_GEN} gens × pop {POP_SIZE}')
    n_proc = min(len(SEEDS), os.cpu_count() or 2)
    print(f'Running on {n_proc} parallel processes')
    print(f'Seeds: {SEEDS}')
    n_free = int(np.sum(XL != XU))
    print(f'FIT_MODE={FIT_MODE!r}  sanity={ENABLE_SANITY}  '
          f'free params={n_free}/{N_VAR}  n_constr={N_CONSTR}')
    print('=' * 70)

    t_start   = time.time()
    args_list = [(s, N_GEN, POP_SIZE, MAX_IDEAL, MIN_GEN) for s in SEEDS]
    with mp.Pool(n_proc) as pool:
        results = pool.map(run_one_seed, args_list)
    total = time.time() - t_start
    serial_total = sum(r['elapsed'] for r in results)
    print(f'\nAll seeds finished in {total:.1f}s  '
          f'(speedup ≈ {serial_total/total:.1f}× over serial)')

    good = [r for r in results if r['best_x'] is not None]
    if not good:
        print('\nNo seed produced a usable solution.')
        for r in results:
            print(f'  seed {r["seed"]}: {r.get("error")}')
        raise SystemExit(1)

    good.sort(key=lambda r: float(np.sum(r['best_f'])))

    print('\n' + '=' * 78)
    print('SUMMARY  (sorted by sum of all objectives)')
    print('=' * 78)
    header = f'{"seed":>6}  ' + '  '.join(f'{n:>11}' for n in OBJ_NAMES) + f'  {"sum":>9}  {"ideal":>5}  {"t(s)":>7}'
    print(header)
    for r in good:
        f = r['best_f']
        row = f'{r["seed"]:>6}  ' + '  '.join(f'{v:>11.4f}' for v in f)
        row += f'  {sum(f):>9.4f}  {r["ideal_count"]:>5}  {r["elapsed"]:>7.1f}'
        print(row)
    for r in results:
        if r['best_x'] is None:
            print(f'  seed {r["seed"]}: FAILED — {r.get("error")}')

    best = next((r for r in good if r['obs'] is not None), None)
    if best is None:
        print('\nAll seeds had verify failures.')
        raise SystemExit(1)

    obs = best['obs']
    (eg_ga, eg_y, eg_z,
     mv, mc, eg_2d, mu_2d,
     eg_x, eg_s, eg_u, eg_r, eg_t,
     dvbm, dcbm,
     d2v_3d, d2c_3d, d2v_2d, d2c_2d) = obs

    print('\n' + '=' * 70)
    print(f'BEST RESULT: seed = {best["seed"]}')
    print('=' * 70)
    print(f'F = {np.round(best["best_f"], 6).tolist()}   {OBJ_NAMES}')
    print(f'eg  [Γ,Y,Z]     = {[round(eg_ga,4), round(eg_y,4), round(eg_z,4)]}')
    print(f'    target      = [{EG_3D_GA}, {EG_3D_Y}, {EG_3D_Z}]')
    print(f'eg  [X,S,U,R,T] = {[round(eg_x,4), round(eg_s,4), round(eg_u,4), round(eg_r,4), round(eg_t,4)]}')
    print(f'    target      = [{EG_3D_X}, {EG_3D_S}, {EG_3D_U}, {EG_3D_R}, {EG_3D_T}]')
    print(f'dVBM [Y,Z,X,S,U,R,T] = {np.round(dvbm, 4).tolist()}')
    print(f'     target          = {TARGET_DVBM.tolist()}')
    print(f'dCBM [Y,Z,X,S,U,R,T] = {np.round(dcbm, 4).tolist()}')
    print(f'     target          = {TARGET_DCBM.tolist()}')
    print(f'mv   [xx,yy,zz] = {np.round(mv, 4).tolist()}   target={TARGET_MV.tolist()}')
    print(f'mc   [xx,yy,zz] = {np.round(mc, 4).tolist()}   target={TARGET_MC.tolist()}')
    print(f'eg2D [n1..n5]   = {np.round(eg_2d, 4).tolist()}')
    print(f'     target     = {TARGET_EG_2D.tolist()}')
    print(f'mu2D [n1..n5]   = {np.round(mu_2d, 4).tolist()}')
    print(f'     target     = {TARGET_MU_2D.tolist()}')
    print('\nparameters:')
    for name, val in zip(PARAM_NAMES, best['best_x']):
        print(f'  {name:14s} = {val:+8.4f}')

    np.savez(
        OUTPUT_FILE,
        best_x=best['best_x'],
        best_f=best['best_f'],
        best_seed=np.array(best['seed']),
        eg_3d=np.array([eg_ga, eg_y, eg_z]),
        eg_2d=eg_2d,
        mv=mv, mc=mc, mu_2d=mu_2d,
        all_seeds=np.array([r['seed'] for r in results]),
        all_F=np.array([r['best_f'] if r['best_f'] is not None
                        else np.full(N_OBJ, np.nan) for r in results]),
    )
    print(f'\nSaved to {OUTPUT_FILE}')


if __name__ == '__main__':
    run_fit()
