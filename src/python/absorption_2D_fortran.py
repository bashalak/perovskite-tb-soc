"""
Optical absorption for 3D bulk and 2D Ruddlesden-Popper perovskite slabs (n=1-5).

Formula (Boyer-Richard et al., J. Phys. Chem. Lett. 7, 3833 (2016)):

    α(ω) [cm⁻¹] = (πe²/ε₀ℏc) × 10⁸ / (a[Å]³ × n_r × ω[eV] × N_k)
                   × Σ_{cv,k} |⟨v,k|∂H/∂k_x|c,k⟩|² × G(E_c−E_v−ω, η)

Key points:
  - H(k) and diagonalization via ctypesroutines Fortran backend (no sisl)
  - ∂H/∂k_x by central finite differences of H(k)
  - 3D: full 3D k-grid, 19 bulk parameters (no IIz, no surface corrections)
  - 2D: k_z = 0 grid, 26 parameters with surface corrections
  - x-polarization only (consistent with Boyer-Richard Fortran)

Run:
  source ~/TBT-TS-sisl-workshop/setup.sh
  cd /home/baker/newparam5
  python3 /mnt/c/Users/Bakr/AppData/Local/Temp/absorption_2D_fortran.py
"""

import sys, os
sys.path.insert(0, '/home/baker/newparam5')
os.chdir('/home/baker/newparam5')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from ctypesroutines import (
    hamiltonian_cubic3D_fortran,
    hamiltonian_cubic2D_apical_fortran,
    diagonalize_hamiltonian_fortran,
)

# ── Best-fit 26 parameters (from newparam4/convergence_plot.py) ───────────────
PARAMS = np.array([
    -2.75, +3.42, -9.42, +2.58, +1.95, +0.74, -3.64, -1.86, -0.90, +1.30, +0.90,
    -0.19, -0.59, +0.09, +0.27,
    -0.49, +0.01, +0.27, +0.20,
     0.0,   0.0,   0.0,   0.0,
    -0.16,  3.60,  3.52,
])

#PARAMS = np.array([
#    -9.01, 2.34  ,-13.01,-1.96 ,-1.10 ,1.19  ,0.70  ,-3.65 ,0.55  , +1.30, +0.90,
#    0.0,   0.0,   0.0,   0.0,
#    0.0,   0.0,   0.0,   0.0,
#     0.0,   0.0,   0.0,   0.0,
#    0.0,   0.0,   0.0,
#])
(Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
 deltaSO0, deltaSO1,
 Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
 Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
 Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
 Es1_ap, Ep1_ap, Ep0_surf) = PARAMS

# ── Physical / numerical settings ────────────────────────────────────────────
A_LAT    = 6.28    # lattice constant [Å]
N_REFRAC = 2.5     # refractive index
ETA      = 0.01    # Gaussian broadening [eV]  (use 0.10 for smoother curves)
C_ABS    = 0.288   # πe²/(ε₀ℏc)
DK       = 1e-5    # finite-difference step [Å⁻¹]

OMEGA_MIN = 1.4    # spectrum start [eV]
OMEGA_MAX = 4.0    # spectrum end   [eV]
N_OMEGA   = 6000   # energy grid points

N_K    = 50        # k-points per in-plane direction  (N_K²  total for 2D)
N_K_3D = 50        # k-points per direction for 3D   (N_K_3D³ total; 20³=8000 pts)
N_VB   = np.array([2, 2, 2, 2, 2, 2])         # VB bands below and including IVB  (matches Boyer-Richard: 2 VB)
N_CB   = np.array([6, 6, 12, 18, 24, 30])         # CB bands above and including ICB  (matches Boyer-Richard: 6 CB)

# ── 3D bulk band indices (0-based, 32×32 matrix) ─────────────────────────────
IVB_3D = 25        # top valence band  (band 26 in 1-based)
ICB_3D = 26        # bottom conduction (band 27 in 1-based)

# 3D uses 19 bulk parameters: no IIz terms, no surface corrections
P3D = [Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
       deltaSO0, deltaSO1,
       Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
       Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb]

# ── 2D slab definitions: IVB, ICB (0-based), matrix size ─────────────────────
SLABS = {
    1: (33,  34,  40),
    2: (59,  60,  72),
    3: (85,  86, 104),
    4: (111, 112, 136),
    5: (137, 138, 168),
}

# ── 3D Hamiltonian helpers ────────────────────────────────────────────────────
def _ham_3d(kx, ky, kz):
    return hamiltonian_cubic3D_fortran(kx, ky, kz, *P3D)

def _dHdkx_3d(kx, ky, kz):
    """∂H/∂kx [eV·Å] by central finite differences."""
    return (_ham_3d(kx + DK, ky, kz) - _ham_3d(kx - DK, ky, kz)) / (2.0 * DK)

def _kpoint_pairs_3d(kx, ky, kz):
    """Return list of (|Mx|², ΔE) for 3D bulk at one k-point."""
    eigvals, eigvecs = diagonalize_hamiltonian_fortran(_ham_3d(kx, ky, kz))
    dh = _dHdkx_3d(kx, ky, kz)

    vb_range = range(IVB_3D - N_VB[0] + 1, IVB_3D + 1)
    cb_range = range(ICB_3D, ICB_3D + N_CB[0])

    pairs = []
    for ic in cb_range:
        psi_c    = eigvecs[:, ic]
        dh_psi_c = dh @ psi_c
        for iv in vb_range:
            psi_v = eigvecs[:, iv]
            Mx  = np.dot(psi_v.conj(), dh_psi_c)
            Mx2 = float(np.real(Mx * Mx.conj()))
            dE  = float(eigvals[ic] - eigvals[iv])
            pairs.append((Mx2, dE))
    return pairs

# ── 2D Hamiltonian helpers ────────────────────────────────────────────────────
def _ham(n_layers, kx, ky, kz=0.0):
    return hamiltonian_cubic2D_apical_fortran(
        n_layers, kx, ky, kz,
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        Es1_ap, Ep1_ap, Ep0_surf,
    )

def _dHdkx(n_layers, kx, ky):
    """∂H/∂kx [eV·Å] by central finite differences (k_z = 0)."""
    return (_ham(n_layers, kx + DK, ky) - _ham(n_layers, kx - DK, ky)) / (2.0 * DK)

# ── Per-k-point computation ───────────────────────────────────────────────────
def _kpoint_pairs(n_layers, kx, ky, ivb, icb):
    """
    Return list of (|Mx|², ΔE) for all relevant v→c transitions.
    Eigenvectors come from LAPACK ZHEEV: columns of the returned matrix.
    """
    eigvals, eigvecs = diagonalize_hamiltonian_fortran(_ham(n_layers, kx, ky))
    dh = _dHdkx(n_layers, kx, ky)

    vb_range = range(ivb - N_VB[n_layers] + 1, ivb + 1)   # [ivb-1, ivb] for N_VB=2
    cb_range = range(icb, icb + N_CB[n_layers])            # [icb .. icb+5] for N_CB=6

    pairs = []
    for ic in cb_range:
        psi_c = eigvecs[:, ic]
        dh_psi_c = dh @ psi_c          # ∂H/∂kx · |c,k⟩  [eV·Å]
        for iv in vb_range:
            psi_v = eigvecs[:, iv]
            Mx  = np.dot(psi_v.conj(), dh_psi_c)   # ⟨v,k|∂H/∂kx|c,k⟩
            Mx2 = float(np.real(Mx * Mx.conj()))    # |Mx|² [(eV·Å)²]
            dE  = float(eigvals[ic] - eigvals[iv])  # transition energy [eV]
            pairs.append((Mx2, dE))
    return pairs

# ── Gaussian broadening ───────────────────────────────────────────────────────
def _gaussian(omega, dE_arr, eta):
    return (1.0 / (eta * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((omega - dE_arr) / eta) ** 2)

# ── Full absorption spectrum: 3D bulk ────────────────────────────────────────
def compute_absorption_3d():
    kpts = np.linspace(0.0, 2.0 * np.pi / A_LAT, N_K_3D, endpoint=False)
    Nk   = N_K_3D ** 3

    print(f"  3D bulk: {N_K_3D}³ = {Nk} k-points, matrix 32×32")
    print(f"           VB bands [{IVB_3D-N_VB[0]+1}..{IVB_3D}], CB bands [{ICB_3D}..{ICB_3D+N_CB[0]-1}]")

    all_results = Parallel(n_jobs=-1)(
        delayed(_kpoint_pairs_3d)(kx, ky, kz)
        for kx in kpts for ky in kpts for kz in kpts
    )

    all_Mx2 = np.array([p[0] for pairs in all_results for p in pairs])
    all_dE  = np.array([p[1] for pairs in all_results for p in pairs])

    energies  = np.linspace(OMEGA_MIN, OMEGA_MAX, N_OMEGA)
    prefactor = C_ABS * 1.0e8 / (A_LAT**3 * N_REFRAC * Nk)
    alphas    = np.zeros(N_OMEGA)

    for l, omega in enumerate(energies):
        G = _gaussian(omega, all_dE, ETA)
        alphas[l] = prefactor / omega * np.dot(all_Mx2, G)

    return energies, alphas

# ── Full absorption spectrum for one slab ────────────────────────────────────
def compute_absorption_2d(n_layers):
    ivb, icb, n_mat = SLABS[n_layers]
    kpts = np.linspace(0.0, 2.0 * np.pi / A_LAT, N_K, endpoint=False)
    Nk   = N_K ** 2  # 2D BZ

    print(f"  n={n_layers}: {N_K}² = {Nk} k-points, matrix {n_mat}×{n_mat}")
    print(f"           VB bands [{ivb-N_VB[n_layers]+1}..{ivb}], CB bands [{icb}..{icb+N_CB[n_layers]-1}]")

    # Step 1 — collect all transition pairs in parallel over k-points
    all_results = Parallel(n_jobs=-1)(
        delayed(_kpoint_pairs)(n_layers, kx, ky, ivb, icb)
        for kx in kpts for ky in kpts
    )

    all_Mx2 = np.array([p[0] for pairs in all_results for p in pairs])
    all_dE  = np.array([p[1] for pairs in all_results for p in pairs])

    # Step 2 — vectorised sum over ω
    energies   = np.linspace(OMEGA_MIN, OMEGA_MAX, N_OMEGA)
    prefactor  = C_ABS * 1.0e8 / (A_LAT**3 * N_REFRAC * Nk)
    alphas     = np.zeros(N_OMEGA)

    for l, omega in enumerate(energies):
        G = _gaussian(omega, all_dE, ETA)
        alphas[l] = prefactor / omega * np.dot(all_Mx2, G)

    return energies, alphas

# ── Run ───────────────────────────────────────────────────────────────────────
import time

results    = {}   # key: n = 1..5 for 2D, 'bulk' for 3D
results_3d = None

# 3D bulk
print("\nComputing 3D bulk absorption ...")
t0 = time.time()
e3d, a3d = compute_absorption_3d()
print(f"  Done in {time.time()-t0:.1f} s")
results_3d = (e3d, a3d)
np.savetxt(
    'absorption_3D_bulk.dat',
    np.column_stack([e3d, a3d]),
    header=f'omega_eV  alpha_cm-1  (3D bulk, a={A_LAT} A, eta={ETA} eV, {N_K_3D}^3 k-pts)',
    fmt='%.6f  %.8e',
)
print("  Saved absorption_3D_bulk.dat")

# 2D slabs
for n in [1, 2, 3, 4, 5]:
    print(f"\nComputing 2D n={n} absorption ...")
    t0 = time.time()
    energies, alphas = compute_absorption_2d(n)
    print(f"  Done in {time.time()-t0:.1f} s")

    results[n] = (energies, alphas)
    np.savetxt(
        f'absorption_2D_n{n}.dat',
        np.column_stack([energies, alphas]),
        header=f'omega_eV  alpha_cm-1  (2D slab n={n}, a={A_LAT} A, eta={ETA} eV, {N_K}^2 k-pts)',
        fmt='%.6f  %.8e',
    )
    print(f"  Saved absorption_2D_n{n}.dat")



# ── DFT dielectric function (George) → absorption α ∝ E·κ ───────────────────
# κ(E) = Im(sqrt(ε_r + i ε_i)) = sqrt((sqrt(ε_r² + ε_i²) − ε_r) / 2)
_DFT_XEPSI = np.array([
    1.0669642857142858, 1.1540178571428572, 1.2477678571428572, 1.3683035714285714,
    1.4620535714285714, 1.5558035714285714, 1.5825892857142856, 1.609375,
    1.6361607142857144, 1.6763392857142856, 1.7366071428571428, 1.8303571428571428,
    1.9107142857142856, 1.9977678571428572, 2.084821428571429, 2.1383928571428568,
    2.1919642857142856, 2.2857142857142856, 2.366071428571429, 2.4330357142857144,
    2.5267857142857144, 2.59375, 2.7008928571428568, 2.8013392857142856,
    2.8883928571428568, 2.9888392857142856, 3.075892857142857, 3.149553571428571,
    3.216517857142857, 3.2566964285714284, 3.323660714285714, 3.404017857142857,
    3.450892857142857, 3.497767857142857, 3.53125, 3.571428571428571,
    3.618303571428571, 3.665178571428571, 3.71875, 3.799107142857143,
    3.8727678571428568, 3.9598214285714284, 4.040178571428571, 4.133928571428571,
    4.2611607142857135, 4.4017857142857135, 4.5424107142857135, 4.662946428571429,
    4.796875, 4.924107142857142, 5.051339285714286, 5.185267857142857,
    5.272321428571428, 5.379464285714286, 5.493303571428571, 5.580357142857142,
    5.667410714285714, 5.747767857142857, 5.848214285714286, 5.948660714285714,
    6.049107142857142, 6.169642857142857, 6.283482142857142, 6.404017857142857,
    6.484375,
])
_DFT_YEPSI = np.array([
    1, 1, 1, 1, 1.0098199672667758, 1.039279869067103, 1.1767594108019641,
    1.579378068739771, 1.7954173486088378, 1.9230769230769231, 1.9623567921440261,
    2.0016366612111294, 2.07037643207856, 2.198036006546645, 2.3944353518821604,
    2.5711947626841245, 2.797054009819967, 3.258592471358429, 3.729950900163666,
    4.152209492635024, 4.5646481178396066, 4.6923076923076925, 4.741407528641571,
    4.819967266775778, 5.016366612111293, 5.399345335515548, 5.870703764320786,
    6.263502454991817, 6.548281505728315, 6.695581014729951, 6.754500818330605,
    6.646481178396072, 6.48936170212766, 6.283142389525368, 6.096563011456628,
    5.880523731587561, 5.654664484451718, 5.369885433715221, 5.094926350245499,
    4.731587561374795, 4.456628477905074, 4.201309328968904, 3.9950900163666123,
    3.8477905073649756, 3.729950900163666, 3.700490998363339, 3.7397708674304417,
    3.7397708674304417, 3.6612111292962357, 3.5335515548281506, 3.4157119476268414,
    3.337152209492635, 3.327332242225859, 3.376432078559738, 3.4942716857610474,
    3.602291325695581, 3.6710310965630115, 3.6219312602291325, 3.513911620294599,
    3.4451718494271684, 3.4058919803600656, 3.4157119476268414, 3.46481178396072,
    3.5335515548281506, 3.592471358428805,
]) - 1.0
_DFT_XEPSR = np.array([
    1.0066964285714286, 1.1004464285714286, 1.2075892857142856, 1.3013392857142856,
    1.4017857142857142, 1.46875, 1.5357142857142856, 1.5691964285714284,
    1.6026785714285714, 1.65625, 1.6897321428571428, 1.7700892857142856,
    1.890625, 1.9977678571428572, 2.078125, 2.178571428571429, 2.252232142857143,
    2.3125, 2.392857142857143, 2.439732142857143, 2.4799107142857144,
    2.5200892857142856, 2.5602678571428568, 2.6138392857142856, 2.674107142857143,
    2.7477678571428568, 2.8415178571428568, 2.9352678571428568, 3.015625,
    3.0691964285714284, 3.109375, 3.149553571428571, 3.183035714285714,
    3.216517857142857, 3.236607142857143, 3.263392857142857, 3.290178571428571,
    3.323660714285714, 3.34375, 3.377232142857143, 3.417410714285714,
    3.450892857142857, 3.497767857142857, 3.544642857142857, 3.604910714285714,
    3.6919642857142856, 3.7857142857142856, 3.939732142857143, 4.080357142857142,
    4.200892857142857, 4.3080357142857135, 4.415178571428571, 4.488839285714286,
    4.582589285714286, 4.723214285714286, 4.84375, 4.944196428571429,
    5.091517857142857, 5.205357142857142, 5.332589285714286, 5.433035714285714,
    5.5, 5.566964285714286, 5.633928571428571, 5.700892857142857,
    5.767857142857142, 5.868303571428571, 5.995535714285714, 6.116071428571428,
    6.209821428571428, 6.316964285714286, 6.417410714285714, 6.484375,
])
_DFT_YEPSR = np.array([
    5.026186579378069, 5.114566284779051, 5.242225859247136, 5.389525368248773,
    5.585924713584288, 5.762684124386252, 6.027823240589198, 6.30278232405892,
    6.528641571194763, 6.332242225859247, 6.18494271685761, 6.1063829787234045,
    6.204582651391162, 6.391162029459902, 6.577741407528642, 6.783960720130933,
    6.901800327332242, 6.941080196399345, 6.872340425531915, 6.725040916530278,
    6.577741407528642, 6.391162029459902, 6.224222585924713, 6.018003273322423,
    5.801963993453355, 5.693944353518821, 5.6644844517184945, 5.6644844517184945,
    5.576104746317513, 5.438625204582651, 5.281505728314239, 5.094926350245499,
    4.869067103109656, 4.682487725040916, 4.466448445171849, 4.260229132569558,
    4.0343698854337156, 3.7986906710310966, 3.5433715220949265, 3.2880523731587563,
    3.0327332242225857, 2.7381342062193124, 2.414075286415712, 2.148936170212766,
    1.9132569558101473, 1.7463175122749592, 1.6677577741407528, 1.7659574468085106,
    1.9427168576104745, 2.099836333878887, 2.227495908346972, 2.3060556464811786,
    2.32569558101473, 2.315875613747954, 2.227495908346972, 2.1783960720130935,
    2.1685761047463172, 2.237315875613748, 2.345335515548282, 2.4631751227495906,
    2.551554828150573, 2.5711947626841245, 2.5319148936170213, 2.4631751227495906,
    2.364975450081833, 2.2962356792144023, 2.276595744680851, 2.3355155482815055,
    2.414075286415712, 2.4828150572831422, 2.541734860883797, 2.600654664484452,
    2.6104746317512273,
])

# Interpolate onto fine grid and compute κ = Im(sqrt(ε_r + i ε_i))
_E_dft  = np.linspace(max(_DFT_XEPSI.min(), _DFT_XEPSR.min()),
                       min(_DFT_XEPSI.max(), _DFT_XEPSR.max()), 800)
_eps_i  = np.maximum(np.interp(_E_dft, _DFT_XEPSI, _DFT_YEPSI), 0.0)
_eps_r  = np.interp(_E_dft, _DFT_XEPSR, _DFT_YEPSR)
_kappa  = np.sqrt(np.maximum((np.sqrt(_eps_r**2 + _eps_i**2) - _eps_r) / 2.0, 0.0))
#_alpha_dft = _E_dft * _kappa   # ∝ ω × κ  (arbitrary units)
_hbarc = 1.9732e-5          # eV·cm
_alpha_dft = (2 * _E_dft / _hbarc) * _kappa   # cm⁻¹
# Scale DFT curve to match 3D bulk TB peak in the plot window
_mask_plot  = (_E_dft >= OMEGA_MIN) & (_E_dft <= OMEGA_MAX)
_mask_3d    = (e3d >= OMEGA_MIN) & (e3d <= OMEGA_MAX)
_scale_dft  = a3d[_mask_3d].max() / _alpha_dft[_mask_plot].max()
_alpha_dft_scaled = _alpha_dft * _scale_dft

# ── Plot: 3D bulk + 2D n=1-5 ─────────────────────────────────────────────────
COLORS_2D = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, ax = plt.subplots(figsize=(9, 5))

# 3D bulk as thick dashed black line
ax.plot(e3d, a3d, color='black', lw=2.0, ls='--', label='3D bulk (TB)', zorder=10)
fortran_data = np.loadtxt('simon1.txt')
ax.plot(fortran_data[:, 0], fortran_data[:, 1], 'r--', lw=1.5,
        label='Fortran Soline model', alpha=0.8)
# DFT dielectric function → α ∝ E·κ  (scaled to TB 3D bulk peak)
ax.plot(_E_dft[_mask_plot], _alpha_dft_scaled[_mask_plot],
        color='darkorange', lw=2.0, ls='-.', label='DFT  α∝E·κ (George)', zorder=9)
ax.legend(fontsize=12)
# 2D slabs
for i, n in enumerate([1, 2, 3, 4, 5]):
    e, a = results[n]
    ax.plot(e, a, color=COLORS_2D[i], lw=1.5, label=f'2D n = {n}')

ax.set_xlabel('Photon energy (eV)', fontsize=13)
ax.set_ylabel(r'$\alpha$ (cm$^{-1}$)', fontsize=13)
ax.set_title(
    f'Optical absorption: 3D bulk and 2D slabs (SP³ TB+SOC)\n'
    f'a = {A_LAT} Å, η = {ETA} eV, x-polarisation',
    fontsize=11,
)
ax.set_xlim(OMEGA_MIN, OMEGA_MAX)
ax.set_yscale('log')
ax.set_ylim(bottom=1)
ax.legend(fontsize=11, framealpha=0.85)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig('absorption_3D_and_2D.png', dpi=150, bbox_inches='tight')
print('\nSaved absorption_3D_and_2D.png')
print('All done.')
