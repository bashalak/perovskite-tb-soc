"""
Full effective mass tensor: in-plane (x,y) and out-of-plane (z) components.

For 3D bulk: compute mh and me along kx, ky, kz at R, M, X.
For 2D slabs: in-plane mass vs n; out-of-plane mass = ∞ (quantum confinement).
Also shows the anisotropy ratio m_z / m_xy as a function of k-point.

Run:  source ~/TBT-TS-sisl-workshop/setup.sh && python3 effective_mass_tensor.py

Index convention returned by effective_mass_*_fortran:
  inv_mass[0] = 1/m_kx   (in-plane)
  inv_mass[1] = 1/m_ky   (in-plane)
  inv_mass[2] = 1/m_kz   (out-of-plane / stacking direction)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ctypesroutines import (
    hamiltonian_cubic3D_fortran,
    hamiltonian_cubic2D_apical_fortran,
    effective_mass_3d_fortran,
    effective_mass_2d_apical_fortran,
    eigenvalues_only_fortran,
)

# ---- best-fit parameters ----
PARAMS = np.array([
    -2.75, +3.42, -9.42, +2.58, +1.95, +0.74, -3.64, -1.86, -0.90, +1.30, +0.90,
    -0.19, -0.59, +0.09, +0.27,
    -0.49, +0.01, +0.27, +0.20,
     0.0,   0.0,   0.0,   0.0,
    -0.16,  3.60,  3.52,
])
(Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
 deltaSO0, deltaSO1,
 Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
 Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
 Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
 Es1_ap, Ep1_ap, Ep0_surf) = PARAMS

PARAMS_3D = [Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
             deltaSO0, deltaSO1,
             Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
             Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb]

A_LAT  = 6.28
EPS_EM = 0.05
KR  = np.array([np.pi/A_LAT, np.pi/A_LAT, np.pi/A_LAT])
KM  = np.array([np.pi/A_LAT, np.pi/A_LAT, 0.0])
KX  = np.array([np.pi/A_LAT, 0.0,         0.0])
IVB_3D = 25; ICB_3D = 26
IVB_2D = [33, 59, 85, 111, 137]
ICB_2D = [34, 60, 86, 112, 138]


def _em3d(bidx, k):
    return effective_mass_3d_fortran(k[0], k[1], k[2], bidx, 32,
                                     A_LAT, A_LAT, A_LAT, EPS_EM, *PARAMS_3D)

def _em2d(nL, bidx, k):
    return effective_mass_2d_apical_fortran(
        nL, k[0], k[1], k[2], bidx,
        A_LAT, A_LAT, A_LAT, EPS_EM,
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        Es1_ap, Ep1_ap, Ep0_surf)


# ============================================================
# 3D bulk: full mass tensor at R, M, X
# ============================================================
print("=" * 65)
print("3D BULK — effective mass tensor (m₀)")
print("=" * 65)
print(f"{'k-point':<8} {'band':<6} {'m_kx':>8} {'m_ky':>8} {'m_kz':>8}  "
      f"{'m_xy (avg)':>11}  {'anisotropy kz/kx':>18}")
print("-" * 65)

kpoints = [("R", KR), ("M", KM), ("X", KX)]
for kname, k in kpoints:
    for bname, bidx in [("VBM", IVB_3D), ("CBM", ICB_3D)]:
        inv = _em3d(bidx, k)
        mx  = 1.0 / inv[0]
        my  = 1.0 / inv[1]
        mz  = 1.0 / inv[2]
        mxy = 2.0 / (1.0/mx + 1.0/my) if (inv[0] != 0 and inv[1] != 0) else float('nan')
        ani = abs(mz / mx) if (inv[0] != 0 and inv[2] != 0) else float('nan')
        print(f"  {kname:<6} {bname:<6} {mx:>8.4f} {my:>8.4f} {mz:>8.4f}  "
              f"{mxy:>11.4f}  {ani:>18.3f}")
    print()

# Reduced masses (as currently used in fitting)
print("Reduced masses μ = |me·mh|/(|me|+|mh|) :")
print(f"{'k-point':<8} {'μ_xy (fitted)':>14} {'μ_z (out-of-plane)':>20}")
print("-" * 45)
for kname, k in kpoints:
    inv_v = _em3d(IVB_3D, k); inv_c = _em3d(ICB_3D, k)
    mh_xy = 2.0 / (inv_v[0] + inv_v[1])   # harmonic mean in-plane
    me_xy = 2.0 / (inv_c[0] + inv_c[1])
    mh_z  = 1.0 / inv_v[2] if inv_v[2] != 0 else float('inf')
    me_z  = 1.0 / inv_c[2] if inv_c[2] != 0 else float('inf')
    mu_xy = abs(me_xy * mh_xy) / (abs(me_xy) + abs(mh_xy))
    if np.isfinite(mh_z) and np.isfinite(me_z) and inv_v[2] != 0 and inv_c[2] != 0:
        mu_z = abs(me_z * mh_z) / (abs(me_z) + abs(mh_z))
        muz_s = f"{mu_z:>20.4f}"
    else:
        muz_s = f"{'∞ (confined)':>20}"
    print(f"  {kname:<6} {mu_xy:>14.4f} {muz_s}")

# ============================================================
# 2D slabs: in-plane and out-of-plane
# ============================================================
print()
print("=" * 65)
print("2D SLABS — effective masses at M-point (m₀)")
print("=" * 65)
print(f"{'n':<4} {'mh_x':>8} {'mh_y':>8} {'mh_z':>12} {'me_x':>8} {'me_y':>8} {'me_z':>12}  {'μ_xy':>8}")
print("-" * 75)

mu_in  = []
mh_in  = []
me_in  = []
mh_out = []
me_out = []

for i, nL in enumerate(range(1, 6)):
    inv_v = _em2d(nL, IVB_2D[i], KM)
    inv_c = _em2d(nL, ICB_2D[i], KM)
    mhx = 1.0 / inv_v[0]; mhy = 1.0 / inv_v[1]
    mex = 1.0 / inv_c[0]; mey = 1.0 / inv_c[1]
    mhz_s = "∞ (confined)" if inv_v[2] == 0 else f"{1.0/inv_v[2]:.4f}"
    mez_s = "∞ (confined)" if inv_c[2] == 0 else f"{1.0/inv_c[2]:.4f}"
    mu_xy = abs(mex * mhx) / (abs(mex) + abs(mhx))
    mh_in.append(mhx); me_in.append(mex); mu_in.append(mu_xy)
    print(f"  {nL:<2}  {mhx:>8.4f} {mhy:>8.4f} {mhz_s:>12} {mex:>8.4f} {mey:>8.4f} {mez_s:>12}  {mu_xy:>8.4f}")

print()
print("Note: m_z = ∞ for 2D slabs — carriers are quantum-confined in z.")
print("      In-plane kx = ky by symmetry at M-point (checked above).")
print()

# Also: 3D bulk at R for reference (limit of 2D as n→∞)
inv_v3R = _em3d(IVB_3D, KR); inv_c3R = _em3d(ICB_3D, KR)
mh3R = 1.0/inv_v3R[0]; me3R = 1.0/inv_c3R[0]
mu3R = abs(me3R*mh3R)/(abs(me3R)+abs(mh3R))
inv_v3M = _em3d(IVB_3D, KM); inv_c3M = _em3d(ICB_3D, KM)
mh3Mz = 1.0/inv_v3M[2]; me3Mz = 1.0/inv_c3M[2]
mu3Mz = abs(me3Mz*mh3Mz)/(abs(me3Mz)+abs(mh3Mz))
print(f"  3D R  {mh3R:>8.4f}  (iso)   {1.0/inv_v3R[2]:>12.4f}  {me3R:>8.4f}  (iso)   {1.0/inv_c3R[2]:>12.4f}  {mu3R:>8.4f}")
print(f"  3D M  {1.0/inv_v3M[0]:>8.4f}  (xy)   {mh3Mz:>12.4f}  {1.0/inv_c3M[0]:>8.4f}  (xy)   {me3Mz:>12.4f}  {mu3Mz:>8.4f}  (z)")

# ============================================================
# Plot
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle("Effective mass tensor: in-plane vs out-of-plane", fontsize=13)

n_vals = np.arange(1, 6)
inv_n  = 1.0 / n_vals

# -- panel 1: electron masses 3D full tensor at R, M, X --
ax = axes[0]
kp_labels = ["R", "M\n(xy)", "M\n(z)", "X\n(∥kx)", "X\n(⊥kx)"]
me_vals = []
mh_vals = []
for k in [KR, KM]:
    iv = _em3d(IVB_3D, k); ic = _em3d(ICB_3D, k)
    me_vals += [1.0/ic[0], 1.0/ic[2]]
    mh_vals += [abs(1.0/iv[0]), abs(1.0/iv[2])]
iv = _em3d(IVB_3D, KX); ic = _em3d(ICB_3D, KX)
me_vals += [1.0/ic[0]]
mh_vals += [abs(1.0/iv[0])]

x_pos = [0, 1, 2, 3, 4]
colors_e = ["#1f77b4", "#ff7f0e", "#ff7f0e", "#2ca02c", "#2ca02c"]
hatches   = ["", "", "///", "", "///"]
bars_e = ax.bar(x_pos, me_vals, color=colors_e, alpha=0.7)
for bar, h in zip(bars_e, hatches):
    bar.set_hatch(h)
ax.set_xticks(x_pos); ax.set_xticklabels(kp_labels, fontsize=9)
ax.set_ylabel("Electron mass ($m_0$)", fontsize=10)
ax.set_title("3D CBM effective masses\n(solid=in-plane, hatch=out-of-plane)")
ax.axhline(0, color='k', lw=0.5)

# -- panel 2: hole masses 3D --
ax = axes[1]
mh_plot = [abs(x) for x in [
    1.0/_em3d(IVB_3D,KR)[0],
    1.0/_em3d(IVB_3D,KM)[0],
    1.0/_em3d(IVB_3D,KM)[2],
    1.0/_em3d(IVB_3D,KX)[0],
    1.0/_em3d(IVB_3D,KX)[2],
]]
bars_h = ax.bar(x_pos, mh_plot, color=colors_e, alpha=0.7)
for bar, h in zip(bars_h, hatches):
    bar.set_hatch(h)
ax.set_xticks(x_pos); ax.set_xticklabels(kp_labels, fontsize=9)
ax.set_ylabel("|Hole mass| ($m_0$)", fontsize=10)
ax.set_title("3D VBM effective masses\n(solid=in-plane, hatch=out-of-plane)")

# -- panel 3: 2D in-plane vs 3D z-direction convergence --
ax = axes[2]
ax.plot(inv_n, np.abs(mh_in), 'o-', color='steelblue', ms=7, lw=1.8, label='2D $|m_h|$ in-plane')
ax.plot(inv_n, me_in,          's-', color='tomato',   ms=7, lw=1.8, label='2D $m_e$ in-plane')
ax.plot(inv_n, np.abs(mu_in),  '^-', color='seagreen', ms=7, lw=1.8, label='2D $\\mu$ in-plane')

ax.axhline(abs(mh3R),  color='steelblue', ls='--', lw=1, label=f'3D $|m_h|$(R)={abs(mh3R):.3f}')
ax.axhline(me3R,       color='tomato',    ls='--', lw=1, label=f'3D $m_e$(R)={me3R:.3f}')
ax.axhline(mu3R,       color='seagreen',  ls='--', lw=1, label=f'3D $\\mu$(R)={mu3R:.3f}')

# Add annotation for z-direction
ax.annotate("2D $m_z$ = ∞\n(quantum confined)", xy=(0.5, 0.85),
            xycoords='axes fraction', ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd', alpha=0.8))

xticks = [0, 1/5, 1/4, 1/3, 1/2, 1]
xlabels = ["bulk\n(3D)", "1/5", "1/4", "1/3", "1/2", "1/1"]
ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
ax.set_xlim(-0.05, 1.08)
ax.set_xlabel("$1/n$", fontsize=11)
ax.set_ylabel("Mass ($m_0$)", fontsize=10)
ax.set_title("2D in-plane masses vs $1/n$\n(dashed = 3D bulk limits)")
ax.legend(fontsize=7.5, ncol=2)

plt.tight_layout()
plt.savefig("effective_mass_tensor.png", dpi=150)
print("Saved effective_mass_tensor.png")
