"""
Plot band gap and reduced mass vs 1/n from n=1 (2D) to bulk (3D).
Run:  source ~/TBT-TS-sisl-workshop/setup.sh && python3 convergence_plot.py

Gaps and masses are evaluated at the M-point (π/a, π/a, 0) for both
2D slabs and 3D bulk, so the bulk point at 1/n=0 is a genuine limit.
The 3D R-point gap (direct fundamental gap) is also shown for reference.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ctypesroutines import (
    hamiltonian_cubic2D_apical_fortran,
    effective_mass_2d_apical_fortran,
    hamiltonian_cubic3D_fortran,
    effective_mass_3d_fortran,
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

# 3D params (no apical / no IIz)
PARAMS_3D = [Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
             deltaSO0, deltaSO1,
             Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
             Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb]

A_LAT   = 6.28
EPS_EM  = 0.05
IVB_3D  = 25   # 0-based
ICB_3D  = 26
IVB_2D  = [33, 59, 85, 111, 137]   # 0-based, n=1..5
ICB_2D  = [34, 60, 86, 112, 138]

KM  = np.array([np.pi / A_LAT, np.pi / A_LAT, 0.0])
KR  = np.array([np.pi / A_LAT, np.pi / A_LAT, np.pi / A_LAT])
KGA = np.array([0.0, 0.0, 0.0])


def mu_from_inv(inv_vb, inv_cb):
    me = 1.0 / inv_cb[0]
    mh = 1.0 / inv_vb[0]
    return abs(me * mh) / (abs(me) + abs(mh))


# ---- 3D bulk quantities ----
def _ham3d(k):
    return hamiltonian_cubic3D_fortran(k[0], k[1], k[2], *PARAMS_3D)

def _emass3d(band_idx, k):
    return effective_mass_3d_fortran(k[0], k[1], k[2], band_idx, 32,
                                     A_LAT, A_LAT, A_LAT, EPS_EM, *PARAMS_3D)

ev_3dM = eigenvalues_only_fortran(_ham3d(KM))
ev_3dR = eigenvalues_only_fortran(_ham3d(KR))

eg_3dM = ev_3dM[ICB_3D] - ev_3dM[IVB_3D]   # M-point gap (bulk limit for 2D series)
eg_3dR = ev_3dR[ICB_3D] - ev_3dR[IVB_3D]   # R-point gap (true direct fundamental gap)

inv_vb_3dM = _emass3d(IVB_3D, KM)
inv_cb_3dM = _emass3d(ICB_3D, KM)
mu_3dM = mu_from_inv(inv_vb_3dM, inv_cb_3dM)

inv_vb_3dR = _emass3d(IVB_3D, KR)
inv_cb_3dR = _emass3d(ICB_3D, KR)
mu_3dR = mu_from_inv(inv_vb_3dR, inv_cb_3dR)

print(f"3D bulk:  eg(M) = {eg_3dM:.4f} eV   mu(M) = {mu_3dM:.4f} m0")
print(f"3D bulk:  eg(R) = {eg_3dR:.4f} eV   mu(R) = {mu_3dR:.4f} m0")


# ---- 2D slab quantities (n=1..5) ----
def _ham2d(nL, k):
    return hamiltonian_cubic2D_apical_fortran(
        nL, k[0], k[1], k[2],
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        Es1_ap, Ep1_ap, Ep0_surf,
    )

def _emass2d(nL, band_idx, k):
    return effective_mass_2d_apical_fortran(
        nL, k[0], k[1], k[2], band_idx,
        A_LAT, A_LAT, A_LAT, EPS_EM,
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        Es1_ap, Ep1_ap, Ep0_surf,
    )

# Experimental targets for comparison
EXP_EG = [3.016, 2.432, 2.206, 2.078, 1.971]   # n=1..5, eV
EXP_MU = [0.221, 0.217, 0.201, 0.196, 0.186]   # n=1..5, m0

n_vals   = np.arange(1, 6)
inv_n    = 1.0 / n_vals
eg_2d    = np.zeros(5)
mu_2d    = np.zeros(5)

for i, nL in enumerate(n_vals):
    ev = eigenvalues_only_fortran(_ham2d(nL, KM))
    eg_2d[i] = ev[ICB_2D[i]] - ev[IVB_2D[i]]
    inv_vb = _emass2d(nL, IVB_2D[i], KM)
    inv_cb = _emass2d(nL, ICB_2D[i], KM)
    mu_2d[i] = mu_from_inv(inv_vb, inv_cb)
    print(f"n={nL}: eg = {eg_2d[i]:.4f} eV   mu = {mu_2d[i]:.4f} m0")


# ---- plot ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
fig.suptitle("Convergence of 2D slab to 3D bulk with layer number $n$",
             fontsize=13)

x_2d   = inv_n                   # 1/1 .. 1/5
x_bulk = 0.0
xticks = np.array([0, 1/5, 1/4, 1/3, 1/2, 1])
xlabels = ["bulk\n(3D)", "1/5", "1/4", "1/3", "1/2", "1/1"]

# ---- gap panel ----
ax1.plot(x_2d, eg_2d, 'o-', color='steelblue', ms=7, lw=1.8, label='Model (M-point)')
ax1.scatter(x_bulk, eg_3dM, s=80, color='steelblue', marker='*', zorder=5,
            label=f'Bulk M: {eg_3dM:.3f} eV')
ax1.scatter(x_bulk, eg_3dR, s=80, color='navy', marker='D', zorder=5,
            label=f'Bulk R: {eg_3dR:.3f} eV (direct)')
ax1.plot(x_2d, EXP_EG, 's--', color='tomato', ms=6, lw=1.4, label='Experiment')
ax1.set_xlabel("$1/n$", fontsize=12)
ax1.set_ylabel("Band gap (eV)", fontsize=12)
ax1.set_title("Band gap vs $1/n$")
ax1.set_xticks(xticks)
ax1.set_xticklabels(xlabels)
ax1.legend(fontsize=9)
ax1.set_xlim(-0.05, 1.08)

# ---- mass panel ----
ax2.plot(x_2d, mu_2d, 'o-', color='darkorange', ms=7, lw=1.8, label='Model (M-point)')
ax2.scatter(x_bulk, mu_3dM, s=80, color='darkorange', marker='*', zorder=5,
            label=f'Bulk M: {mu_3dM:.4f} $m_0$')
ax2.scatter(x_bulk, mu_3dR, s=80, color='saddlebrown', marker='D', zorder=5,
            label=f'Bulk R: {mu_3dR:.4f} $m_0$ (direct)')
ax2.plot(x_2d, EXP_MU, 's--', color='tomato', ms=6, lw=1.4, label='Experiment')
ax2.set_xlabel("$1/n$", fontsize=12)
ax2.set_ylabel("Reduced mass ($m_0$)", fontsize=12)
ax2.set_title("Reduced mass vs $1/n$")
ax2.set_xticks(xticks)
ax2.set_xticklabels(xlabels)
ax2.legend(fontsize=9)
ax2.set_xlim(-0.05, 1.08)

plt.tight_layout()
out = "convergence_plot.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")
