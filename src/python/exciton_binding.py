"""
Exciton binding energy via Wannier-Mott model:
    E_b = 13.6 eV × μ/m₀ / ε_eff²

Two panels:
  1. E_b vs 1/n for a range of ε_eff values — shows where experiment lands
  2. Optical gap = electronic gap − E_b vs 1/n, compared with measured
     optical absorption onsets

Run:  source ~/TBT-TS-sisl-workshop/setup.sh && python3 exciton_binding.py
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

PARAMS_3D = [Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
             deltaSO0, deltaSO1,
             Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
             Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb]

A_LAT   = 6.28
EPS_EM  = 0.05
IVB_3D  = 25
ICB_3D  = 26
IVB_2D  = [33, 59, 85, 111, 137]
ICB_2D  = [34, 60, 86, 112, 138]
KM  = np.array([np.pi / A_LAT, np.pi / A_LAT, 0.0])
KR  = np.array([np.pi / A_LAT, np.pi / A_LAT, np.pi / A_LAT])

RYD = 13.6056   # eV, Rydberg constant


# ---------------------------------------------------------------------------
# Experimental data from literature for (BA)₂(MA)_{n-1}Pb_nI_{3n+1}
# Electronic (quasiparticle) gaps and optical (excitonic) gaps at M
# Sources: Blancon et al. Nat. Commun. 2018; Soe et al. JACS 2017
# Adjust these if you have more precise values for your specific compound.
# ---------------------------------------------------------------------------
EXP_EG_ELEC = np.array([3.016, 2.432, 2.206, 2.078, 1.971])  # n=1..5 (fitted targets)
EXP_EG_OPT  = np.array([2.57,  2.18,  2.04,  1.95,  1.89 ])  # n=1..5, optical onset (eV)
EXP_EB       = EXP_EG_ELEC - EXP_EG_OPT                        # derived E_b
EXP_EB_3D    = 0.030   # bulk MAPbI₃ ~16–50 meV, use ~30 meV
EXP_EG_OPT_3D = 1.65 - EXP_EB_3D                               # ~1.62 eV

print("Experimental exciton binding energies (from gap difference):")
for i, n in enumerate(range(1, 6)):
    print(f"  n={n}: E_b = {EXP_EB[i]*1000:.0f} meV  "
          f"(E_elec={EXP_EG_ELEC[i]:.3f}, E_opt={EXP_EG_OPT[i]:.3f})")
print(f"  bulk: E_b = {EXP_EB_3D*1000:.0f} meV")


# ---------------------------------------------------------------------------
# Reduced masses from model
# ---------------------------------------------------------------------------
def mu_from_inv(inv_vb, inv_cb):
    me = 1.0 / inv_cb[0]
    mh = 1.0 / inv_vb[0]
    return abs(me * mh) / (abs(me) + abs(mh))

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

mu_2d = np.zeros(5)
eg_2d = np.zeros(5)
for i, nL in enumerate(range(1, 6)):
    ev = eigenvalues_only_fortran(_ham2d(nL, KM))
    eg_2d[i] = ev[ICB_2D[i]] - ev[IVB_2D[i]]
    inv_vb = _emass2d(nL, IVB_2D[i], KM)
    inv_cb = _emass2d(nL, ICB_2D[i], KM)
    mu_2d[i] = mu_from_inv(inv_vb, inv_cb)

# 3D at R-point (fundamental gap)
ev_3dR = eigenvalues_only_fortran(
    hamiltonian_cubic3D_fortran(KR[0], KR[1], KR[2], *PARAMS_3D))
eg_3dR = ev_3dR[ICB_3D] - ev_3dR[IVB_3D]
inv_vb_3dR = effective_mass_3d_fortran(KR[0], KR[1], KR[2], IVB_3D, 32,
                                        A_LAT, A_LAT, A_LAT, EPS_EM, *PARAMS_3D)
inv_cb_3dR = effective_mass_3d_fortran(KR[0], KR[1], KR[2], ICB_3D, 32,
                                        A_LAT, A_LAT, A_LAT, EPS_EM, *PARAMS_3D)
mu_3dR = mu_from_inv(inv_vb_3dR, inv_cb_3dR)


# ---------------------------------------------------------------------------
# Wannier-Mott binding energy
# E_b = Ry × μ / ε_eff²
#
# Dielectric model:
#   ε_eff(n) = (n·ε_well + 2·ε_barrier) / (n + 2)
#   ε_well   = optical dielectric constant of perovskite ≈ 4.5
#   ε_barrier = optical dielectric constant of organic  ≈ 2.1
#   This is a simple "slab average" — Keldysh gives a more rigorous result
#   but requires knowing the slab thickness and barrier thickness separately.
# ---------------------------------------------------------------------------
eps_well    = 5.75#4.5   # adjust to match bulk E_b n prov=2.4-2.5
eps_barrier = 1.0#2.1

n_vals = np.arange(1, 6)
inv_n  = 1.0 / n_vals

eps_eff_2d = (n_vals * eps_well + 2 * eps_barrier) / (n_vals + 2)
eps_eff_3d = eps_well   # bulk limit (n → ∞)

eb_2d_model = RYD * mu_2d / eps_eff_2d**2
eb_3d_model = RYD * mu_3dR / eps_eff_3d**2

opt_gap_model_2d = eg_2d - eb_2d_model
opt_gap_model_3d = eg_3dR - eb_3d_model

print("\nWannier-Mott E_b and optical gap (model):")
print(f"  {'n':>3}  {'μ (m₀)':>8}  {'ε_eff':>6}  {'E_b (meV)':>10}  "
      f"{'E_gap (eV)':>11}  {'E_opt (eV)':>11}")
for i, n in enumerate(n_vals):
    print(f"  {n:>3}  {mu_2d[i]:>8.4f}  {eps_eff_2d[i]:>6.2f}  "
          f"{eb_2d_model[i]*1000:>10.1f}  {eg_2d[i]:>11.4f}  {opt_gap_model_2d[i]:>11.4f}")
print(f"  {'3D':>3}  {mu_3dR:>8.4f}  {eps_eff_3d:>6.2f}  "
      f"{eb_3d_model*1000:>10.1f}  {eg_3dR:>11.4f}  {opt_gap_model_3d:>11.4f}")

# E_b for a family of ε_eff values (for sensitivity plot)
eps_scan = np.array([2.0, 2.5, 3.0, 3.5, 4.5])
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Wannier-Mott exciton binding energy from TB reduced masses", fontsize=13)

xticks  = np.array([0, 1/5, 1/4, 1/3, 1/2, 1])
xlabels = ["bulk\n(3D)", "1/5", "1/4", "1/3", "1/2", "1/1"]
xlim    = (-0.06, 1.10)

# ---- panel 1: E_b vs 1/n ----
# Sensitivity: family of ε_eff curves
#for eps, col in zip(eps_scan, colors):
#    eb_fam = RYD * mu_2d / eps**2
#    eb_3d_fam = RYD * mu_3dR / eps**2
#    ax1.plot(inv_n, eb_fam * 1000, '-', color=col, lw=1.2, alpha=0.6,
#             label=f"ε={eps:.1f}")
#    ax1.scatter(0, eb_3d_fam * 1000, marker='*', s=60, color=col, zorder=5)

# Slab-average model (highlighted)
ax1.plot(inv_n, eb_2d_model * 1000, 'o-', color='black', lw=2.0, ms=7,
         label=f"slab avg ε(n)", zorder=6)
ax1.scatter(0, eb_3d_model * 1000, marker='*', s=120, color='black', zorder=7)

# Experimental E_b
ax1.plot(inv_n, EXP_EB * 1000, 's--', color='tomato', lw=1.8, ms=7,
         label='Exp. ($E_{elec}-E_{opt}$)')
ax1.scatter(0, EXP_EB_3D * 1000, marker='D', s=60, color='tomato', zorder=5)

ax1.set_xticks(xticks)
ax1.set_xticklabels(xlabels)
ax1.set_xlim(*xlim)
ax1.set_xlabel("$1/n$", fontsize=12)
ax1.set_ylabel("Exciton binding energy (meV)", fontsize=11)
ax1.set_title("$E_b = R_y \\cdot \\mu / \\varepsilon_{\\rm eff}^2$")
ax1.legend(fontsize=8, ncol=2)

# ---- panel 2: optical gap vs 1/n ----
ax2.plot(inv_n, opt_gap_model_2d, 'o-', color='steelblue', lw=2.0, ms=7,
         label='Model: $E_{gap} - E_b$')
ax2.scatter(0, opt_gap_model_3d, marker='*', s=120, color='steelblue', zorder=5,
            label=f'Bulk opt.: {opt_gap_model_3d:.3f} eV')
ax2.plot(inv_n, eg_2d, 's:', color='steelblue', lw=1.2, ms=5, alpha=0.5,
         label='Model: $E_{gap}$ (electronic)')

ax2.plot(inv_n, EXP_EG_OPT, 's--', color='tomato', lw=1.8, ms=7,
         label='Exp. optical gap')
ax2.plot(inv_n, EXP_EG_ELEC, '^:', color='tomato', lw=1.2, ms=5, alpha=0.5,
         label='Exp. electronic gap')
ax2.scatter(0, EXP_EG_OPT_3D, marker='D', s=60, color='tomato', zorder=5)

ax2.set_xticks(xticks)
ax2.set_xticklabels(xlabels)
ax2.set_xlim(*xlim)
ax2.set_xlabel("$1/n$", fontsize=12)
ax2.set_ylabel("Energy (eV)", fontsize=11)
ax2.set_title("Optical gap $= E_{gap}^{\\rm TB} - E_b^{\\rm WM}$")
ax2.legend(fontsize=8)

plt.tight_layout()
out = "exciton_binding.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")
