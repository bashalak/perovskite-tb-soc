"""
Comprehensive comparison table: Experiment vs DFT/Wannier vs This TB model.
Prints a formatted ASCII table and saves comparison_table.png.

Run:  source ~/TBT-TS-sisl-workshop/setup.sh && python3 comparison_table.py

Data sources
------------
  Experiment (electronic gaps) : Blancon et al., Nat. Commun. 9, 2254 (2018)
                                  4s exciton series ≈ quasiparticle gap
  Experiment (optical gaps)    : Blancon 2018 / Stoumpos, Chem. Mater. 28, 2852 (2016)
  Experiment (3D gaps, masses) : Various literature (see targets in tb_optimize.py)
  DFT/Wannier (2D gaps)        : Wannier-TB from DFT+SOC on experimental structure 293K
                                  band_gaps.dat — indirect gaps (Rashba), included for
                                  comparison only (not used in fitting)
  This model                   : 26-parameter SP3 TB, best-fit parameters
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

A_LAT  = 6.28
EPS_EM = 0.05
KM  = np.array([np.pi/A_LAT, np.pi/A_LAT, 0.0])
KR  = np.array([np.pi/A_LAT, np.pi/A_LAT, np.pi/A_LAT])
KX  = np.array([np.pi/A_LAT, 0.0, 0.0])
KGA = np.array([0.0, 0.0, 0.0])
IVB_3D = 25; ICB_3D = 26
IVB_2D = [33, 59, 85, 111, 137]
ICB_2D = [34, 60, 86, 112, 138]

NaN = float("nan")

# ============================================================
# REFERENCE DATA
# ============================================================

# --- Experimental electronic gaps (Blancon 2018, 4K) ---
EXP_EG_ELEC = {
    "eg3DR":  1.650,
    "eg3DM":  2.750,
    "eg3DX":  3.400,
    "eg2D1":  3.016,
    "eg2D2":  2.432,
    "eg2D3":  2.206,
    "eg2D4":  2.078,
    "eg2D5":  1.971,
    "dEhRX":  1.000,
    "dEhMX":  0.700,
    "ege2R":  2.750,
    "egh2R":  2.750,
    "mu3dR":  0.108,
    "mu2D1":  0.221,
    "mu2D2":  0.217,
    "mu2D3":  0.201,
    "mu2D4":  0.196,
    "mu2D5":  0.186,
    "mhM":   -0.240,
    "mhX":   -0.350,
}

# --- Experimental optical gaps (Blancon/Stoumpos, 293K or 4K) ---
EXP_EG_OPT = {
    "eg2D1": 2.54,
    "eg2D2": 2.18,
    "eg2D3": 2.03,
    "eg2D4": 1.92,
    "eg2D5": 1.85,
    "eg3DR": 1.620,   # ~30 meV below electronic
}

# --- DFT without SOC (Wannier interpolation, spinless basis) ---
# Source: wannier_band3D.dat (16 spinless bands) and wannier_band2Dn1.dat (20 bands)
# Gap extracted as min(CB) - max(VB) along the interpolated k-path.
# NO data exists for n=2..5.
DFT_NO_SOC = {
    "eg3DR": 1.497,   # wannier_band3D.dat: gap between spinless band 13 and 14
    "eg2D1": 1.848,   # wannier_band2Dn1.dat: gap between spinless band 17 and 18
}

# --- Prior TB+SOC model (from band_gaps.dat, indirect gaps with Rashba, 293K) ---
# These are from a previous TB parameterisation applied to the experimental crystal
# structure at 293 K.  The gaps are INDIRECT (Rashba split) and are systematically
# higher than the direct M-point gaps we fit. Included for reference only.
PRIOR_TB_SOC = {
    "eg2D1": 3.355,
    "eg2D2": 2.960,
    "eg2D3": 2.750,
    "eg2D4": 2.845,   # Rashba anomaly: n=4 > n=3 in indirect gap
    "eg2D5": 2.644,
}

# ============================================================
# COMPUTE MODEL VALUES
# ============================================================

def mu_from_inv(inv_vb, inv_cb):
    me = 1.0 / inv_cb[0]; mh = 1.0 / inv_vb[0]
    return abs(me * mh) / (abs(me) + abs(mh))

def _ham3d(k):
    return hamiltonian_cubic3D_fortran(k[0], k[1], k[2], *PARAMS_3D)

def _emass3d(bidx, k):
    return effective_mass_3d_fortran(k[0], k[1], k[2], bidx, 32,
                                     A_LAT, A_LAT, A_LAT, EPS_EM, *PARAMS_3D)

def _ham2d(nL, k):
    return hamiltonian_cubic2D_apical_fortran(
        nL, k[0], k[1], k[2],
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        Es1_ap, Ep1_ap, Ep0_surf)

def _emass2d(nL, bidx, k):
    return effective_mass_2d_apical_fortran(
        nL, k[0], k[1], k[2], bidx,
        A_LAT, A_LAT, A_LAT, EPS_EM,
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        Es1_ap, Ep1_ap, Ep0_surf)

# 3D
evR = eigenvalues_only_fortran(_ham3d(KR))
evM = eigenvalues_only_fortran(_ham3d(KM))
evX = eigenvalues_only_fortran(_ham3d(KX))

eg3DR = evR[ICB_3D] - evR[IVB_3D]
eg3DM = evM[ICB_3D] - evM[IVB_3D]
eg3DX = evX[ICB_3D] - evX[IVB_3D]
dEhRX = evR[IVB_3D] - evX[IVB_3D]
dEhMX = evM[IVB_3D] - evX[IVB_3D]
# At R, bands are Kramers-degenerate (pairs); skip the degenerate partner
# ICB+1 is degenerate with ICB, so true 2nd CB state is ICB+2
# IVB-1 is degenerate with IVB, so true 2nd VB state is IVB-2
ege2R = evR[ICB_3D + 2] - evR[IVB_3D]
egh2R = evR[ICB_3D]     - evR[IVB_3D - 2]

inv_vb3R = _emass3d(IVB_3D, KR); inv_cb3R = _emass3d(ICB_3D, KR)
mu3dR = mu_from_inv(inv_vb3R, inv_cb3R)
mhM   = 1.0 / _emass3d(IVB_3D, KM)[0]
mhX   = 1.0 / _emass3d(IVB_3D, KX)[0]

# 2D
eg2d = []; mu2d = []
for i, nL in enumerate(range(1, 6)):
    ev = eigenvalues_only_fortran(_ham2d(nL, KM))
    eg2d.append(ev[ICB_2D[i]] - ev[IVB_2D[i]])
    ivb = _emass2d(nL, IVB_2D[i], KM)
    icb = _emass2d(nL, ICB_2D[i], KM)
    mu2d.append(mu_from_inv(ivb, icb))

MODEL = {
    "eg3DR":  eg3DR, "eg3DM":  eg3DM, "eg3DX":  eg3DX,
    "eg2D1":  eg2d[0], "eg2D2": eg2d[1], "eg2D3": eg2d[2],
    "eg2D4":  eg2d[3], "eg2D5": eg2d[4],
    "dEhRX":  dEhRX, "dEhMX": dEhMX,
    "ege2R":  ege2R, "egh2R": egh2R,
    "mu3dR":  mu3dR,
    "mu2D1":  mu2d[0], "mu2D2": mu2d[1], "mu2D3": mu2d[2],
    "mu2D4":  mu2d[3], "mu2D5": mu2d[4],
    "mhM":   mhM, "mhX": mhX,
}

# ============================================================
# TABLE DEFINITION
# ============================================================
# Each row: (label, key, unit, exp_elec, exp_opt, dft_wannier, note)
# Use None where no data is available

# Each row: (label, key, unit, exp_elec, exp_opt, dft_no_soc, prior_tb_soc, note)
ROWS = [
    # ---- 3D bulk ----
    ("─── 3D bulk ─────────────────────────────────────────────────────────", None, None, None, None, None, None, ""),
    ("  Eg at R (direct fund. gap)",  "eg3DR", "eV",  1.650, 1.620, 1.497,  None,  "DFT w/o SOC: Wannier band 13→14"),
    ("  Eg at M",                     "eg3DM", "eV",  2.750, None,  None,   None,  "M=(π/a,π/a,0)"),
    ("  Eg at X",                     "eg3DX", "eV",  3.400, None,  None,   None,  "X=(π/a,0,0)"),
    ("  VBM(R)−VBM(X)  [dEhRX]",     "dEhRX", "eV",  1.000, None,  None,   None,  "VB dispersion"),
    ("  VBM(M)−VBM(X)  [dEhMX]",     "dEhMX", "eV",  0.700, None,  None,   None,  "VB dispersion"),
    ("  Eg(CBM+2→VBM) at R [ege2R]", "ege2R", "eV",  2.750, None,  None,   None,  "2nd CB pair (skip Kramers)"),
    ("  Eg(CBM→VBM-2) at R [egh2R]", "egh2R", "eV",  2.750, None,  None,   None,  "2nd VB pair (skip Kramers)"),
    ("  μ (reduced mass) at R",       "mu3dR", "m₀",  0.108, None,  None,   None,  ""),
    ("  mh at M (hole mass, signed)", "mhM",   "m₀", -0.240, None,  None,   None,  ""),
    ("  mh at X (hole mass, signed)", "mhX",   "m₀", -0.350, None,  None,   None,  ""),
    # ---- 2D slabs ----
    ("─── 2D slabs: direct band gap at M-point ───────────────────────────", None, None, None, None, None, None, ""),
    ("  n=1  Eg at M",  "eg2D1", "eV", 3.016, 2.54, 1.848, 3.355, "DFT w/o SOC: Wannier band 17→18; prior TB: indirect+Rashba"),
    ("  n=2  Eg at M",  "eg2D2", "eV", 2.432, 2.18, None,  2.960, "prior TB: indirect+Rashba only"),
    ("  n=3  Eg at M",  "eg2D3", "eV", 2.206, 2.03, None,  2.750, "prior TB: indirect+Rashba only"),
    ("  n=4  Eg at M",  "eg2D4", "eV", 2.078, 1.92, None,  2.845, "prior TB: Rashba anomaly n4>n3"),
    ("  n=5  Eg at M",  "eg2D5", "eV", 1.971, 1.85, None,  2.644, "prior TB: indirect+Rashba only"),
    # ---- 2D masses ----
    ("─── 2D slabs: reduced effective mass at M ──────────────────────────", None, None, None, None, None, None, ""),
    ("  n=1  μ at M",  "mu2D1", "m₀", 0.221, None, None, None, ""),
    ("  n=2  μ at M",  "mu2D2", "m₀", 0.217, None, None, None, ""),
    ("  n=3  μ at M",  "mu2D3", "m₀", 0.201, None, None, None, ""),
    ("  n=4  μ at M",  "mu2D4", "m₀", 0.196, None, None, None, ""),
    ("  n=5  μ at M",  "mu2D5", "m₀", 0.186, None, None, None, ""),
]

# ============================================================
# PRINT ASCII TABLE
# ============================================================

def fmt(v, unit):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "    —    "
    if unit == "eV":
        return f"  {v:+7.4f}"
    if unit == "m₀":
        return f"  {v:+7.4f}"
    return f"  {v:8.4f}"

def err_fmt(res, rel):
    if np.isnan(res):
        return "    —         —   "
    return f"  {res:+7.4f}  ({rel:+6.1f}%)"

HDR = (f"{'Quantity':<50} {'Unit':>4}  {'Exp.(elec.)':>10}  {'Exp.(opt.)':>10}"
       f"  {'DFT w/o SOC':>11}  {'Prior TB+SOC':>12}  {'This model':>10}"
       f"  {'Residual':>9}  {'Rel.err.':>8}")
SEP = "─" * len(HDR)

print(SEP)
print(HDR)
print(SEP)

res_all = []
for row in ROWS:
    label, key, unit, exp_e, exp_o, dft_nosoc, prior_tb, note = row
    if key is None:
        print(f"\n{label}")
        continue
    mod  = MODEL.get(key, NaN)
    ref  = exp_e
    res  = (mod - ref) if (ref is not None and not np.isnan(mod)) else NaN
    rel  = 100 * res / abs(ref) if (ref is not None and ref != 0 and not np.isnan(res)) else NaN
    if not np.isnan(res):
        res_all.append((key, res, rel, unit))

    print(f"  {label:<48} {unit:>4}  {fmt(exp_e, unit)}  {fmt(exp_o, unit)}"
          f"  {fmt(dft_nosoc, unit)}  {fmt(prior_tb, unit)}  {fmt(mod, unit)}"
          f"  {err_fmt(res, rel)}")

print(SEP)

# ---- group error metrics ----
groups = {
    "3D gaps   (eg3DR, eg3DM, eg3DX)":       ["eg3DR", "eg3DM", "eg3DX"],
    "VB disp.  (dEhRX, dEhMX)":              ["dEhRX", "dEhMX"],
    "2nd-pair  (ege2R, egh2R)":              ["ege2R", "egh2R"],
    "3D masses (mu3dR, mhM, mhX)":           ["mu3dR", "mhM", "mhX"],
    "2D gaps   (n=1..5)":                    ["eg2D1","eg2D2","eg2D3","eg2D4","eg2D5"],
    "2D masses (n=1..5)":                    ["mu2D1","mu2D2","mu2D3","mu2D4","mu2D5"],
}

res_map = {k: (r, rl) for k, r, rl, u in res_all}

print("\nError metrics per group:")
print(f"  {'Group':<44} {'N':>3}  {'RMS':>8}  {'MAE':>8}  {'MaxErr':>9}  {'R²':>7}")
print("  " + "─" * 82)
for gname, keys in groups.items():
    vals = [res_map[k][0] for k in keys if k in res_map]
    if not vals:
        continue
    vals = np.array(vals)
    rms  = np.sqrt(np.mean(vals**2))
    mae  = np.mean(np.abs(vals))
    maxe = np.max(np.abs(vals))
    exp_vals = np.array([EXP_EG_ELEC.get(k, NaN) for k in keys
                         if k in res_map and not np.isnan(EXP_EG_ELEC.get(k, NaN))])
    mod_vals = np.array([MODEL.get(k, NaN)   for k in keys
                         if k in res_map and not np.isnan(EXP_EG_ELEC.get(k, NaN))])
    if len(exp_vals) > 1:
        ss_res = np.sum((mod_vals - exp_vals)**2)
        ss_tot = np.sum((exp_vals - np.mean(exp_vals))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else NaN
    else:
        r2 = NaN
    r2s = f"{r2:7.4f}" if not np.isnan(r2) else "    —  "
    print(f"  {gname:<44} {len(vals):>3}  {rms:8.4f}  {mae:8.4f}  {maxe:9.4f}  {r2s}")

print(SEP)

# ============================================================
# MATPLOTLIB TABLE FIGURE
# ============================================================
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#f8f8f8")
gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.35)

ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1])
for ax in (ax_top, ax_bot):
    ax.axis("off")

# -- top table --
col_labels = ["Quantity", "Unit", "Exp. (elec.)", "Exp. (opt.)",
              "DFT w/o SOC", "Prior TB+SOC", "This model", "Residual", "Rel. err (%)"]
col_widths = [0.27, 0.04, 0.09, 0.08, 0.09, 0.10, 0.09, 0.08, 0.09]

def fv(v, unit):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:+.4f}" if unit in ("eV", "m₀") else f"{v:.4f}"

table_rows = []
row_colors = []
for row in ROWS:
    label, key, unit, exp_e, exp_o, dft_nosoc, prior_tb, note = row
    if key is None:
        table_rows.append([label] + [""] * 8)
        row_colors.append(["#dde8f0"] * 9)
        continue
    mod = MODEL.get(key, NaN)
    res = (mod - exp_e) if (exp_e is not None and not np.isnan(mod)) else NaN
    rel = 100 * res / abs(exp_e) if (exp_e is not None and exp_e != 0 and not np.isnan(res)) else NaN

    res_s = f"{res:+.4f}" if not np.isnan(res) else "—"
    rel_s = f"{rel:+.1f}%" if not np.isnan(rel) else "—"

    if np.isnan(res):
        rc = "#ffffff"
    elif abs(rel) < 5:
        rc = "#d4f1d4"
    elif abs(rel) < 15:
        rc = "#fff4cc"
    else:
        rc = "#fcd5d5"

    row_colors.append(["#ffffff"] * 7 + [rc, rc])
    table_rows.append([label, unit if unit else "",
                       fv(exp_e, unit), fv(exp_o, unit),
                       fv(dft_nosoc, unit), fv(prior_tb, unit),
                       f"{mod:+.4f}" if not np.isnan(mod) else "—",
                       res_s, rel_s])

tbl = ax_top.table(
    cellText=table_rows,
    colLabels=col_labels,
    cellColours=row_colors,
    colWidths=col_widths,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.35)

# Bold header
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor("#2b4e6e")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold", fontsize=7.5)

ax_top.set_title(
    "Comprehensive comparison: Experiment vs DFT/Wannier vs SP³ TB model (this work)\n"
    "(BA)₂(MA)_{n-1}Pb_nI_{3n+1} — Ruddlesden-Popper lead iodide perovskites",
    fontsize=11, fontweight="bold", pad=10,
)

# -- bottom table: error metrics --
bot_col_labels = ["Group", "N", "RMS", "MAE", "Max |error|", "R²"]
bot_rows = []
for gname, keys in groups.items():
    vals = [res_map[k][0] for k in keys if k in res_map]
    if not vals:
        continue
    vals = np.array(vals)
    rms = np.sqrt(np.mean(vals**2))
    mae = np.mean(np.abs(vals))
    maxe = np.max(np.abs(vals))
    exp_v = np.array([EXP_EG_ELEC.get(k, NaN) for k in keys
                      if k in res_map and not np.isnan(EXP_EG_ELEC.get(k, NaN))])
    mod_v = np.array([MODEL.get(k, NaN) for k in keys
                      if k in res_map and not np.isnan(EXP_EG_ELEC.get(k, NaN))])
    if len(exp_v) > 1:
        ss_res = np.sum((mod_v - exp_v)**2)
        ss_tot = np.sum((exp_v - np.mean(exp_v))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else NaN
    else:
        r2 = NaN
    u_label = "(eV)" if "gap" in gname.lower() or "disp" in gname.lower() or "pair" in gname.lower() else "(m₀)"
    bot_rows.append([
        gname, str(len(vals)),
        f"{rms:.4f} {u_label}", f"{mae:.4f} {u_label}", f"{maxe:.4f} {u_label}",
        f"{r2:.4f}" if not np.isnan(r2) else "—",
    ])

bot_colors = [["#eef3f8"] * 6] * len(bot_rows)
btbl = ax_bot.table(
    cellText=bot_rows,
    colLabels=bot_col_labels,
    cellColours=bot_colors,
    colWidths=[0.38, 0.05, 0.17, 0.17, 0.17, 0.08],
    loc="center",
    cellLoc="center",
)
btbl.auto_set_font_size(False)
btbl.set_fontsize(9)
btbl.scale(1, 1.5)
for j in range(len(bot_col_labels)):
    btbl[(0, j)].set_facecolor("#2b4e6e")
    btbl[(0, j)].set_text_props(color="white", fontweight="bold")

ax_bot.set_title("Error metrics per observable group", fontsize=10,
                 fontweight="bold", pad=8)

fig.text(0.5, 0.01,
         "Green: |rel. err| < 5%  |  Yellow: < 15%  |  Red: ≥ 15%  |  "
         "DFT w/o SOC: Wannier interpolation (spinless, 3D and n=1 only).  "
         "Prior TB+SOC: previous TB on experimental structure, indirect gaps with Rashba splitting.",
         ha="center", fontsize=7.5, color="#444444")

plt.savefig("comparison_table.png", dpi=150, bbox_inches="tight")
print("\nSaved comparison_table.png")
