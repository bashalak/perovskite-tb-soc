"""
bse_2d.py — Phase 5 Level-2: k-space Bethe-Salpeter Equation for 2D perovskite excitons.

Builds the BSE Hamiltonian on a 2D k-patch around the M-point band edge:

    H_exc[k,k'] = (Ec(k) - Ev(k)) delta(k,k')
                + V_RK(|k-k'|) * F(k,k') * (dk)^2 / (2*pi)^2

where:
    V_RK(q)   — Rytova-Keldysh potential in k-space [eV·nm²] (from rytova_keldysh.py)
    F(k,k')   = |<psi_c,k | psi_v,k'>|^2  (form factor from TB 32*n+8 spinor eigenvectors)
    (dk)^2/(2pi)^2  — phase-space factor converting V_RK [eV·nm²] to [eV]

Solves via scipy.linalg.eigh → binding energies Omega_s, oscillator strengths f_s.

Material: (BA)_2(MA)_{n-1}Pb_n I_{3n+1}, n = 1-5
Run from src/python/:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 bse_2d.py [Nk]   (default Nk=21)
"""

import numpy as np
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from ctypesroutines import (
    hamiltonian_cubic2D_apical_fortran,
    diagonalize_hamiltonian_fortran,
)
from rytova_keldysh import V_RK_q, r0_nm, EL_EXP, EL_BSE, solve_binding_energy

# ── TB parameters (best-fit, 26 values — same as exciton_binding.py) ──────────
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

A_LAT = 6.28          # in-plane lattice constant [Å]
KM    = np.pi / A_LAT  # M-point kx = ky [Å⁻¹]

# 0-based VBM / CBM indices at M for each n (from diagonalization of H(kM))
IVB = {1: 33, 2: 59, 3: 85, 4: 111, 5: 137}
ICB = {1: 34, 2: 60, 3: 86, 4: 112, 5: 138}


def _ham(n_layers, kx, ky):
    """TB Hamiltonian H(kx, ky) [Å⁻¹], kz=0."""
    return hamiltonian_cubic2D_apical_fortran(
        n_layers, kx, ky, 0.0,
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        Es1_ap, Ep1_ap, Ep0_surf,
    )


def solve_bse(n, Nk=21, dk_max=2.0, n_states=10):
    """
    Build and solve the BSE for slab n on an Nk×Nk k-patch around M.

    Parameters
    ----------
    n        : slab index 1–5
    Nk       : grid points per side  (total Nk² k-points)
    dk_max   : half-width of k-patch around M  [nm⁻¹]
    n_states : number of lowest BSE eigenvalues to return

    Returns
    -------
    dict with keys:
        E_bind_meV : ground-state binding energy [meV]  (positive = bound)
        omega      : (n_states,) excitation energies relative to E_gap [eV]
        osc        : (n_states,) oscillator strengths (unnormalized)
        E_gap      : TB band gap at M [eV]
        Nk2        : total k-points used
    """
    r0       = r0_nm(n)
    ivb, icb = IVB[n], ICB[n]
    n_orb    = 32 * n + 8

    # k-grid in Å⁻¹ centered at M=(KM, KM); dk_max in nm⁻¹ → Å⁻¹ (/10)
    dk_ang      = dk_max / 10.0
    dkv         = np.linspace(-dk_ang, dk_ang, Nk)
    dk_step_nm  = (dkv[1] - dkv[0]) * 10.0   # grid spacing [nm⁻¹]
    KX, KY      = np.meshgrid(KM + dkv, KM + dkv, indexing='ij')
    kx_flat     = KX.ravel()
    ky_flat     = KY.ravel()
    Nk2         = Nk * Nk

    # ── Diagonalize TB at each k-point, with band tracking ───────────────────
    # Fixed indices (ivb, icb) identify VBM/CBM only at the exact M point.
    # For other k, bands can cross or hybridize, so we track the correct band
    # by maximising the overlap with the reference states at M.
    print(f"  n={n}: TB diagonalization at {Nk2} k-points ...", flush=True)
    Ec    = np.zeros(Nk2)
    Ev    = np.zeros(Nk2)
    psi_c = np.zeros((Nk2, n_orb), dtype=np.complex128)
    psi_v = np.zeros((Nk2, n_orb), dtype=np.complex128)

    # Reference eigenvectors at M (grid centre)
    ik_center = Nk2 // 2
    ham_M = _ham(n, kx_flat[ik_center], ky_flat[ik_center])
    evals_M, evecs_M = diagonalize_hamiltonian_fortran(ham_M)
    psi_c_ref = evecs_M[:, icb].copy()
    psi_v_ref = evecs_M[:, ivb].copy()
    Ec[ik_center]    = evals_M[icb]
    Ev[ik_center]    = evals_M[ivb]
    psi_c[ik_center] = psi_c_ref
    psi_v[ik_center] = psi_v_ref

    for ik in range(Nk2):
        if ik == ik_center:
            continue
        ham = _ham(n, kx_flat[ik], ky_flat[ik])
        evals, evecs = diagonalize_hamiltonian_fortran(ham)
        # Track CB: band with maximum |<psi_c_ref|evecs[:,j]>|
        olap_c = np.abs(psi_c_ref.conj() @ evecs)   # (n_orb,)
        ic_k   = np.argmax(olap_c)
        # Track VB: band with maximum |<psi_v_ref|evecs[:,j]>|
        olap_v = np.abs(psi_v_ref.conj() @ evecs)
        iv_k   = np.argmax(olap_v)
        Ec[ik]    = evals[ic_k]
        Ev[ik]    = evals[iv_k]
        psi_c[ik] = evecs[:, ic_k]
        psi_v[ik] = evecs[:, iv_k]

    # Band gap at M (grid centre)
    E_gap = Ec[ik_center] - Ev[ik_center]

    # ── Build BSE Hamiltonian ─────────────────────────────────────────────────
    print(f"  n={n}: building {Nk2}×{Nk2} BSE matrix ...", flush=True)

    # Diagonal block: free e-h pair excitation energies
    H_bse = np.diag(Ec - Ev)

    # Form factors F[k,k'] = |<c,k|c,k'>| × |<v,k|v,k'>|
    # Intraband overlaps: near 1 for small k-k', decays smoothly for large q.
    # The correct BSE kernel uses <u_c,k|u_c,k'> * <u_v,k'|u_v,k>,
    # taking the modulus gives a real symmetric F without gauge ambiguity.
    S_c = psi_c.conj() @ psi_c.T   # S_c[ik,ik'] = <c,k|c,k'>  (Nk2,Nk2)
    S_v = psi_v.conj() @ psi_v.T   # S_v[ik,ik'] = <v,k|v,k'>  (Nk2,Nk2)
    F   = np.abs(S_c) * np.abs(S_v)  # (Nk2,Nk2), real ≥ 0, =1 on diagonal

    # q = |k - k'| in nm⁻¹
    dkx_nm = (kx_flat[:, None] - kx_flat[None, :]) * 10.0
    dky_nm = (ky_flat[:, None] - ky_flat[None, :]) * 10.0
    q_nm   = np.sqrt(dkx_nm ** 2 + dky_nm ** 2)

    # V_RK matrix [eV·nm²]; q=0 diagonal → 0 (no self-interaction)
    Vrk = V_RK_q(q_nm, r0)
    np.fill_diagonal(Vrk, 0.0)

    # Phase-space factor: (dk [nm⁻¹])^2 / (2*pi)^2  →  [nm^-2 / nm^-2] = 1
    # V_RK [eV·nm²] × (dk)^2/(2pi)^2 [nm^-2] = eV
    scale  = dk_step_nm ** 2 / (2.0 * np.pi) ** 2
    H_bse += Vrk * F * scale   # Vrk < 0 → lowers eigenvalues (attractive)

    # H_bse is real-symmetric (V_RK real, F real); eigh handles real arrays
    # ── Solve BSE ─────────────────────────────────────────────────────────────
    print(f"  n={n}: diagonalizing {Nk2}×{Nk2} real symmetric BSE ...", flush=True)
    nb = min(n_states, Nk2)
    if nb < Nk2 // 4:
        evals_bse, evecs_bse = eigh(H_bse, subset_by_index=[0, nb - 1])
    else:
        evals_bse, _all_evecs = eigh(H_bse)
        evecs_bse = _all_evecs[:, :nb]
        evals_bse = evals_bse[:nb]

    # Excitation energies relative to the free e-h threshold (band gap at M)
    omega = evals_bse - E_gap   # < 0 for bound states

    E_bind_meV = -omega[0] * 1000.0   # positive [meV]

    # Oscillator strengths: f_s = |Σ_k A_s(k)|^2
    # (optical matrix element p_cv assumed constant across the small k-patch)
    osc = np.abs(evecs_bse.sum(axis=0)) ** 2   # (nb,)

    return dict(
        n          = n,
        E_bind_meV = E_bind_meV,
        omega      = omega,
        osc        = osc,
        E_gap      = E_gap,
        Nk2        = Nk2,
        dk_step_nm = dk_step_nm,
    )


# ── Run all n ─────────────────────────────────────────────────────────────────

def run_all(Nk=21, dk_max=2.0):
    """Solve BSE for n=1–5, print comparison table, return results dict."""
    print(f"\nPhase 5 Level-2 — k-space BSE exciton binding energies")
    print(f"  k-grid: {Nk}²={Nk**2} points,  dk_max={dk_max:.1f} nm⁻¹\n")
    hdr = (f"{'n':>3}  {'r₀(nm)':>7}  {'E_gap(eV)':>10}  "
           f"{'BSE-L2(meV)':>11}  {'L1(meV)':>8}  "
           f"{'Exp(meV)':>9}  {'Cho(meV)':>9}")
    print(hdr)
    print("─" * len(hdr))

    results = {}
    for n in range(1, 6):
        res          = solve_bse(n, Nk=Nk, dk_max=dk_max)
        E_l1, _      = solve_binding_energy(n)
        results[n]   = res
        cho_str      = f"{EL_BSE[n]:>9}" if n in EL_BSE else f"{'—':>9}"
        print(f"{n:>3}  {r0_nm(n):>7.2f}  {res['E_gap']:>10.4f}  "
              f"{res['E_bind_meV']:>11.1f}  {E_l1:>8.1f}  "
              f"{EL_EXP[n]:>9}  {cho_str}")

    return results


# ── Diagnostic plot ───────────────────────────────────────────────────────────

def plot_bse_comparison(results, savepath=None):
    """Bar chart: BSE Level-2 vs Level-1 vs experiment vs Cho reference."""
    if savepath is None:
        _here = os.path.dirname(os.path.abspath(__file__))
        savepath = os.path.join(_here, "..", "..", "results", "figures",
                                "bse_binding_energies.png")
    ns    = list(range(1, 6))
    E_l2  = [results[n]["E_bind_meV"] for n in ns]
    E_l1  = [solve_binding_energy(n)[0] for n in ns]
    E_exp = [EL_EXP[n] for n in ns]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x, w = np.array(ns), 0.20
    ax.bar(x - 1.5*w, E_l2,  w, label="BSE Level-2 (this work)",  color="#4c72b0")
    ax.bar(x - 0.5*w, E_l1,  w, label="R-K Level-1 (this work)",  color="#8172b3")
    ax.bar(x + 0.5*w, E_exp, w, label="Exp. Blancon 2018 (60T)",  color="#55a868")

    bse_x = [n for n in ns if n in EL_BSE]
    bse_y = [EL_BSE[n] for n in bse_x]
    ax.scatter(np.array(bse_x) + 1.5*w, bse_y, marker='D', s=60,
               color="#c44e52", zorder=5, label="Cho & Berkelbach 2019")

    ax.set_xlabel("n (layer number)", fontsize=12)
    ax.set_ylabel("Binding energy  (meV)", fontsize=12)
    ax.set_title("Exciton binding energy — BSE Level-2 comparison", fontsize=11)
    ax.set_xticks(ns)
    ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.savefig(savepath, dpi=150)
    print(f"Saved {savepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Nk=21: fast (~30 s), E_bind(n=1)~141 meV
    # Nk=41: better (~9 min), E_bind(n=1)~160 meV (converging toward Level-1)
    Nk = int(sys.argv[1]) if len(sys.argv) > 1 else 21
    results = run_all(Nk=Nk)
    plot_bse_comparison(results)
