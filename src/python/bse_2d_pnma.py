"""
bse_2d_pnma.py — k-space BSE on Pnma TB Gen 140 (Option C of Phase 7A).

Builds the BSE Hamiltonian on a 2D k-patch around Γ for each Pnma RP slab,
using the Gen 140 sp³+SOC + Harrison Hamiltonian from ham_ortho_harrison.

    H_BSE[k,k'] = (Ec(k) - Ev(k)) · δ_{k,k'}
                + V_RK(|k-k'|) · F(k,k') · (dk)² / (2π)²

The form factor F(k,k') = |<ψ_c,k | ψ_c,k'>| · |<ψ_v,k | ψ_v,k'>| comes from
the spinor eigenvectors of the Pnma TB Hamiltonian.

Compared to bse_2d.py (which used the cubic Hamiltonian and built the patch
around the M point), this version:
  - uses the Pnma Hamiltonian (ham_ortho_harrison.hamiltonian)
  - centers the k-patch at Γ (where Pnma slab band edges live)
  - uses Gen 140 parameters

Output: results/data/phase7a_bse.npz

Run:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 bse_2d_pnma.py [Nk]   (default Nk=15)
"""
import os, sys, time
import numpy as np
from scipy.linalg import eigh

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ham_ortho_harrison as ham_ortho
from rytova_keldysh import V_RK_q, r0_nm

# ── Gen 140 parameters (canonical) ────────────────────────────────────────────
GEN140 = dict(zip(ham_ortho.PARAM_KEYS, [
    -2.79, +3.29, -10.37, -4.06,
    -0.67, -1.20, -0.22, -1.75, -1.53,
    +1.30, +0.90,
    -0.76, -0.22, -0.14, +0.10,
    -0.05, +0.33, +0.19, -0.49,
    0.0, 0.0, 0.0, 0.0,
    -9.38, -2.49, +3.18,
    +2.36,
]))

LABELS = ['n1', 'n2', 'n3', 'n4', 'n5']


def vasp_path_for(label):
    if label == '3D':
        return os.path.join(_HERE, '..', '..', 'data', 'pos3D2Dtilted', 'POSCAR')
    return os.path.join(_HERE, '..', '..', 'data', 'pos3D2Dpnma_xyz',
                        f'pnma_xyz_{label}.vasp')


def n_fill(geom):
    """Number of filled states (counting spin) = 2*n_Pb + 8*n_I."""
    return geom['n_Pb'] * 2 + geom['n_I'] * 8


def in_plane_recip(label, geom):
    """In-plane reciprocal lattice vectors b1_2d, b2_2d [Å⁻¹]."""
    from ase.io import read
    cell = read(vasp_path_for(label)).cell.array
    b1, b2 = ham_ortho.in_plane_reciprocal_vectors(cell, geom['slab_normal'])
    return b1, b2


def diagonalize_at_k(label, kvec, params, geom):
    """Return (eigvals, eigvecs) of the Pnma TB Hamiltonian at k."""
    H = ham_ortho.hamiltonian(label, kvec, params, geom=geom)
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs


def solve_bse_pnma(label, Nk=15, dk_max_invnm=2.0, n_states=10):
    """
    Build and solve the BSE on a 2D k-patch around Γ for a Pnma slab.

    Parameters
    ----------
    label        : '3D' or 'n1'..'n5'
    Nk           : grid points per in-plane direction (total Nk² k-points)
    dk_max_invnm : half-width of k-patch around Γ  [nm⁻¹]
    n_states     : number of lowest BSE eigenvalues to return

    Returns
    -------
    dict with E_b_meV (binding energy), E_gap, omega, oscillators, etc.
    """
    t_start = time.perf_counter()
    print(f"\n  {label}: Loading geometry ...", flush=True)
    geom = ham_ortho.load_geometry(label)
    nf   = n_fill(geom)
    ivb, icb = nf - 1, nf

    # Get in-plane reciprocal vectors
    b1, b2 = in_plane_recip(label, geom)
    e1 = b1 / np.linalg.norm(b1)
    e2 = b2 / np.linalg.norm(b2)
    # Unit vectors in Cartesian — projections on b1, b2 give k in those directions

    # Build a 2D k-patch around Γ (size dk_max in each direction)
    # We work in Cartesian k (Å⁻¹), with the patch span dk_max [nm⁻¹] = dk_max/10 [Å⁻¹]
    dk_max_ang = dk_max_invnm / 10.0
    dkv = np.linspace(-dk_max_ang, dk_max_ang, Nk)
    dk_step_invnm = (dkv[1] - dkv[0]) * 10.0   # spacing in nm⁻¹

    # k = i * dk1 * e1 + j * dk2 * e2  (Cartesian)
    K_list = []
    for i in range(Nk):
        for j in range(Nk):
            kvec = dkv[i] * e1 + dkv[j] * e2
            K_list.append(kvec)
    K_array = np.array(K_list)   # (Nk², 3)
    Nk2 = Nk * Nk

    # Index of Γ (centre of grid)
    ik_center = Nk2 // 2

    print(f"  {label}: TB diagonalization at {Nk2} k-points "
          f"(dim={8 * (geom['n_Pb'] + geom['n_I'])}) ...", flush=True)

    # Diagonalize TB at each k, with band tracking
    H_M = ham_ortho.hamiltonian(label, K_array[ik_center], GEN140, geom=geom)
    n_orb = H_M.shape[0]
    evals_M, evecs_M = np.linalg.eigh(H_M)
    psi_c_ref = evecs_M[:, icb].copy()
    psi_v_ref = evecs_M[:, ivb].copy()
    Ec = np.zeros(Nk2)
    Ev = np.zeros(Nk2)
    psi_c = np.zeros((Nk2, n_orb), dtype=complex)
    psi_v = np.zeros((Nk2, n_orb), dtype=complex)
    Ec[ik_center] = evals_M[icb]
    Ev[ik_center] = evals_M[ivb]
    psi_c[ik_center] = psi_c_ref
    psi_v[ik_center] = psi_v_ref

    t_diag_start = time.perf_counter()
    for ik in range(Nk2):
        if ik == ik_center:
            continue
        H_k = ham_ortho.hamiltonian(label, K_array[ik], GEN140, geom=geom)
        evals, evecs = np.linalg.eigh(H_k)
        # Track bands by maximum overlap with reference at Γ
        ol_c = np.abs(psi_c_ref.conj() @ evecs)
        ic_k = int(np.argmax(ol_c))
        ol_v = np.abs(psi_v_ref.conj() @ evecs)
        iv_k = int(np.argmax(ol_v))
        Ec[ik] = evals[ic_k]
        Ev[ik] = evals[iv_k]
        psi_c[ik] = evecs[:, ic_k]
        psi_v[ik] = evecs[:, iv_k]
    t_diag = time.perf_counter() - t_diag_start
    print(f"  {label}: diagonalization done in {t_diag:.1f} s", flush=True)

    E_gap = Ec[ik_center] - Ev[ik_center]
    print(f"  {label}: E_gap(Γ) = {E_gap:.4f} eV", flush=True)

    # ── R-K potential ─────────────────────────────────────────────────────────
    n_int = int(label[1:])   # n1 → 1, n5 → 5
    r0 = r0_nm(n_int)

    # Compute |q| = |k - k'| in nm⁻¹ for all pairs
    dkx_nm = (K_array[:, 0:1] - K_array[None, :, 0]) * 10.0
    dky_nm = (K_array[:, 1:2] - K_array[None, :, 1]) * 10.0
    dkz_nm = (K_array[:, 2:3] - K_array[None, :, 2]) * 10.0
    q_nm = np.sqrt(dkx_nm**2 + dky_nm**2 + dkz_nm**2)

    Vrk = V_RK_q(q_nm, r0)
    np.fill_diagonal(Vrk, 0.0)

    # Form factor F(k,k')
    S_c = psi_c.conj() @ psi_c.T     # (Nk², Nk²)
    S_v = psi_v.conj() @ psi_v.T
    F = np.abs(S_c) * np.abs(S_v)
    # Ensure real symmetric
    F = (F + F.T) / 2.0

    # BSE Hamiltonian
    print(f"  {label}: building {Nk2}×{Nk2} BSE matrix and diagonalizing ...",
          flush=True)
    scale = dk_step_invnm**2 / (2.0 * np.pi)**2
    H_bse = np.diag(Ec - Ev) + (Vrk * F * scale)
    H_bse = (H_bse + H_bse.T) / 2.0   # enforce symmetry

    # Solve
    nb = min(n_states, Nk2)
    if nb < Nk2 // 4:
        evals_bse, evecs_bse = eigh(H_bse, subset_by_index=[0, nb - 1])
    else:
        evals_bse, _all = eigh(H_bse)
        evecs_bse = _all[:, :nb]
        evals_bse = evals_bse[:nb]

    omega = evals_bse - E_gap   # bound states have omega < 0
    E_b_meV = -omega[0] * 1000.0

    # Oscillator strengths (matrix element constant approximation)
    osc = np.abs(evecs_bse.sum(axis=0))**2

    t_total = time.perf_counter() - t_start
    print(f"  {label}: BSE done in {t_total:.1f} s   "
          f"E_b = {E_b_meV:.1f} meV", flush=True)

    return dict(
        label=label,
        n=n_int,
        E_b_meV=E_b_meV,
        E_gap_eV=E_gap,
        omega_eV=omega,
        osc=osc,
        Nk2=Nk2,
        dk_step_invnm=dk_step_invnm,
        time_s=t_total,
    )


def main():
    Nk = int(sys.argv[1]) if len(sys.argv) > 1 else 15

    # Blancon E_b for comparison
    EB_EXP = {1: 476, 2: 252, 3: 176, 4: 158, 5: 121}
    # Cho & Berkelbach 2019 BSE benchmark
    EB_CHO = {1: 302, 5: 97}
    # R-K Option A results (from phase7a_excitons.npz)
    RK_path = os.path.join(_HERE, '..', '..', 'results', 'data',
                            'phase7a_excitons.npz')
    if os.path.exists(RK_path):
        rk_data = np.load(RK_path)
        EB_RK = {int(rk_data['n_vals'][i]): float(rk_data['E_b_RK_meV'][i])
                  for i in range(5)}
    else:
        EB_RK = {n: None for n in range(1, 6)}

    print("=" * 80)
    print(f"Phase 7A Option C — k-space BSE on Pnma Gen 140  (Nk={Nk})")
    print("=" * 80)

    results = {}
    for label in LABELS:
        try:
            res = solve_bse_pnma(label, Nk=Nk, dk_max_invnm=2.0)
            results[label] = res
        except Exception as e:
            print(f"  {label}: FAILED — {e}", flush=True)
            results[label] = None

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: BSE on Pnma Gen 140 vs R-K (Option A) vs Blancon vs Cho")
    print("=" * 80)
    print(f"{'n':>3}  {'E_g(Γ)':>9}  {'E_b BSE':>9}  {'E_b R-K(A)':>11}  "
          f"{'E_b Cho':>8}  {'E_b Blancon':>12}")
    for label in LABELS:
        n = int(label[1:])
        r = results.get(label)
        if r is None:
            continue
        cho = f"{EB_CHO[n]}" if n in EB_CHO else "—"
        print(f"{n:>3}  {r['E_gap_eV']:>9.4f}  {r['E_b_meV']:>9.1f}  "
              f"{EB_RK[n]:>11.1f}  {cho:>8}  {EB_EXP[n]:>12d}")

    # Save
    eb_bse = np.array([results[lab]['E_b_meV'] if results[lab] else np.nan
                       for lab in LABELS])
    eg_bse = np.array([results[lab]['E_gap_eV'] if results[lab] else np.nan
                       for lab in LABELS])
    out_path = os.path.join(_HERE, '..', '..', 'results', 'data',
                             'phase7a_bse.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
             n_vals=np.arange(1, 6),
             E_b_BSE_meV=eb_bse,
             E_gap_BSE_eV=eg_bse,
             Nk=Nk,
             E_b_blancon_meV=np.array([EB_EXP[n] for n in range(1, 6)]),
             E_b_cho_meV=np.array([EB_CHO.get(n, np.nan) for n in range(1, 6)]),
             E_b_RK_meV=np.array([EB_RK.get(n, np.nan) for n in range(1, 6)]),
             )
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
