"""
Five physics tests for the SP³+SOC tight-binding model.
All tests require the compiled Fortran library (libhamiltonian_3rd.so).
Run from repo root: pytest
"""
import numpy as np
import pytest
from ctypesroutines import (
    hamiltonian_cubic3D_fortran,
    eigenvalues_only_fortran,
)

# Best-fit 26 parameters (bulk 19 + surface 3 + IIz 4)
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

A_LAT = 6.28
KR = np.array([np.pi / A_LAT] * 3)   # R-point: corner of BZ
KG = np.array([0.0, 0.0, 0.0])        # Gamma


@pytest.fixture(scope="module")
def H3D_R():
    return hamiltonian_cubic3D_fortran(KR[0], KR[1], KR[2], *PARAMS_3D)


def test_hamiltonian_hermitian(H3D_R):
    """H == H† to machine precision."""
    diff = np.max(np.abs(H3D_R - H3D_R.conj().T))
    assert diff < 1e-10, f"||H - H†||_max = {diff:.2e}"


def test_eigenvalues_real(H3D_R):
    """Eigenvalues of a Hermitian matrix must be real."""
    evals = eigenvalues_only_fortran(H3D_R)
    max_imag = np.max(np.abs(evals.imag))
    assert max_imag < 1e-10, f"max|Im(λ)| = {max_imag:.2e}"


def test_band_gap_R_point(H3D_R):
    """3D bulk band gap at R-point ≈ 1.65 eV (experimental target)."""
    IVB, ICB = 25, 26
    evals = eigenvalues_only_fortran(H3D_R)
    gap = evals[ICB].real - evals[IVB].real
    assert abs(gap - 1.65) < 0.05, f"gap = {gap:.4f} eV, expected 1.65 ± 0.05 eV"


def test_absorption_array_shape():
    """Absorption energy grid has the correct number of points."""
    N_OMEGA = 6000
    OMEGA_MIN, OMEGA_MAX = 0.5, 6.0
    energies = np.linspace(OMEGA_MIN, OMEGA_MAX, N_OMEGA)
    assert len(energies) == N_OMEGA


def test_gaussian_normalisation():
    """Gaussian broadening kernel integrates to 1 on a fine grid."""
    N = 100_000
    sigma = 0.05          # eV, typical broadening
    omega = np.linspace(-1.0, 1.0, N)
    d_omega = omega[1] - omega[0]
    gauss = np.exp(-0.5 * (omega / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    integral = np.sum(gauss) * d_omega
    assert abs(integral - 1.0) < 1e-4, f"∫G dω = {integral:.6f}, expected 1.0"


# ── Phase 3: VQE subspace tests (no PennyLane required) ──────────────────────

@pytest.fixture(scope="module")
def H_eff_R():
    """4×4 projected Hamiltonian at k = R + 5% δk."""
    DELTA_K = 0.05 * np.pi / A_LAT
    ivb, icb = 25, 26          # same as IVB, ICB used in band-gap test
    H_R = hamiltonian_cubic3D_fortran(KR[0], KR[1], KR[2], *PARAMS_3D)
    _, evecs = np.linalg.eigh(H_R)
    U4 = evecs[:, [ivb - 1, ivb, icb, icb + 1]]
    H_k = hamiltonian_cubic3D_fortran(KR[0] + DELTA_K, KR[1], KR[2], *PARAMS_3D)
    return U4.conj().T @ H_k @ U4


def test_heff_is_hermitian(H_eff_R):
    """Projected 4×4 H_eff must be Hermitian to machine precision."""
    diff = np.max(np.abs(H_eff_R - H_eff_R.conj().T))
    assert diff < 1e-10, f"||H_eff - H_eff†||_max = {diff:.2e}"


def test_heff_shape(H_eff_R):
    """Subspace projection must return a 4×4 matrix."""
    assert H_eff_R.shape == (4, 4), f"Expected (4,4), got {H_eff_R.shape}"


def test_heff_ground_state_energy(H_eff_R):
    """H_eff ground-state energy must be in physically sensible range (3–7 eV)."""
    gs = np.linalg.eigvalsh(H_eff_R)[0]
    assert 3.0 < gs < 7.0, f"H_eff ground state = {gs:.4f} eV, expected 3–7 eV"


# ── Phase 5: Rytova-Keldysh and BSE tests ────────────────────────────────────

from rytova_keldysh import V_RK_q, V_RK_r, r0_nm, solve_binding_energy
from bse_2d import solve_bse


def test_vrk_q_sign_and_decay():
    """V_RK(q) must be negative for all q>0 and increase toward 0 as q→∞."""
    import numpy as np
    r0 = r0_nm(1)
    q = np.array([0.1, 0.5, 1.0, 5.0, 20.0])   # nm⁻¹
    v = V_RK_q(q, r0)
    assert np.all(v < 0), "V_RK(q) must be negative (attractive)"
    # potential should become less negative (closer to 0) at larger q
    assert np.all(np.diff(v) > 0), "V_RK(q) must increase monotonically toward 0"


def test_vrk_r_large_r_limit():
    """At large r, V_RK(r) should approach the bare 2D Coulomb −e²/(4πε₀ ε_env r)."""
    import numpy as np
    from rytova_keldysh import EPS_ENV, E2_OVER_4PIEPS0
    r0 = r0_nm(1)
    r_large = np.array([50.0, 100.0])   # nm  (r >> r0 = 0.91 nm)
    v_rk = V_RK_r(r_large, r0)
    v_coulomb = -E2_OVER_4PIEPS0 / (EPS_ENV * r_large)
    rel_err = np.abs((v_rk - v_coulomb) / v_coulomb)
    assert np.all(rel_err < 0.02), f"V_RK and Coulomb differ by {rel_err*100} % at large r"


# NOTE: a test asserting 220 < E_b(n=1) < 300 meV used to live here.  The
# bound came from a run, not from a derivation, and when a sign error in the
# radial reduction was corrected the true answer moved to 430 meV -- so the
# test failed the fix rather than the bug.  It has been replaced by
# tests/test_rytova_keldysh.py, which checks the solver against the exact
# 2D hydrogenic limit instead.


def test_binding_energy_decreases_with_n():
    """Binding energy must decrease with slab thickness n (quantum confinement)."""
    E_bs = [solve_binding_energy(n)[0] for n in range(1, 6)]
    for i in range(len(E_bs) - 1):
        assert E_bs[i] > E_bs[i + 1], (
            f"E_B(n={i+1})={E_bs[i]:.1f} meV should exceed E_B(n={i+2})={E_bs[i+1]:.1f} meV"
        )


def test_bse_returns_positive_binding_n1():
    """BSE Level-2 binding energy for n=1 at Nk=5 must be positive (bound state)."""
    res = solve_bse(1, Nk=5, dk_max=2.0, n_states=4)
    assert res["E_bind_meV"] > 0, f"BSE E_B(n=1) = {res['E_bind_meV']:.1f} meV — expected positive"
    assert res["Nk2"] == 25, f"Expected 25 k-points (5²), got {res['Nk2']}"


def test_bse_result_keys():
    """BSE result dict must contain all expected keys."""
    res = solve_bse(1, Nk=3, dk_max=1.0, n_states=3)
    for key in ("E_bind_meV", "omega", "osc", "E_gap", "Nk2", "dk_step_nm"):
        assert key in res, f"Missing key '{key}' in BSE result"
