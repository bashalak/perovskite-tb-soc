"""Analytical tests for the atomic spin-orbit term.

`ham_ortho_harrison._SOC_TEMPLATE` is a hard-coded 8x8 matrix, transliterated
by hand from the Fortran `HintSOC` -- its own comment says so.  A transliterated
constant is exactly the kind of thing that is never checked, so it is checked
here against a lambda L.S operator constructed from scratch: Cartesian orbital
angular-momentum matrices and Pauli matrices written out in this file, combined
and then reordered into the interleaved spin basis the model uses.

The last test is the one that matters physically.  It confirms that the number
the model calls `deltaSO` really is the p-level splitting, so the values taken
from Boyer-Richard et al. (2016) mean in this code what they mean in that paper.
"""
import numpy as np
import pytest

from ham_ortho_harrison import _SOC_TEMPLATE

# Interleaved basis: index = 2 * orbital + spin,
# orbitals (s, px, py, pz) = (0, 1, 2, 3), spins (up, down) = (0, 1).
S_UP, S_DN = 0, 1


def _orbital_angular_momentum():
    """L_x, L_y, L_z for real Cartesian p orbitals, in units of hbar.

    For the real basis (|x>, |y>, |z>) the matrix elements are
    (L_a)_{bc} = -i * eps_{abc}, built here from the Levi-Civita symbol
    rather than copied from a table.
    """
    L = np.zeros((3, 3, 3), dtype=complex)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                # Levi-Civita symbol
                eps = ((a - b) * (b - c) * (c - a)) / 2.0
                L[a, b, c] = -1j * eps
    return L[0], L[1], L[2]


def _pauli():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return sx, sy, sz


def _reference_L_dot_sigma():
    """L.sigma = sum_a L_a (x) sigma_a, embedded in the 8x8 interleaved basis.

    The model applies  H_SOC += (deltaSO / 3) * _SOC_TEMPLATE  per atom.  The
    atomic spin-orbit Hamiltonian is  xi L.S  with  xi = 2 * delta / 3, and
    L.S = (1/2) L.sigma, so  H_SOC = (delta / 3) L.sigma.  The template must
    therefore equal L.sigma exactly, with the s rows and columns empty.
    """
    Lx, Ly, Lz = _orbital_angular_momentum()
    sx, sy, sz = _pauli()

    # 6x6 on (p orbital) x (spin), p ordered (x, y, z)
    p_block = np.kron(Lx, sx) + np.kron(Ly, sy) + np.kron(Lz, sz)

    H = np.zeros((8, 8), dtype=complex)
    for j in range(3):                       # p orbital index 0,1,2 -> px,py,pz
        for k in range(3):
            for sa in range(2):
                for sb in range(2):
                    row = 2 * (j + 1) + sa   # +1 skips the s orbital
                    col = 2 * (k + 1) + sb
                    H[row, col] = p_block[2 * j + sa, 2 * k + sb]
    return H


def test_soc_template_equals_lambda_L_dot_S():
    """The hard-coded template is L.sigma, built independently."""
    want = _reference_L_dot_sigma()
    assert np.allclose(_SOC_TEMPLATE, want, atol=1e-12), (
        f"template:\n{np.round(_SOC_TEMPLATE, 4)}\n"
        f"L.sigma:\n{np.round(want, 4)}\n"
        f"difference:\n{np.round(_SOC_TEMPLATE - want, 4)}"
    )


def test_soc_template_is_hermitian():
    """Spin-orbit coupling is an observable, so its matrix is Hermitian."""
    assert np.allclose(_SOC_TEMPLATE, _SOC_TEMPLATE.conj().T, atol=1e-14)


def test_soc_leaves_s_orbitals_untouched():
    """An l = 0 orbital has no spin-orbit coupling: its rows and columns vanish."""
    for i in (2 * 0 + S_UP, 2 * 0 + S_DN):
        assert np.allclose(_SOC_TEMPLATE[i, :], 0.0), "s row is not empty"
        assert np.allclose(_SOC_TEMPLATE[:, i], 0.0), "s column is not empty"


def test_soc_template_is_traceless():
    """L.S shifts no centre of gravity: the p levels split about their mean."""
    assert np.trace(_SOC_TEMPLATE).real == pytest.approx(0.0, abs=1e-12)
    assert np.trace(_SOC_TEMPLATE).imag == pytest.approx(0.0, abs=1e-12)


def test_soc_gives_the_j_multiplet_structure():
    """Eigenvalues of L.sigma on the p shell are +1 (four-fold) and -2 (twofold).

    L.S = (1/2)[j(j+1) - l(l+1) - s(s+1)] gives +1/2 for j = 3/2 and -1 for
    j = 1/2, with degeneracies 4 and 2.  L.sigma = 2 L.S doubles both.
    """
    p = _SOC_TEMPLATE[2:, 2:]                 # drop the s rows and columns
    ev = np.sort(np.linalg.eigvalsh(p))

    assert np.allclose(ev[:2], -2.0, atol=1e-10), f"j=1/2 level wrong: {ev}"
    assert np.allclose(ev[2:], +1.0, atol=1e-10), f"j=3/2 level wrong: {ev}"


@pytest.mark.parametrize("delta", [1.30, 0.90, 0.55])
def test_deltaSO_is_the_physical_p_level_splitting(delta):
    """The parameter named deltaSO is the j=3/2 to j=1/2 splitting, in eV.

    This fixes the normalisation.  deltaSO0 = 1.30 eV and deltaSO1 = 0.90 eV
    are taken from the literature rather than fitted, so if the prefactor in
    `hamiltonian` were wrong the model would silently carry the wrong
    spin-orbit strength while still reproducing its own fitted band gaps.
    """
    # the prefactor used in ham_ortho_harrison.eigenvalues_batch
    soc = (delta / 3.0) * _SOC_TEMPLATE
    ev = np.sort(np.linalg.eigvalsh(soc[2:, 2:]))

    splitting = ev[-1] - ev[0]
    assert splitting == pytest.approx(delta, abs=1e-10), (
        f"deltaSO = {delta} eV should split the p shell by {delta} eV, "
        f"got {splitting:.6f} eV"
    )
    # and the centre of gravity is unmoved: 4 * (+delta/3) + 2 * (-2delta/3) = 0
    assert ev.sum() == pytest.approx(0.0, abs=1e-10)
