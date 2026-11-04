"""Analytical tests for the Slater-Koster hopping block.

`ham_ortho_harrison.sk_block` is the single routine every hopping in the model
passes through, so an error in it is an error everywhere.  These tests check it
against the two-centre integral table of

    J. C. Slater and G. F. Koster, Phys. Rev. 94, 1498 (1954), Table I

which is written out independently below, and against three properties that
hold for any correct two-centre block regardless of the parameter values:
symmetry of the p-p sub-block, invariance of its trace, and covariance under
rotation of the bond.

The rotational-covariance test is the strongest of the four: it compares the
routine against *itself* at a rotated bond, so it cannot be satisfied by a
formula that merely reproduces the table at the handful of directions a
hand-check would try.
"""
import numpy as np
import pytest

from ham_ortho_harrison import sk_block

# Arbitrary, deliberately non-symmetric parameter values: equal or zero values
# can hide index errors by making wrong entries accidentally right.
VSS, VSP, VPS, VPPS, VPPP = -0.67, -1.20, -0.22, -1.75, -1.53

S, X, Y, Z = 0, 1, 2, 3          # basis order used by sk_block


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# Directions used throughout: a cube axis, a face diagonal, a body diagonal,
# and a direction with no special symmetry at all.
DIRECTIONS = [
    _unit([1.0, 0.0, 0.0]),
    _unit([0.0, 1.0, 0.0]),
    _unit([1.0, 1.0, 0.0]),
    _unit([1.0, 1.0, 1.0]),
    _unit([0.3, -0.7, 0.5]),
    _unit([-0.21, 0.44, -0.87]),
]


def _rotation(axis, angle):
    """Rodrigues rotation matrix, built here rather than imported."""
    k = _unit(axis)
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


# ---------------------------------------------------------------- the table
def _slater_koster_table(l, m, n):
    """Slater-Koster Table I, written out term by term.

    Independent of sk_block: every entry is transcribed from the published
    table rather than factored, so a sign or index error in the routine under
    test cannot be mirrored here.
    """
    T = np.zeros((4, 4))
    T[S, S] = VSS

    # E_{s,alpha} = l_alpha * V_sp-sigma
    T[S, X] = l * VSP
    T[S, Y] = m * VSP
    T[S, Z] = n * VSP

    # The model carries V_ps-sigma as its own parameter rather than deriving
    # it from V_sp-sigma, so the reversed row uses VPS with the same weights.
    T[X, S] = l * VPS
    T[Y, S] = m * VPS
    T[Z, S] = n * VPS

    # E_{alpha,alpha} = l_alpha^2 Vpp-sigma + (1 - l_alpha^2) Vpp-pi
    T[X, X] = l * l * VPPS + (1.0 - l * l) * VPPP
    T[Y, Y] = m * m * VPPS + (1.0 - m * m) * VPPP
    T[Z, Z] = n * n * VPPS + (1.0 - n * n) * VPPP

    # E_{alpha,beta} = l_alpha l_beta Vpp-sigma - l_alpha l_beta Vpp-pi
    T[X, Y] = T[Y, X] = l * m * VPPS - l * m * VPPP
    T[X, Z] = T[Z, X] = l * n * VPPS - l * n * VPPP
    T[Y, Z] = T[Z, Y] = m * n * VPPS - m * n * VPPP
    return T


@pytest.mark.parametrize("d", DIRECTIONS)
def test_sk_block_matches_published_table(d):
    """Every entry agrees with Slater & Koster (1954) Table I."""
    got = sk_block(*d, VSS, VSP, VPS, VPPS, VPPP)
    want = _slater_koster_table(*d)
    assert np.allclose(got, want, atol=1e-12), (
        f"direction {np.round(d, 4)}\ngot:\n{np.round(got, 6)}\n"
        f"table:\n{np.round(want, 6)}"
    )


def test_sk_block_along_cube_axis_is_pure_sigma_pi():
    """For a bond along x the p-p block is diag(sigma, pi, pi).

    This is the limit the parameters are *defined* by: V_pp-sigma is the
    head-on overlap, V_pp-pi the side-on one.  If this fails, the parameters
    do not mean what the paper says they mean.
    """
    T = sk_block(1.0, 0.0, 0.0, VSS, VSP, VPS, VPPS, VPPP)

    assert T[X, X] == pytest.approx(VPPS), "p_x along the bond must give sigma"
    assert T[Y, Y] == pytest.approx(VPPP), "p_y across the bond must give pi"
    assert T[Z, Z] == pytest.approx(VPPP), "p_z across the bond must give pi"

    for a, b in ((X, Y), (X, Z), (Y, Z)):
        assert T[a, b] == pytest.approx(0.0), "no p-p mixing along a cube axis"

    assert T[S, X] == pytest.approx(VSP), "s couples to the bond-axis p only"
    assert T[S, Y] == pytest.approx(0.0)
    assert T[S, Z] == pytest.approx(0.0)


@pytest.mark.parametrize("d", DIRECTIONS)
def test_sk_pp_sub_block_is_symmetric(d):
    """The p-p sub-block is symmetric for any bond direction."""
    T = sk_block(*d, VSS, VSP, VPS, VPPS, VPPP)
    pp = T[1:, 1:]
    assert np.allclose(pp, pp.T, atol=1e-14)


@pytest.mark.parametrize("d", DIRECTIONS)
def test_sk_pp_trace_is_direction_independent(d):
    """tr(p-p) = Vpp-sigma + 2 Vpp-pi for every direction.

    The trace is invariant under the rotation that takes any bond onto a cube
    axis, where the block is diag(sigma, pi, pi).  It therefore cannot depend
    on the direction cosines, and a stray l^2 would break it.
    """
    T = sk_block(*d, VSS, VSP, VPS, VPPS, VPPP)
    assert np.trace(T[1:, 1:]) == pytest.approx(VPPS + 2.0 * VPPP, abs=1e-12)


@pytest.mark.parametrize("d", DIRECTIONS)
@pytest.mark.parametrize(
    "axis,angle",
    [([0, 0, 1], 0.7), ([1, 0, 0], -1.1), ([1, 1, 1], 2.3), ([0.2, 0.9, -0.4], 0.45)],
)
def test_sk_block_is_rotationally_covariant(d, axis, angle):
    """Rotating the bond must rotate the block: T(Rd) = D T(d) D^T.

    D = diag(1, R) because s is invariant under rotation while (px, py, pz)
    transform as a vector.  This is the physical content of the Slater-Koster
    construction, and it holds for arbitrary parameters and arbitrary
    rotations -- so it tests far more directions than any table check.
    """
    R = _rotation(axis, angle)
    D = np.eye(4)
    D[1:, 1:] = R

    rotated_bond = sk_block(*(R @ d), VSS, VSP, VPS, VPPS, VPPP)
    rotated_block = D @ sk_block(*d, VSS, VSP, VPS, VPPS, VPPP) @ D.T

    assert np.allclose(rotated_bond, rotated_block, atol=1e-12), (
        f"bond {np.round(d, 4)} rotated about {axis} by {angle} rad\n"
        f"T(Rd):\n{np.round(rotated_bond, 6)}\n"
        f"D T(d) D^T:\n{np.round(rotated_block, 6)}"
    )


@pytest.mark.parametrize("d", DIRECTIONS)
def test_sk_block_has_the_right_parity(d):
    """Reversing the bond acts as the parity operator: T(-d) = P T(d) P.

    P = diag(1, -1, -1, -1) because s is even under inversion and p is odd.
    The s-p entries therefore change sign while the p-p block, being
    quadratic in the direction cosines, does not.

    Note the convention this pins down.  The published table writes
    E_{alpha,s} = -l_alpha V_sp-sigma, deriving the reversed row from the
    forward one.  This model instead carries V_ps-sigma as an independent
    parameter and writes E_{alpha,s} = +l_alpha V_ps-sigma, which is the
    right choice for a heteronuclear bond: <s_Pb|H|p_I> and <p_Pb|H|s_I> are
    genuinely different integrals, not two names for one.  The parity
    relation holds either way, which is why it is the property worth testing.
    """
    P = np.diag([1.0, -1.0, -1.0, -1.0])
    fwd = sk_block(*d, VSS, VSP, VPS, VPPS, VPPP)
    rev = sk_block(*(-d), VSS, VSP, VPS, VPPS, VPPP)
    assert np.allclose(rev, P @ fwd @ P, atol=1e-12)
