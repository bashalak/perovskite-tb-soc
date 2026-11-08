"""Analytical tests for the Rytova-Keldysh exciton solver.

The solver previously carried a sign error: the 2D m=0 radial reduction was
written with a centrifugal term +hbar^2/(8 mu r^2) instead of minus.  It was
guarded by a test asserting that the n=1 binding energy fell between 220 and
300 meV -- a number remembered from a run, not derived.  When the sign was
corrected the answer moved to 430 meV and that test went red, so the test was
defending the bug rather than the physics.

What follows replaces it with a check against a result that can be derived.
As the screening length r0 goes to zero the Rytova-Keldysh potential becomes
the bare 2D Coulomb interaction, whose ground state is known exactly:

    E_1s = -4 Ry*,      Ry* = mu e^4 / (2 (4 pi eps0 eps)^2 hbar^2)

The expected value is computed from the module's own constants, so there is
no number here to go stale.

Note on the choice of mass.  A small reduced mass is used, which makes the
test exciton large and diffuse: its radius a*/2 is then a few nm rather than
the 0.3 nm of the physical system, so a coarse grid resolves it and each test
runs in a fraction of a second.  The analytical result is exact for any mass,
so nothing about the physics being tested changes -- only the cost.
"""
import numpy as np
import pytest

import rytova_keldysh as rk


def _coulomb_strength():
    """A = e^2 / (4 pi eps0 eps_env)  [eV nm], the 1/r prefactor at large r."""
    return rk.E2_OVER_4PIEPS0 / rk.EPS_ENV


def _exact_2d_ground_state_meV(mu):
    """-4 Ry* in meV, from the module's own constants.

    With A = e^2/(4 pi eps0 eps) and B = hbar^2/(2 mu), the 3D Rydberg is
    Ry* = A^2 / (4B), and the 2D hydrogenic ground state is four times that.
    """
    A = _coulomb_strength()
    B = rk.HBAR2_OVER_2ME / mu
    return (A * A / B) * 1000.0


def _exciton_radius_nm(mu):
    """a*/2, the 2D hydrogenic ground-state radius [nm]."""
    return (rk.HBAR2_OVER_2ME / mu) / _coulomb_strength()


@pytest.mark.parametrize("mu,r0,tol_pct", [
    (0.05, 0.005, 2.0),
    (0.02, 0.005, 2.0),
    (0.02, 0.020, 3.5),
])
def test_coulomb_limit_matches_2d_hydrogen(monkeypatch, mu, r0, tol_pct):
    """With negligible screening the solver reproduces E = -4 Ry*.

    This is the test the old regression bound should always have been: it
    pins the radial reduction, the mass handling and the discretisation
    against a result derived on paper rather than observed once.
    """
    monkeypatch.setattr(rk, "r0_nm", lambda n, **kw: r0)

    E_b, _ = rk.solve_binding_energy(1, r_max=60.0, Nr=6000, mu=mu)
    exact = _exact_2d_ground_state_meV(mu)
    err_pct = 100.0 * abs(E_b - exact) / exact

    assert err_pct < tol_pct, (
        f"mu={mu}, r0={r0} nm (exciton radius {_exciton_radius_nm(mu):.2f} nm): "
        f"got {E_b:.2f} meV, exact -4Ry* = {exact:.2f} meV, off by {err_pct:.2f}%"
    )


def test_binding_approaches_the_exact_result_as_screening_vanishes(monkeypatch):
    """Shrinking r0 moves the answer towards -4 Ry*, not away from it.

    The residual discrepancy in the test above is physical, not numerical:
    at finite r0 the potential is weaker than Coulomb at short range, so the
    state is slightly less bound. Reducing r0 must therefore close the gap
    monotonically. A solver that happened to land near the right number for
    the wrong reason would not show this trend.
    """
    mu = 0.02
    exact = _exact_2d_ground_state_meV(mu)

    errors = []
    for r0 in (0.04, 0.02, 0.01, 0.005):
        monkeypatch.setattr(rk, "r0_nm", lambda n, _v=r0, **kw: _v)
        E_b, _ = rk.solve_binding_energy(1, r_max=60.0, Nr=6000, mu=mu)
        errors.append(exact - E_b)          # positive: under-bound at finite r0

    assert all(e > 0 for e in errors), (
        f"finite screening must under-bind relative to bare Coulomb, got {errors}"
    )
    for a, b in zip(errors, errors[1:]):
        assert b < a, f"error should shrink as r0 shrinks, got {errors}"


def test_binding_energies_are_grid_converged():
    """The production grid is fine enough that the answer no longer moves.

    Guards the discretisation rather than the value: if someone changes the
    finite-volume scheme and it stops converging, this fails even though the
    number it converges to is not asserted anywhere.
    """
    coarse, _ = rk.solve_binding_energy(1, r_max=120.0, Nr=4000)
    fine, _ = rk.solve_binding_energy(1, r_max=120.0, Nr=8000)

    assert abs(coarse - fine) < 1.0, (
        f"not grid converged: Nr=4000 gives {coarse:.2f} meV, "
        f"Nr=8000 gives {fine:.2f} meV"
    )


def test_screening_length_follows_cudazzo():
    """r0 = (eps_slab - eps_env) d / 2, not the Keldysh-1979 form.

    The two conventions differ by roughly a factor of three here, and the
    change was part of the same correction. Spelling it out keeps a silent
    revert from passing unnoticed.
    """
    for n, d in rk.D_NM.items():
        assert rk.r0_nm(n) == pytest.approx(
            (rk.EPS_SLAB - rk.EPS_ENV) * d / 2.0
        ), f"n={n}"
