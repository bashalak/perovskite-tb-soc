"""Cross-implementation tests for the Hamiltonian builders.

The model is written twice over, and neither copy was checked against the
other anywhere in the repository:

  * `ham_ortho_harrison.hamiltonian` builds H(k) one k-point at a time;
    `ham_ortho_harrison.eigenvalues_batch` rebuilds the same operator in a
    vectorised form for many k at once.  They are maintained by hand, side by
    side, and the production fit calls only the second one.
  * The Fortran routines in `src/fortran` build the same model for an
    idealised cubic lattice.  The Python builder generalises them to arbitrary
    bond directions; in the cubic limit the two must agree.

Both comparisons use eigenvalues rather than matrix elements, because the two
implementations order their basis differently.  Eigenvalues do not depend on
that choice, so agreement is evidence about the physics rather than about the
bookkeeping.

The cubic cell is constructed here rather than read from `data/`, so these
tests need no structure file and no ASE.
"""
import numpy as np
import pytest

import ham_ortho_harrison as ham

A_LAT = 6.28                      # cubic lattice constant, A
D_PBI = A_LAT / 2.0               # Pb-I nearest-neighbour distance, A
LABEL = "cubic_builder_test"      # not '3D', so it cannot collide in the cache

# A deliberately generic parameter set.  Second-neighbour and surface terms are
# zero: the cubic Fortran routine has no I-I-along-z or surface parameters, so
# switching them off is what makes the two models the same model.
PARAMS = dict(
    Es0=-2.79, Ep0=+3.29, Es1=-10.37, Ep1=-4.06,
    Vss=-0.67, Vsp=-1.20, Vps=-0.22, Vppsig=-1.75, Vpppi=-1.53,
    deltaSO0=1.30, deltaSO1=0.90,
    Vss_II=0.0, Vsp_II=0.0, Vppsig_II=0.0, Vpppi_II=0.0,
    Vss_IIz=0.0, Vsp_IIz=0.0, Vppsig_IIz=0.0, Vpppi_IIz=0.0,
    Vss_PbPb=0.0, Vsp_PbPb=0.0, Vppsig_PbPb=0.0, Vpppi_PbPb=0.0,
    Es1_ap=-10.37, Ep1_ap=-4.06, Ep0_surf=+3.29,
    n_harrison=0.0,               # constant Slater-Koster, as in the Fortran
)

# Order expected by hamiltonian_cubic3D_fortran (19 arguments, no IIz, no surface)
PARAMS_FORTRAN = [
    PARAMS["Es0"], PARAMS["Ep0"], PARAMS["Es1"], PARAMS["Ep1"],
    PARAMS["Vss"], PARAMS["Vsp"], PARAMS["Vps"], PARAMS["Vppsig"], PARAMS["Vpppi"],
    PARAMS["deltaSO0"], PARAMS["deltaSO1"],
    PARAMS["Vss_II"], PARAMS["Vsp_II"], PARAMS["Vppsig_II"], PARAMS["Vpppi_II"],
    PARAMS["Vss_PbPb"], PARAMS["Vsp_PbPb"], PARAMS["Vppsig_PbPb"], PARAMS["Vpppi_PbPb"],
]

# Cartesian k-points, 1/A.  Gamma, three zone-boundary points, the corner, and
# one low-symmetry point where accidental degeneracies cannot hide a mismatch.
KPOINTS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([np.pi / A_LAT, 0.0, 0.0]),
    np.array([0.0, np.pi / A_LAT, 0.0]),
    np.array([np.pi / A_LAT, np.pi / A_LAT, 0.0]),
    np.array([np.pi / A_LAT] * 3),
    np.array([0.31, -0.17, 0.08]),
]


def _cubic_geometry():
    """One formula unit of cubic PbI3: Pb at the origin, I on the cube edges.

    Each Pb carries six bonds -- +-x, +-y, +-z -- and the two bonds along a
    given axis point at the same iodine in neighbouring cells, which is how
    the geometry files produced by parse_geometry.py are laid out.
    """
    bonds = []
    for axis, i_idx in enumerate((1, 2, 3)):
        for sign in (+1.0, -1.0):
            dr = [0.0, 0.0, 0.0]
            dr[axis] = sign * D_PBI
            lmn = [0.0, 0.0, 0.0]
            lmn[axis] = sign
            bonds.append(dict(I_idx=i_idx, dr_ang=dr, lmn=lmn,
                              dist_ang=D_PBI, bond_type="equatorial"))
    return dict(
        label=LABEL,
        lattice=[[A_LAT, 0.0, 0.0], [0.0, A_LAT, 0.0], [0.0, 0.0, A_LAT]],
        slab_normal=[0.0, 0.0, 1.0],
        n_Pb=1, n_I=3,
        pb_atoms=[dict(idx=0, cart=[0.0, 0.0, 0.0], type="Pb_interior",
                       n_bonds=6, bonds=bonds)],
    )


@pytest.fixture(scope="module")
def cubic_geom():
    """The cubic cell, with an empty second-neighbour bond list injected.

    `_find_second_neighbour_bonds` consults `_BOND_CACHE` before reading a
    structure file, so seeding the cache keeps these tests independent of
    `data/` and of ASE.  All second-neighbour parameters are zero anyway.
    """
    geom = _cubic_geometry()
    key = (LABEL, 7.5, 6.5)                # the function's default cutoffs
    previous = ham._BOND_CACHE.get(key)
    ham._BOND_CACHE[key] = ([], [])
    yield geom
    if previous is None:
        ham._BOND_CACHE.pop(key, None)
    else:
        ham._BOND_CACHE[key] = previous


# ------------------------------------------------- the two Python builders
@pytest.mark.parametrize("k", KPOINTS)
def test_hamiltonian_is_hermitian(cubic_geom, k):
    """H(k) from the Python builder is Hermitian at every k.

    The existing suite checks this for the Fortran routine only; the Python
    builder adds each hopping and its conjugate transpose by hand, which is
    where the property can be lost.
    """
    H = ham.hamiltonian(LABEL, k, PARAMS, geom=cubic_geom)
    assert np.max(np.abs(H - H.conj().T)) < 1e-12


def test_single_and_batch_builders_agree(cubic_geom):
    """`hamiltonian` and `eigenvalues_batch` describe the same operator.

    These are two hand-written implementations of one Hamiltonian living in
    the same module.  The fit and every figure call the batched one; the
    readable one is what gets inspected when something looks wrong.  Nothing
    in the repository currently requires them to agree.
    """
    one_at_a_time = np.array([
        ham.eigenvalues(LABEL, k, PARAMS, geom=cubic_geom) for k in KPOINTS
    ])
    batched = ham.eigenvalues_batch(LABEL, np.array(KPOINTS), PARAMS,
                                    geom=cubic_geom)

    assert one_at_a_time.shape == batched.shape
    worst = np.max(np.abs(one_at_a_time - batched))
    assert worst < 1e-9, f"largest eigenvalue discrepancy {worst:.3e} eV"


def test_batch_builder_handles_a_single_k(cubic_geom):
    """The batched path gives the same answer for one k as for many."""
    k = KPOINTS[3]
    alone = ham.eigenvalues_batch(LABEL, np.array([k]), PARAMS, geom=cubic_geom)[0]
    within = ham.eigenvalues_batch(LABEL, np.array(KPOINTS), PARAMS,
                                   geom=cubic_geom)[3]
    assert np.max(np.abs(alone - within)) < 1e-10


# ------------------------------------------------- Python versus Fortran
@pytest.mark.parametrize("k", KPOINTS)
def test_python_matches_fortran_in_the_cubic_limit(cubic_geom, k):
    """The general Python builder reduces to the cubic Fortran one.

    The Python code was written to generalise the Fortran model to tilted,
    orthorhombic cells with real direction cosines.  Setting the bond
    directions back to the cube axes and the Harrison exponent to zero should
    recover the original model exactly -- two independent implementations,
    one observable.
    """
    ctypes_mod = pytest.importorskip(
        "ctypesroutines", reason="compiled Fortran library not available"
    )

    H_f = ctypes_mod.hamiltonian_cubic3D_fortran(k[0], k[1], k[2], *PARAMS_FORTRAN)
    ev_f = np.sort(np.linalg.eigvalsh(H_f).real)
    ev_p = ham.eigenvalues(LABEL, k, PARAMS, geom=cubic_geom)

    assert ev_f.shape == ev_p.shape, (
        f"basis sizes differ: Fortran {ev_f.shape}, Python {ev_p.shape}"
    )
    worst = np.max(np.abs(ev_f - ev_p))
    assert worst < 1e-8, (
        f"k = {np.round(k, 4)}: largest eigenvalue difference {worst:.4e} eV\n"
        f"Fortran: {np.round(ev_f, 5)}\nPython : {np.round(ev_p, 5)}"
    )
