"""
ham_ortho.py — TB+SOC Hamiltonian for orthorhombic/tilted perovskite slabs.

Builds H(k) from the geometry_{label}.json bond tables produced by
parse_geometry.py, using general Slater-Koster direction cosines.

Compared to the cubic code (ham2d_general.f90), only the hopping blocks
change: bonds now have real (l,m,n) instead of hardcoded ±x/±y/±z.
The SOC template, spin-doubling, and parametrization are identical.

Public API
----------
hamiltonian(label, kvec, params, geom=None)   → complex Hermitian matrix
eigenvalues(label, kvec, params, geom=None)   → sorted real eigenvalues
load_geometry(label)                          → geometry dict (cached)

Run standalone for a quick band-gap test:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 ham_ortho.py
"""

import os
import json
import numpy as np
from functools import lru_cache

_HERE = os.path.dirname(os.path.abspath(__file__))
_DAT  = os.path.join(_HERE, '..', '..', 'results', 'data')


# ── geometry loading ──────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def load_geometry(label):
    """Load and cache geometry_{label}.json."""
    path = os.path.join(_DAT, f'geometry_{label}.json')
    with open(path) as f:
        return json.load(f)


# ── in-plane primitive lattice helper ────────────────────────────────────────

def in_plane_primitive_vectors(real_cell, slab_normal):
    """
    Given a 3×3 real-space cell and the slab normal unit vector,
    return two real-space lattice vectors that span the slab plane.

    Tries each pair (a_i, a_i ± a_j) for i ≠ j and keeps the pair with the
    smallest |projection on slab_normal| — i.e. the most truly in-plane.
    Works whether the cell vectors are already in-plane (n1: cell[0], cell[1]
    perpendicular to the normal) or have to be combined (n2-n5: cell[0]+cell[1]
    is in-plane while cell[0] and cell[1] individually are not).
    """
    cell = np.asarray(real_cell, dtype=float)
    n    = np.asarray(slab_normal, dtype=float); n = n / np.linalg.norm(n)

    # Build candidate lattice vectors: each cell vector + small integer combinations
    candidates = []
    for i in range(3):
        candidates.append(cell[i])
        for j in range(3):
            if j == i: continue
            candidates.append(cell[i] + cell[j])
            candidates.append(cell[i] - cell[j])

    def in_plane_score(v):
        L = np.linalg.norm(v)
        if L < 1e-9: return np.inf
        return abs(np.dot(v, n)) / L          # 0 = perfectly in-plane

    # Sort by in-plane-ness and pick the two shortest linearly-independent ones
    candidates.sort(key=lambda v: (in_plane_score(v), np.linalg.norm(v)))
    a1 = candidates[0]
    for v in candidates[1:]:
        # require linear independence with a1 (parallel check)
        cross = np.cross(a1, v)
        if np.linalg.norm(cross) > 1e-6:
            a2 = v
            break
    else:
        raise RuntimeError("could not find two independent in-plane vectors")
    return np.array(a1), np.array(a2)


def in_plane_reciprocal_vectors(real_cell, slab_normal):
    """
    Build the 2D reciprocal lattice basis (b1_2d, b2_2d) in Cartesian units [Å⁻¹]
    from the true in-plane real-space primitive vectors.
    """
    a1, a2 = in_plane_primitive_vectors(real_cell, slab_normal)
    n      = np.asarray(slab_normal, dtype=float); n /= np.linalg.norm(n)
    # 2D reciprocal: b1 ⟂ a2 within the slab plane, b1·a1 = 2π (and symmetric).
    # Use the slab normal as the third axis: ti = 2π (aj × n) / [ai · (aj × n)].
    b1 = 2*np.pi * np.cross(a2, n) / np.dot(a1, np.cross(a2, n))
    b2 = 2*np.pi * np.cross(n, a1) / np.dot(a2, np.cross(n, a1))
    return b1, b2


# ── Slater-Koster blocks ──────────────────────────────────────────────────────

def sk_block(l, m, n, Vss, Vsp, Vps, Vppsig, Vpppi):
    """
    4×4 Slater-Koster hopping matrix for a bond with direction cosines (l,m,n).
    Basis order: s=0, px=1, py=2, pz=3.
    Convention: T[row, col] = <row orbital on dest | H | col orbital on src>.
    """
    T = np.zeros((4, 4))
    T[0, 0] = Vss
    T[0, 1] = l * Vsp;  T[0, 2] = m * Vsp;  T[0, 3] = n * Vsp
    T[1, 0] = l * Vps;  T[2, 0] = m * Vps;  T[3, 0] = n * Vps
    T[1, 1] = l*l * Vppsig + (1 - l*l) * Vpppi
    T[2, 2] = m*m * Vppsig + (1 - m*m) * Vpppi
    T[3, 3] = n*n * Vppsig + (1 - n*n) * Vpppi
    T[1, 2] = T[2, 1] = l * m * (Vppsig - Vpppi)
    T[1, 3] = T[3, 1] = l * n * (Vppsig - Vpppi)
    T[2, 3] = T[3, 2] = m * n * (Vppsig - Vpppi)
    return T


def sk_block_II(l, m, n, Vss_II, Vsp_II, Vppsig_II, Vpppi_II):
    """SK block for I-I bonds (same formula, different parameters)."""
    return sk_block(l, m, n, Vss_II, Vsp_II, Vsp_II, Vppsig_II, Vpppi_II)


def sk_block_PbPb(l, m, n, Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb):
    """SK block for Pb-Pb bonds."""
    return sk_block(l, m, n, Vss_PbPb, Vsp_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb)


# ── SOC template (geometry-independent, identical to Fortran HintSOC) ────────

def _build_soc_template():
    """
    8×8 atomic SOC matrix in the interleaved spin basis:
    {s↑,s↓, px↑,px↓, py↑,py↓, pz↑,pz↓} = indices 0..7 (0-based).
    Encoding: L·S in the |s,px,py,pz⟩ ⊗ |↑,↓⟩ basis.
    Applied as: H_SOC += (lamd/2) * HintSOC  per atom.
    """
    ii = 1j
    H = np.zeros((8, 8), dtype=complex)
    # Fortran (1-indexed) → Python (0-indexed): subtract 1
    H[2, 4] = -ii   # px↑ ← py↑
    H[2, 7] =  1.0  # px↑ ← pz↓
    H[3, 5] =  ii   # px↓ ← py↓
    H[3, 6] = -1.0  # px↓ ← pz↑
    H[4, 2] =  ii   # py↑ ← px↑
    H[4, 7] = -ii   # py↑ ← pz↓
    H[5, 3] = -ii   # py↓ ← px↓
    H[5, 6] = -ii   # py↓ ← pz↑
    H[6, 3] = -1.0  # pz↑ ← px↓
    H[6, 5] =  ii   # pz↑ ← py↓
    H[7, 2] =  1.0  # pz↓ ← px↑
    H[7, 4] =  ii   # pz↓ ← py↑
    return H

_SOC_TEMPLATE = _build_soc_template()

_BOND_CACHE = {}   # (label, cutoff_PbPb, cutoff_II) → (pb_pb_bonds, ii_bonds)


# ── second-neighbour bond finder (Pb-Pb and I-I) ─────────────────────────────

def _find_second_neighbour_bonds(geom, cutoff_PbPb=7.5, cutoff_II=6.5):
    """
    Find Pb-Pb and I-I bonds with PBC using ASE neighbor_list.
    Result is cached by label so subsequent calls (different k-points) are free.
    """
    key = (geom['label'], cutoff_PbPb, cutoff_II)
    if key in _BOND_CACHE:
        return _BOND_CACHE[key]

    from ase.io import read
    from ase.neighborlist import neighbor_list

    # 3D bulk: original orthogonal Pnma POSCAR in pos3D2Dtilted/.
    # 2D slabs: orthogonal-cell Pnma POSCARs in pos3D2Dpnma_xyz/ (BFS-unwrapped,
    # organics dropped).
    if geom['label'] == '3D':
        vasp_path = os.path.join(_HERE, '..', '..', 'data', 'pos3D2Dtilted', 'POSCAR')
    else:
        vasp_path = os.path.join(_HERE, '..', '..', 'data', 'pos3D2Dpnma_xyz',
                                 f'pnma_xyz_{geom["label"]}.vasp')
    atoms = read(vasp_path)

    pb_set = {a.index for a in atoms if a.symbol == 'Pb'}
    i_set  = {a.index for a in atoms if a.symbol == 'I'}

    ia, ja, Darr = neighbor_list('ijD', atoms, max(cutoff_PbPb, cutoff_II))

    # ASE neighbor_list returns each pair in both directions ((i,j,D) and (j,i,-D));
    # we keep only i < j so each bond is iterated once.  The Hamiltonian builder
    # adds the +h.c. term explicitly, so double-iteration would zero out the s-p
    # cross terms (Vsp=Vps in sk_block_II / sk_block_PbPb) and double everything else.
    pb_pb_bonds = [(int(i), int(j), D)
                   for i, j, D in zip(ia, ja, Darr)
                   if i in pb_set and j in pb_set and i < j
                   and np.linalg.norm(D) < cutoff_PbPb]

    ii_bonds    = [(int(i), int(j), D)
                   for i, j, D in zip(ia, ja, Darr)
                   if i in i_set and j in i_set and i < j
                   and np.linalg.norm(D) < cutoff_II]

    _BOND_CACHE[key] = (pb_pb_bonds, ii_bonds)
    return pb_pb_bonds, ii_bonds


# ── main Hamiltonian builder ──────────────────────────────────────────────────

def hamiltonian(label, kvec, params, geom=None):
    """
    Build the full TB+SOC Hamiltonian H(k) for structure `label`.

    Parameters
    ----------
    label  : '3D', 'n1', ..., 'n5'
    kvec   : array-like, shape (3,), k-vector in Cartesian units [1/Å]
    params : dict with keys matching the 29-parameter set (see PARAM_KEYS)
    geom   : pre-loaded geometry dict (optional, avoids re-reading JSON)

    Returns
    -------
    H : complex Hermitian ndarray, shape (8*n_atoms, 8*n_atoms)
    """
    if geom is None:
        geom = load_geometry(label)

    kv = np.asarray(kvec, dtype=float)
    p  = params

    # ── atom index maps ───────────────────────────────────────────────────────
    pb_atoms = geom['pb_atoms']
    n_Pb = len(pb_atoms)

    # collect unique I atoms in order of first appearance
    i_map = {}   # I_idx (VASP index) → local index 0..n_I-1
    i_cart = []
    for pb in pb_atoms:
        for b in pb['bonds']:
            if b['I_idx'] not in i_map:
                i_map[b['I_idx']] = len(i_map)
                cart_I = [pb['cart'][k] + b['dr_ang'][k] for k in range(3)]
                i_cart.append(cart_I)
    n_I = len(i_map)
    n_atoms = n_Pb + n_I

    # ── spinless H0 (4*n_atoms × 4*n_atoms) ──────────────────────────────────
    dim0 = 4 * n_atoms
    H0   = np.zeros((dim0, dim0), dtype=complex)

    def blk(idx):
        """Row/col slice for atom idx in spinless basis."""
        return slice(4 * idx, 4 * idx + 4)

    # ── onsite: Pb ────────────────────────────────────────────────────────────
    for a, pb in enumerate(pb_atoms):
        Es0 = p['Es0']
        Ep0 = p['Ep0_surf'] if pb['type'] == 'Pb_surface' and label != '3D' else p['Ep0']
        H0[blk(a), blk(a)] += np.diag([Es0, Ep0, Ep0, Ep0])

    # ── onsite: I ─────────────────────────────────────────────────────────────
    # Determine I type from bond_type where this I first appears
    i_type = {}   # local I index → 'apical_surface' | 'apical_interior' | 'equatorial'
    for pb in pb_atoms:
        for b in pb['bonds']:
            li = i_map[b['I_idx']]
            if li not in i_type:
                i_type[li] = b['bond_type']

    for li in range(n_I):
        bt   = i_type[li]
        if bt == 'apical_surface' and label != '3D':
            Es = p['Es1_ap']
            Ep = p['Ep1_ap']
        else:
            Es = p['Es1']
            Ep = p['Ep1']
        H0[blk(n_Pb + li), blk(n_Pb + li)] += np.diag([Es, Ep, Ep, Ep])

    # ── Pb-I hoppings (HARRISON-SCALED) ───────────────────────────────────────
    # V_eff(d) = V_0 * (D_REF_PBI / d) ** n_harrison
    # For n_harrison = 0 → constant SK (recovers original ham_ortho.py).
    n_har = p.get('n_harrison', 0.0)
    for a, pb in enumerate(pb_atoms):
        for b in pb['bonds']:
            li  = i_map[b['I_idx']]
            b_idx = n_Pb + li
            D   = np.array(b['dr_ang'])        # displacement Pb → I [Å]
            d   = np.linalg.norm(D)
            lmn = b['lmn']
            scale = (D_REF_PBI / d) ** n_har
            T   = sk_block(*lmn,
                           p['Vss']    * scale,
                           p['Vsp']    * scale,
                           p['Vps']    * scale,
                           p['Vppsig'] * scale,
                           p['Vpppi']  * scale)
            phase = np.exp(1j * np.dot(kv, D))
            H0[blk(a),     blk(b_idx)] += phase       * T
            H0[blk(b_idx), blk(a)]     += phase.conj() * T.conj().T

    # ── Pb-Pb second-neighbour hoppings ───────────────────────────────────────
    pb_cart_arr = np.array([pb['cart'] for pb in pb_atoms])
    pb_vasp_idx = {pb['idx']: a for a, pb in enumerate(pb_atoms)}

    pb_pb_bonds, ii_bonds = _find_second_neighbour_bonds(geom)

    for i_vasp, j_vasp, D in pb_pb_bonds:
        if i_vasp not in pb_vasp_idx or j_vasp not in pb_vasp_idx:
            continue
        a, b_idx_pb = pb_vasp_idx[i_vasp], pb_vasp_idx[j_vasp]
        dist = np.linalg.norm(D)
        lmn  = (D / dist).tolist()
        T    = sk_block_PbPb(*lmn,
                              p['Vss_PbPb'], p['Vsp_PbPb'],
                              p['Vppsig_PbPb'], p['Vpppi_PbPb'])
        phase = np.exp(1j * np.dot(kv, D))
        H0[blk(a),        blk(b_idx_pb)] += phase       * T
        H0[blk(b_idx_pb), blk(a)]        += phase.conj() * T.conj().T

    # ── I-I second-neighbour hoppings ─────────────────────────────────────────
    i_vasp_to_local = i_map   # I_idx (VASP) → local index

    for i_vasp, j_vasp, D in ii_bonds:
        if i_vasp not in i_vasp_to_local or j_vasp not in i_vasp_to_local:
            continue
        a_ii = n_Pb + i_vasp_to_local[i_vasp]
        b_ii = n_Pb + i_vasp_to_local[j_vasp]
        dist = np.linalg.norm(D)
        lmn  = (D / dist).tolist()
        # In 3D bulk all I-I bonds use II params (no stacking direction exists).
        # In 2D slabs, split by projection onto slab_normal: bonds along the
        # stacking direction (proj > 0.5) get IIz params, in-plane bonds get II.
        proj = abs(np.dot(lmn, geom['slab_normal']))
        if label != '3D' and proj > 0.5:
            T = sk_block(*lmn,
                         p['Vss_IIz'], p['Vsp_IIz'],
                         p['Vsp_IIz'], p['Vppsig_IIz'], p['Vpppi_IIz'])
        else:
            T = sk_block_II(*lmn,
                            p['Vss_II'], p['Vsp_II'],
                            p['Vppsig_II'], p['Vpppi_II'])
        phase = np.exp(1j * np.dot(kv, D))
        H0[blk(a_ii), blk(b_ii)] += phase       * T
        H0[blk(b_ii), blk(a_ii)] += phase.conj() * T.conj().T

    # ── spin doubling: H_full = H0 ⊗ I_2 (interleaved) ──────────────────────
    dim = 2 * dim0
    H   = np.zeros((dim, dim), dtype=complex)
    idx = np.arange(dim0)
    H[np.ix_(2*idx,   2*idx)]   = H0   # spin-up block
    H[np.ix_(2*idx+1, 2*idx+1)] = H0   # spin-down block

    # ── atomic SOC ────────────────────────────────────────────────────────────
    lamd_Pb = 2.0 * p['deltaSO0'] / 3.0
    lamd_I  = 2.0 * p['deltaSO1'] / 3.0

    for a in range(n_Pb):
        r0 = 8 * a
        H[r0:r0+8, r0:r0+8] += (lamd_Pb / 2.0) * _SOC_TEMPLATE

    for li in range(n_I):
        r0 = 8 * (n_Pb + li)
        H[r0:r0+8, r0:r0+8] += (lamd_I  / 2.0) * _SOC_TEMPLATE

    return H


def eigenvalues(label, kvec, params, geom=None):
    """Return sorted real eigenvalues of H(k)."""
    H = hamiltonian(label, kvec, params, geom=geom)
    return np.sort(np.linalg.eigvalsh(H).real)


def eigenvalues_batch(label, kvecs, params, geom=None):
    """
    Return eigenvalues for all k-vectors at once.
    Vectorizes the phase exp(i k·D) over k, avoiding a Python loop.
    Returns array of shape (nk, n_states), rows sorted ascending.
    """
    if geom is None:
        geom = load_geometry(label)

    kv = np.asarray(kvecs, dtype=float)   # (nk, 3)
    nk = len(kv)
    p  = params

    pb_atoms = geom['pb_atoms']
    n_Pb = len(pb_atoms)

    i_map = {}
    for pb in pb_atoms:
        for b in pb['bonds']:
            if b['I_idx'] not in i_map:
                i_map[b['I_idx']] = len(i_map)
    n_I    = len(i_map)
    n_atoms = n_Pb + n_I
    dim0   = 4 * n_atoms

    def blk(idx): return slice(4*idx, 4*idx+4)

    # ── k-independent onsite base ─────────────────────────────────────────────
    H0_base = np.zeros((dim0, dim0), dtype=complex)

    for a, pb in enumerate(pb_atoms):
        Es0 = p['Es0']
        Ep0 = p['Ep0_surf'] if pb['type'] == 'Pb_surface' and label != '3D' else p['Ep0']
        H0_base[blk(a), blk(a)] += np.diag([Es0, Ep0, Ep0, Ep0])

    i_type = {}
    for pb in pb_atoms:
        for b in pb['bonds']:
            li = i_map[b['I_idx']]
            if li not in i_type:
                i_type[li] = b['bond_type']
    for li in range(n_I):
        bt = i_type[li]
        if bt == 'apical_surface' and label != '3D':
            Es, Ep = p['Es1_ap'], p['Ep1_ap']
        else:
            Es, Ep = p['Es1'], p['Ep1']
        H0_base[blk(n_Pb+li), blk(n_Pb+li)] += np.diag([Es, Ep, Ep, Ep])

    # ── collect hopping terms: (r0, c0, T_4x4, D_3) ─────────────────────────
    hops_r, hops_c, hops_T, hops_D = [], [], [], []

    # Harrison-scaled Pb-I hoppings (NEW): V_eff(d) = V_0 * (D_REF_PBI / d) ** n_harrison
    n_har = p.get('n_harrison', 0.0)
    for a, pb in enumerate(pb_atoms):
        for b in pb['bonds']:
            li    = i_map[b['I_idx']]
            b_idx = n_Pb + li
            D_vec = np.asarray(b['dr_ang'], dtype=float)
            d     = float(np.linalg.norm(D_vec))
            scale = (D_REF_PBI / d) ** n_har
            T     = sk_block(*b['lmn'],
                             p['Vss']    * scale,
                             p['Vsp']    * scale,
                             p['Vps']    * scale,
                             p['Vppsig'] * scale,
                             p['Vpppi']  * scale)
            hops_r.append(4*a);      hops_c.append(4*b_idx)
            hops_T.append(T);        hops_D.append(b['dr_ang'])

    pb_vasp_idx = {pb['idx']: a for a, pb in enumerate(pb_atoms)}
    pb_pb_bonds, ii_bonds = _find_second_neighbour_bonds(geom)

    for i_vasp, j_vasp, D in pb_pb_bonds:
        if i_vasp not in pb_vasp_idx or j_vasp not in pb_vasp_idx:
            continue
        a, bi = pb_vasp_idx[i_vasp], pb_vasp_idx[j_vasp]
        dist  = np.linalg.norm(D); lmn = (D/dist).tolist()
        T = sk_block_PbPb(*lmn,
                          p['Vss_PbPb'], p['Vsp_PbPb'],
                          p['Vppsig_PbPb'], p['Vpppi_PbPb'])
        hops_r.append(4*a);  hops_c.append(4*bi)
        hops_T.append(T);    hops_D.append(D.tolist())

    for i_vasp, j_vasp, D in ii_bonds:
        if i_vasp not in i_map or j_vasp not in i_map:
            continue
        ai = n_Pb + i_map[i_vasp]; bi = n_Pb + i_map[j_vasp]
        dist = np.linalg.norm(D);  lmn = (D/dist).tolist()
        # In 3D bulk all I-I bonds use II params (no stacking direction exists).
        # In 2D slabs, split by projection onto slab_normal: bonds along the
        # stacking direction (proj > 0.5) get IIz params, in-plane bonds get II.
        proj = abs(np.dot(lmn, geom['slab_normal']))
        if label != '3D' and proj > 0.5:
            T = sk_block(*lmn, p['Vss_IIz'], p['Vsp_IIz'], p['Vsp_IIz'],
                         p['Vppsig_IIz'], p['Vpppi_IIz'])
        else:
            T = sk_block_II(*lmn, p['Vss_II'], p['Vsp_II'],
                            p['Vppsig_II'], p['Vpppi_II'])
        hops_r.append(4*ai); hops_c.append(4*bi)
        hops_T.append(T);    hops_D.append(D.tolist())

    # ── vectorized phase & accumulation ───────────────────────────────────────
    T_arr = np.array(hops_T)                       # (n_hops, 4, 4)
    D_arr = np.array(hops_D, dtype=float)          # (n_hops, 3)
    phases = np.exp(1j * (kv @ D_arr.T))           # (nk, n_hops)

    # weighted hoppings: (nk, n_hops, 4, 4)
    contrib = phases[:, :, None, None] * T_arr[None, :, :, :]

    H0_all = np.tile(H0_base, (nk, 1, 1))         # (nk, dim0, dim0)
    for ih, (r0, c0) in enumerate(zip(hops_r, hops_c)):
        H0_all[:, r0:r0+4, c0:c0+4] += contrib[:, ih, :, :]
        H0_all[:, c0:c0+4, r0:r0+4] += contrib[:, ih, :, :].conj().swapaxes(-2, -1)

    # ── spin doubling (interleaved) ───────────────────────────────────────────
    dim  = 2 * dim0
    H_all = np.zeros((nk, dim, dim), dtype=complex)
    H_all[:, ::2,  ::2]  = H0_all
    H_all[:, 1::2, 1::2] = H0_all

    # ── SOC ───────────────────────────────────────────────────────────────────
    lamd_Pb = p['deltaSO0'] / 3.0
    lamd_I  = p['deltaSO1'] / 3.0
    SOC_Pb  = lamd_Pb * _SOC_TEMPLATE
    SOC_I   = lamd_I  * _SOC_TEMPLATE
    for a in range(n_Pb):
        r0 = 8*a
        H_all[:, r0:r0+8, r0:r0+8] += SOC_Pb
    for li in range(n_I):
        r0 = 8*(n_Pb+li)
        H_all[:, r0:r0+8, r0:r0+8] += SOC_I

    return np.linalg.eigvalsh(H_all)               # (nk, dim), sorted real


# ── parameter keys ────────────────────────────────────────────────────────────

PARAM_KEYS = [
    'Es0', 'Ep0', 'Es1', 'Ep1',
    'Vss', 'Vsp', 'Vps', 'Vppsig', 'Vpppi',
    'deltaSO0', 'deltaSO1',
    'Vss_II', 'Vsp_II', 'Vppsig_II', 'Vpppi_II',
    'Vss_IIz', 'Vsp_IIz', 'Vppsig_IIz', 'Vpppi_IIz',
    'Vss_PbPb', 'Vsp_PbPb', 'Vppsig_PbPb', 'Vpppi_PbPb',
    'Es1_ap', 'Ep1_ap', 'Ep0_surf',
    # Harrison-scaling exponent (NEW): V_eff(d) = V * (D_REF_PBI / d)^n_harrison
    # applied to ALL Pb-I SK integrals; d_0 = 3.14 Å.
    # Harrison's original prediction is n=2 for sp-bonded materials.
    'n_harrison',
]

# Reference Pb-I distance for Harrison scaling (= cubic Pb-I)
D_REF_PBI = 3.14


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Best-fit 26 cubic parameters (from newparam5/absorption_2D_fortran.py)
#    _BEST_FIT = [
#        -2.75, +3.42, -9.42, +2.58,                         # Es0,Ep0,Es1,Ep1
#        +1.95, +0.74, -3.64, -1.86, -0.90,                  # Vss,Vsp,Vps,Vppsig,Vpppi
#        +1.30, +0.90,                                        # deltaSO0, deltaSO1
#        -0.19, -0.59, +0.09, +0.27,                         # Vss_II..Vpppi_II
#        -0.49, +0.01, +0.27, +0.20,                         # Vss_IIz..Vpppi_IIz
#         0.0,   0.0,   0.0,   0.0,                          # Vss_PbPb..Vpppi_PbPb
#        -0.16, +3.60, +3.52,                                 # Es1_ap, Ep1_ap, Ep0_surf
#    ]

    _BEST_FIT = [-6.61,+3.47,-4.89,+5.46,
                 -3.45,+2.31,-3.75,+2.06,+3.43,
                 +1.30,+0.90,
                 -0.46,-0.80,+0.66,-0.08,
                 +0.10,+0.01,+0.30,-0.12,
                 -0.64,+0.42,+0.66,-0.84,
                 +11.26,+3.99,+2.78
    ]
    params = dict(zip(PARAM_KEYS, _BEST_FIT))

    print('ham_ortho.py — quick eigenvalue test at k=Γ\n')
    print(f"{'label':>5}  {'n_states':>8}  {'n_Pb':>5}  {'n_I':>5}  "
          f"{'E_VBM(eV)':>10}  {'E_CBM(eV)':>10}  {'gap(eV)':>8}")
    print('─' * 65)

    for label in ['3D', 'n1', 'n2', 'n3', 'n4', 'n5']:
        geom = load_geometry(label)
        n_Pb = geom['n_Pb']
        n_I  = geom['n_I']
        n_st = 8 * (n_Pb + n_I)
        evals = eigenvalues(label, [0., 0., 0.], params, geom=geom)
        n_fill = n_Pb * 2 + n_I * 8   # Pb 6s²=2e, I 5s²5p⁶=8e
        VBM = evals[n_fill - 1]
        CBM = evals[n_fill]
        gap = CBM - VBM
        print(f"{label:>5}  {n_st:>8}  {n_Pb:>5}  {n_I:>5}  "
              f"{VBM:>10.3f}  {CBM:>10.3f}  {gap:>8.3f}")

#F = [0.20687, 1.312988, 0.0, 0.0]   ['energy_err', 'mass_err', 'order_err', 'disp_err']
#eg  [Γ,Y,Z]     = [np.float64(1.6705), np.float64(2.5283), np.float64(3.6068)]
#    target      = [1.5725, 2.5457, 3.4796]
#dVBM [Y,Z]      = [np.float64(-0.2916), np.float64(-1.0395)]
#    target      = [-0.3389, -0.7998]
#dCBM [Y,Z]      = [np.float64(0.5663), np.float64(0.8969)]
#    target      = [0.6343, 1.1073]
#mv   [xx,yy,zz] = [0.1763, 0.2387, 0.1956]   target=[0.209, 0.234, 0.207]
#mc   [xx,yy,zz] = [0.2057, 0.1217, 0.1926]   target=[0.236, 0.199, 0.245]
#eg2D [n1..n5]   = [2.8684, 2.5417, 2.2817, 2.1725, 2.125]
#     target     = [3.016, 2.432, 2.206, 2.078, 1.971]
#mu2D [n1..n5]   = [0.2022, 0.2226, 0.186, 0.1611, 0.165]
#     target     = [0.221, 0.217, 0.201, 0.196, 0.186]
#Es0            =  -6.6100
#Ep0            =  +3.4700
#Es1            =  -4.8900
#Ep1            =  +5.4600
#Vss            =  -3.4500
#Vsp            =  +2.3100
#Vps            =  -3.7500
#Vppsig         =  +2.0600
#Vpppi          =  +3.4300
#deltaSO0       =  +1.3000
#deltaSO1       =  +0.9000
#Vss_II         =  -0.4600
#Vsp_II         =  -0.8000
#Vppsig_II      =  +0.6600
#Vpppi_II       =  -0.0800
#Vss_IIz        =  +0.1000
#Vsp_IIz        =  +0.0100
#Vppsig_IIz     =  +0.3000
#Vpppi_IIz      =  -0.1200
#Vss_PbPb       =  -0.6400
#Vsp_PbPb       =  +0.4200
#Vppsig_PbPb    =  +0.6600
#Vpppi_PbPb     =  -0.8400
#Es1_ap         = -11.2600
#Ep1_ap         =  +3.9900
#Ep0_surf       =  +2.7800