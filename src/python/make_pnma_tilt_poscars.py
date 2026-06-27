"""
make_pnma_tilt_poscars.py — Step C: apply full Glazer a⁻b⁺a⁻ tilt to cubic √2×√2.

Three rotation axes in our lab frame:
    cubic a-axis  →  lab (1, 1, 0)/√2     (in-plane diagonal)
    cubic b-axis  →  lab (-1, 1, 0)/√2    (other in-plane diagonal)
    cubic c-axis  →  lab (0, 0, 1)        (stacking)

Glazer a⁻b⁺a⁻ rotations per Pb at cubic indices (i_a, i_b, i_c):
    R_pb = R_c(sign_c · θ_a) · R_b(+θ_b) · R_a(sign_a · θ_a)

Sign rules (consistent under PBC of √2×√2 + corner-sharing + apical bending):
    sign_a = (-1)^(i_a + i_b + i_c)   — anti-phase in xy AND along c (the c
                                        alternation is what BENDS apical bonds)
    sign_b = +1                       — in-phase (b⁺)
    sign_c = (-1)^(i_a + i_b + i_c)   — same pattern as sign_a (Pnma symmetry)

Independent θ_a, θ_b, θ_c so the user can tune (e.g., θ_c < θ_a to match
the observation that tilt around stacking is smaller in real Pnma).

Cubic indices in our √2×√2 cell:
    Pb1 at lab (0, 0, z_layer)         → cubic (0, 0, k)
    Pb2 at lab (4.44, 4.44, z_layer)   → cubic (1, 0, k)
    where k = layer index (0, 1, 2, ...)

I-positioning rule (corner-sharing reconciliation):
    Each I is bonded to 2 Pb. Predict I position from each Pb's rotation,
    take the midpoint of the two predictions. Surface apical I (bonded to
    only 1 Pb in a slab) use that Pb's prediction directly.

Output: data/pos3D2Dcubic_pnma/
        cubic_n{1..5}_pnma_a{aa}_b{bb}.vasp
        where aa = θ_a in degrees, bb = θ_b in degrees.

Run:
    python3 make_pnma_tilt_poscars.py
"""

import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(os.path.abspath(__file__))
_IN   = os.path.join(_HERE, '..', '..', 'data', 'pos3D2Dcubic_sqrt2')
_OUT  = os.path.join(_HERE, '..', '..', 'data', 'pos3D2Dcubic_pnma')

# Cubic geometry
PBI    = 3.14
A_CUB  = 6.28
A_NEW  = A_CUB * np.sqrt(2)   # 8.88

# Rotation axes in lab frame
AXIS_A = np.array([+1.0, +1.0, 0.0]) / np.sqrt(2)
AXIS_B = np.array([-1.0, +1.0, 0.0]) / np.sqrt(2)
AXIS_C = np.array([+0.0, +0.0, 1.0])

# Tilt-angle triples (θ_a, θ_b, θ_c) in degrees:
#   θ_a = magnitude around cubic a-axis (in-plane diagonal, bends apical)
#   θ_b = magnitude around cubic b-axis (other in-plane diagonal, bends apical)
#   θ_c = magnitude around cubic c-axis (= lab z, stacking; bends in-plane)
# Setting θ_c=0 isolates the dominant Pnma tilts (consistent with the
# observation that tilt around the stacking axis is smaller in real Pnma).
TILT_TRIPLES = [
    (5.0,  5.0,  5.0),
    (10.0, 10.0, 10.0),
    (15.0, 15.0, 15.0),
    (13.0, 12.0, 13.0),   # classical a⁻b⁺a⁻ (θ_c = θ_a)
    (13.0, 12.0,  0.0),   # in-plane axis tilts only (apical bending dominant)
    (15.0, 15.0,  0.0),
]

POS_TOL = 0.5   # Å tolerance for Pb-type classification


def pb_cubic_indices(pos_lab, layer_z_list):
    """Return (i_a, i_b, i_c) for a Pb at lab position pos_lab.

    i_c = layer index (0-based, layers indexed from low z to high z)
    (i_a, i_b) = (0, 0) for Pb at xy=(0, 0) mod A_NEW
                 (1, 0) for Pb at xy=(A_NEW/2, A_NEW/2)
    """
    x_mod = pos_lab[0] % A_NEW
    y_mod = pos_lab[1] % A_NEW
    z     = pos_lab[2]

    # Classify in-plane Pb (corner vs centre)
    is_corner = (abs(x_mod) < POS_TOL or abs(x_mod - A_NEW) < POS_TOL) and \
                (abs(y_mod) < POS_TOL or abs(y_mod - A_NEW) < POS_TOL)
    is_centre = abs(x_mod - A_NEW / 2) < POS_TOL and \
                abs(y_mod - A_NEW / 2) < POS_TOL

    if is_corner:
        i_a, i_b = 0, 0
    elif is_centre:
        i_a, i_b = 1, 0
    else:
        raise ValueError(f'Pb at {pos_lab} not recognised as corner or centre')

    # Layer index
    i_c = int(np.argmin([abs(z - lz) for lz in layer_z_list]))
    return (i_a, i_b, i_c)


def glazer_rotation(i_a, i_b, i_c, theta_a_deg, theta_b_deg, theta_c_deg):
    """Rotation matrix for Pb at (i_a, i_b, i_c) under Glazer a⁻b⁺a⁻."""
    sign_a = (-1) ** (i_a + i_b + i_c)
    sign_b = +1
    sign_c = (-1) ** (i_a + i_b + i_c)

    theta_a = np.radians(theta_a_deg)
    theta_b = np.radians(theta_b_deg)
    theta_c = np.radians(theta_c_deg)

    R_a = Rotation.from_rotvec(sign_a * theta_a * AXIS_A).as_matrix()
    R_b = Rotation.from_rotvec(sign_b * theta_b * AXIS_B).as_matrix()
    R_c = Rotation.from_rotvec(sign_c * theta_c * AXIS_C).as_matrix()
    # Apply a first (right), then b, then c
    return R_c @ R_b @ R_a


def apply_pnma_tilt(atoms, theta_a_deg, theta_b_deg, theta_c_deg):
    """Apply Glazer a⁻b⁺a⁻ tilt. Pb stay fixed; I get midpoint-averaged
    positions from the rotations of the 2 Pb each I bonds to."""
    pb_indices = [i for i, s in enumerate(atoms.symbols) if s == 'Pb']

    # Layer z values (unique Pb z-coordinates)
    layer_z = sorted(set(round(atoms.positions[i, 2], 3) for i in pb_indices))

    # Compute rotation matrix for each Pb
    pb_R = {}
    for pb_i in pb_indices:
        idx = pb_cubic_indices(atoms.positions[pb_i], layer_z)
        pb_R[pb_i] = glazer_rotation(*idx, theta_a_deg, theta_b_deg, theta_c_deg)

    # Build I → bonded Pb map (with PBC-corrected vectors)
    ia, ja, D = neighbor_list('ijD', atoms, 4.0)
    i_to_pbs = {}
    for i, j, d in zip(ia, ja, D):
        if atoms[i].symbol == 'I' and atoms[j].symbol == 'Pb':
            # d = vector from atom i (I) to atom j (Pb), PBC-corrected
            i_to_pbs.setdefault(i, []).append((j, d))

    # For each I, predict new position from each bonded Pb, take midpoint
    new_pos = atoms.positions.copy()
    for i_idx, pb_list in i_to_pbs.items():
        # Take 2 nearest Pb
        pb_list = sorted(pb_list, key=lambda t: np.linalg.norm(t[1]))[:2]

        predictions = []
        for pb_idx, d_ItoPb in pb_list:
            # PBC-corrected position of this Pb (may be image outside cell)
            pb_pos_pbc = atoms.positions[i_idx] + d_ItoPb
            # Original Pb→I bond vector in cubic frame (= -d_ItoPb)
            bond_PbtoI = -d_ItoPb
            # Rotated bond vector
            new_bond = pb_R[pb_idx] @ bond_PbtoI
            predictions.append(pb_pos_pbc + new_bond)

        if len(predictions) == 1:
            # Surface apical I — only 1 Pb bonded
            new_pos[i_idx] = predictions[0]
        else:
            # Average of 2 predictions (corner-sharing reconciliation)
            new_pos[i_idx] = np.mean(predictions, axis=0)

    return Atoms(symbols=atoms.symbols, positions=new_pos,
                 cell=atoms.cell, pbc=atoms.pbc)


def verify(atoms, theta_a_deg, theta_b_deg, theta_c_deg):
    """Compute bond-length stats and Pb-I-Pb angle stats."""
    ia, ja, D = neighbor_list('ijD', atoms, 4.0)

    pb_i_dists_apical = []
    pb_i_dists_eq     = []
    i_to_pbs = {}
    for i, j, d in zip(ia, ja, D):
        if atoms[i].symbol == 'I' and atoms[j].symbol == 'Pb':
            i_to_pbs.setdefault(i, []).append((j, d))
            # Classify by direction
            dist = np.linalg.norm(d)
            z_frac = abs(d[2]) / dist
            if z_frac > 0.7:
                pb_i_dists_apical.append(dist)
            else:
                pb_i_dists_eq.append(dist)

    # Pb-I-Pb angles
    apical_angs, inplane_angs = [], []
    for i_idx, pb_list in i_to_pbs.items():
        if len(pb_list) < 2: continue
        pb_list = sorted(pb_list, key=lambda t: np.linalg.norm(t[1]))[:2]
        v1, v2 = pb_list[0][1], pb_list[1][1]
        cos_a = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                        -1.0, 1.0)
        ang = np.degrees(np.arccos(cos_a))
        pb_pb = v2 - v1
        z_frac = abs(pb_pb[2]) / max(np.linalg.norm(pb_pb), 1e-9)
        if z_frac > 0.7:
            apical_angs.append(ang)
        elif z_frac < 0.3:
            inplane_angs.append(ang)

    def stat(arr, label, expected=None):
        if not arr:
            return f'{label}: n/a'
        arr = np.array(arr)
        s = (f'{label}: n={len(arr)}, mean={arr.mean():.4f}, '
             f'min={arr.min():.4f}, max={arr.max():.4f}')
        if expected is not None:
            s += f', expect~{expected:.4f}'
        return s

    return {
        'apical_bond': stat(pb_i_dists_apical, 'apical Pb-I'),
        'eq_bond':     stat(pb_i_dists_eq,     'eq Pb-I'),
        'apical_ang':  stat(apical_angs,        'apical ∠'),
        'inplane_ang': stat(inplane_angs,       'in-plane ∠'),
    }


def main():
    os.makedirs(_OUT, exist_ok=True)
    print(f'Output: {_OUT}\n')

    inputs = [(f'cubic_n{n}', f'cubic_n{n}.vasp') for n in range(1, 6)]

    for label, fname in inputs:
        in_path = os.path.join(_IN, fname)
        if not os.path.exists(in_path):
            print(f'SKIP (not found): {in_path}'); continue

        atoms_cubic = read(in_path)
        print(f'\n=== {label} ({len(atoms_cubic)} atoms) ===')

        # Sanity: θ=0 should reproduce cubic exactly
        atoms_0 = apply_pnma_tilt(atoms_cubic, 0.0, 0.0, 0.0)
        max_dev = np.max(np.abs(atoms_0.positions - atoms_cubic.positions))
        print(f'  θ=0 sanity: max deviation = {max_dev:.3e} Å')

        for theta_a, theta_b, theta_c in TILT_TRIPLES:
            atoms_tilt = apply_pnma_tilt(atoms_cubic, theta_a, theta_b, theta_c)
            stats = verify(atoms_tilt, theta_a, theta_b, theta_c)
            print(f'  θ_a={theta_a:>4.1f}° θ_b={theta_b:>4.1f}° θ_c={theta_c:>4.1f}°:')
            for k, v in stats.items():
                print(f'      {v}')

            out_fname = (f'{label}_pnma_a{int(theta_a):02d}'
                         f'_b{int(theta_b):02d}_c{int(theta_c):02d}.vasp')
            write(os.path.join(_OUT, out_fname),
                  atoms_tilt, format='vasp', vasp5=True, sort=True)


if __name__ == '__main__':
    main()
