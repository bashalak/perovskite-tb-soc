"""
make_pnma_xyz_from_supercell.py — extract Pnma slab into orthogonal xyz cell.

Approach (cleaner than supercell):
  1. Read original Pnma POSCAR.
  2. Drop organic atoms (keep Pb and I only).
  3. BFS-unwrap from a seed Pb: walk through Pb-I-Pb bonds, accumulating
     PBC-corrected bond vectors. The result is the original slab as a single
     intact connected component in unwrapped Cartesian space — no PBC
     boundary artefacts, no in-plane copies, exact atom count.
  4. Identify the in-plane Bravais lattice from the original cell's lattice
     vectors (a, b, c). The smallest orthogonal cell has:
        a_new = |c|              (was already aligned with x for these slabs)
        b_new = |a + b|           (the recovered in-plane y-period)
        c_new = slab_thickness + 2·VACUUM
  5. Translate the unwrapped slab so it sits centred in the new cell.
  6. Wrap into the new cell using new PBC (in-plane only).

This gives one slab per n, in an exact orthogonal cell, with the same atom
count as the cubic synthetic version.

Output: data/pos3D2Dpnma_xyz/
        pnma_xyz_n1.vasp ... pnma_xyz_n5.vasp
"""

import os
import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.neighborlist import neighbor_list

_HERE = os.path.dirname(os.path.abspath(__file__))
_IN   = os.path.join(_HERE, '..', '..', 'data', 'pos3D2Dtilted')
_OUT  = os.path.join(_HERE, '..', '..', 'data', 'pos3D2Dpnma_xyz')

VACUUM = 13.0
PB_I_CUTOFF = 4.0


def drop_organics(atoms):
    keep = np.array([s in ('Pb', 'I') for s in atoms.symbols])
    return atoms[keep]


def bfs_unwrap(atoms):
    """BFS through Pb-I bonds, placing each atom at parent + PBC-corrected
    bond vector. Returns the same atoms but with positions unwrapped."""
    ia, ja, D = neighbor_list('ijD', atoms, PB_I_CUTOFF)

    # Build adjacency: atom i → list of (j, d_ij)
    adj = {}
    for i, j, d in zip(ia, ja, D):
        adj.setdefault(int(i), []).append((int(j), np.array(d)))

    # Seed: Pb closest to fractional centre (0.5, 0.5, 0.5)
    pb_indices = [i for i, s in enumerate(atoms.symbols) if s == 'Pb']
    frac = atoms.get_scaled_positions()
    seed = pb_indices[int(np.argmin(np.linalg.norm(frac[pb_indices] - 0.5, axis=1)))]

    # BFS
    new_positions = {seed: atoms.positions[seed].copy()}
    queue = [seed]
    visited = {seed}
    while queue:
        i = queue.pop(0)
        if i not in adj:
            continue
        for j, d in adj[i]:
            if j in visited:
                continue
            # d = vector from i to j (already PBC-corrected by neighbor_list)
            new_positions[j] = new_positions[i] + d
            visited.add(j)
            queue.append(j)

    # Atoms not reached (rare; e.g., surface-only I that the cutoff missed)
    for i in range(len(atoms)):
        if i not in new_positions:
            print(f'  WARNING: atom {i} ({atoms.symbols[i]}) not reached by BFS, '
                  f'using original position')
            new_positions[i] = atoms.positions[i].copy()

    out_pos = np.array([new_positions[i] for i in range(len(atoms))])
    return Atoms(symbols=atoms.symbols, positions=out_pos,
                 cell=atoms.cell, pbc=True)


def determine_orthogonal_cell(atoms_unwrapped, original_cell):
    """From the original (possibly oblique) cell, compute the smallest
    orthogonal cell that has the same in-plane periodicity.

    Original cells observed:
      n=1: orthogonal already: a=(ax,0,0), b=(0,by,0), c=(0,0,cz)
      n>1: oblique: a=(0,-y,+z), b=(0,-y,-z), c=(cx,0,0)
           → in-plane periods are |c| along x and |a+b| along y
    """
    a, b, c = original_cell[0], original_cell[1], original_cell[2]

    # Check if already orthogonal
    def is_axis_aligned(v):
        non_zero = np.abs(v) > 1e-6
        return non_zero.sum() == 1

    if is_axis_aligned(a) and is_axis_aligned(b) and is_axis_aligned(c):
        # Identify which axis each vector is on, then collect lengths
        x_len = max(abs(a[0]), abs(b[0]), abs(c[0]))
        y_len = max(abs(a[1]), abs(b[1]), abs(c[1]))
        z_atoms = atoms_unwrapped.positions[:, 2]
        slab_thickness = z_atoms.max() - z_atoms.min()
        return x_len, y_len, slab_thickness

    # Oblique case: in-plane periods from |c| and |a+b|
    apb = a + b
    amb = a - b

    # The vector that's already along x is |c|, the one perpendicular to it
    # in-plane (along y) is (a+b)
    if abs(c[0]) > 0.9 * np.linalg.norm(c):  # c along x
        x_len = np.linalg.norm(c)
    else:
        x_len = np.linalg.norm(apb) if abs(apb[0]) > abs(apb[1]) else np.linalg.norm(amb)

    if abs(apb[1]) > 0.9 * np.linalg.norm(apb):  # a+b along y
        y_len = np.linalg.norm(apb)
    else:
        y_len = np.linalg.norm(amb)

    z_atoms = atoms_unwrapped.positions[:, 2]
    slab_thickness = z_atoms.max() - z_atoms.min()
    return x_len, y_len, slab_thickness


def place_in_orthogonal_cell(atoms_unwrapped, x_len, y_len, c_len):
    """Translate the unwrapped slab so it sits centred in an orthogonal cell
    of dimensions (x_len, y_len, c_len) and wrap in-plane only."""
    pos = atoms_unwrapped.positions.copy()

    # Centre slab in z within the new c
    z_min, z_max = pos[:, 2].min(), pos[:, 2].max()
    z_centre = (z_min + z_max) / 2
    pos[:, 2] += c_len / 2 - z_centre

    # Centre in x and y by aligning the atomic centroid to (x_len/2, y_len/2)
    xy_centroid = pos[:, :2].mean(axis=0)
    pos[:, 0] += x_len / 2 - xy_centroid[0]
    pos[:, 1] += y_len / 2 - xy_centroid[1]

    # Wrap in x and y only (z is non-periodic for a slab)
    pos[:, 0] = pos[:, 0] % x_len
    pos[:, 1] = pos[:, 1] % y_len

    new_cell = np.diag([x_len, y_len, c_len])
    return Atoms(symbols=atoms_unwrapped.symbols, positions=pos,
                 cell=new_cell, pbc=True)


def process(input_path, label):
    print(f'\n=== {label} ===')
    atoms = read(input_path)
    print(f'  original: {len(atoms)} atoms')
    original_cell = atoms.cell.array.copy()

    atoms_pi = drop_organics(atoms)
    n_pb = sum(1 for s in atoms_pi.symbols if s == 'Pb')
    n_i  = sum(1 for s in atoms_pi.symbols if s == 'I')
    print(f'  after dropping organics: {len(atoms_pi)} atoms ({n_pb} Pb + {n_i} I)')

    atoms_unwrapped = bfs_unwrap(atoms_pi)
    print(f'  BFS-unwrapped. z range: '
          f'[{atoms_unwrapped.positions[:, 2].min():.3f}, '
          f'{atoms_unwrapped.positions[:, 2].max():.3f}]')

    x_len, y_len, slab_thickness = determine_orthogonal_cell(
        atoms_unwrapped, original_cell)
    c_len = slab_thickness + 2 * VACUUM
    print(f'  new orthogonal cell: a={x_len:.4f}, b={y_len:.4f}, c={c_len:.4f}')
    print(f'  slab thickness: {slab_thickness:.4f} Å (vacuum {VACUUM} Å each side)')

    final = place_in_orthogonal_cell(atoms_unwrapped, x_len, y_len, c_len)
    return final


def verify(atoms, label):
    ia, ja, D = neighbor_list('ijD', atoms, 4.0)
    pb_i = []
    for i, j, d in zip(ia, ja, D):
        if atoms[i].symbol == 'Pb' and atoms[j].symbol == 'I':
            pb_i.append(np.linalg.norm(d))
    pb_i = np.array(pb_i)
    n_pb = sum(1 for s in atoms.symbols if s == 'Pb')
    n_i  = sum(1 for s in atoms.symbols if s == 'I')
    print(f'  → output: {n_pb} Pb + {n_i} I = {len(atoms)} atoms')
    print(f'    Pb-I bonds: count={len(pb_i)} '
          f'(={len(pb_i)/max(n_pb,1):.1f} per Pb), '
          f'min={pb_i.min():.4f}, max={pb_i.max():.4f}, '
          f'mean={pb_i.mean():.4f}')


def main():
    os.makedirs(_OUT, exist_ok=True)
    print(f'Output: {_OUT}\n')
    for n in range(1, 6):
        in_path = os.path.join(_IN, f'bama_n{n}.vasp')
        if not os.path.exists(in_path):
            continue
        try:
            atoms_new = process(in_path, f'bama_n{n}')
            verify(atoms_new, f'pnma_xyz_n{n}')
            out = os.path.join(_OUT, f'pnma_xyz_n{n}.vasp')
            write(out, atoms_new, format='vasp', vasp5=True, sort=True)
            print(f'    {out}')
        except Exception as e:
            print(f'  ✗ ERROR: {e}')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
