"""
make_pnma_full_poscars.py — Step B: full Pnma reconstruction = Step C + Pb
displacement + cell contraction.

Additional ingredients on top of Step C (Glazer a⁻b⁺a⁻ tilts):

1. Pb antiferroelectric z-displacement
     For Pb at cubic indices (i_a, i_b, i_c), shift z-coordinate by
         dz = (-1)^(i_a + i_b + i_c) · δ_pb_z
     This breaks the apical bond symmetry: each Pb has one short and one long
     apical bond (the cubic-symmetric 3.14 splits to 3.14 ± something).

2. In-plane cell contraction
     Lab x,y multiplied by (cell_scale_xy) to shrink in-plane.
     Positions in xy are scaled by the same factor (preserves fractional
     coordinates).
     Lab z is NOT scaled for 2D slabs (preserves the vacuum gap).
     For 3D bulk, lab z is scaled by cell_scale_z.

3. Pb in-plane displacement (optional)
     For some Pnma settings, Pb also shifts slightly in xy. Add via δ_pb_xy
     along the Pb1↔Pb2 connecting direction.

Parameter conventions to match experimental MAPbI₃ Pnma:
     θ_a ≈ 13°  (around cubic a, in-plane diagonal)
     θ_b ≈ 12°  (around cubic b, other in-plane diagonal)
     θ_c ≈ 0–5° (around cubic c, stacking — your observation: minor)
     δ_pb_z ≈ 0.03–0.10 Å
     cell_scale_xy ≈ 0.99 (~1% contraction)

Run:
    python3 make_pnma_full_poscars.py
"""

import os
import sys
import numpy as np
from ase.io import read, write
from ase import Atoms

_HERE = os.path.dirname(os.path.abspath(__file__))
_IN   = os.path.join(_HERE, '..', '..', 'data', 'pos3D2Dcubic_sqrt2')
_OUT  = os.path.join(_HERE, '..', '..', 'data', 'pos3D2Dcubic_pnma_full')

# Import the Step C tilt machinery
sys.path.insert(0, _HERE)
from make_pnma_tilt_poscars import (
    apply_pnma_tilt, pb_cubic_indices, verify,
    A_NEW, A_CUB, PBI
)

# Step B parameter scenarios: (θ_a, θ_b, θ_c, δ_pb_z, cell_scale_xy, label)
SCENARIOS = [
    # ── benchmarks ────────────────────────────────────────────────────────
    (0.0,  0.0,  0.0,  0.00,  1.000, 'cubic'),
    # ── tilt only (= Step C) ──────────────────────────────────────────────
    (13.0, 12.0, 0.0,  0.00,  1.000, 'tiltOnly'),
    # ── tilt + Pb displacement ────────────────────────────────────────────
    (13.0, 12.0, 0.0,  0.05,  1.000, 'pbShift05'),
    (13.0, 12.0, 0.0,  0.10,  1.000, 'pbShift10'),
    (13.0, 12.0, 0.0,  0.15,  1.000, 'pbShift15'),
    # ── tilt + cell contraction ───────────────────────────────────────────
    (13.0, 12.0, 0.0,  0.00,  0.995, 'cellShrink05'),
    (13.0, 12.0, 0.0,  0.00,  0.990, 'cellShrink10'),
    # ── full Pnma-like ────────────────────────────────────────────────────
    (13.0, 12.0, 5.0,  0.05,  0.995, 'pnmaLight'),
    (13.0, 12.0, 5.0,  0.10,  0.990, 'pnmaMedium'),
    (15.0, 15.0, 5.0,  0.10,  0.990, 'pnmaStrong'),
]


def apply_step_B(atoms_cubic, theta_a, theta_b, theta_c,
                 delta_pb_z, cell_scale_xy, cell_scale_z=1.0):
    """Apply full Pnma reconstruction to a cubic √2×√2 structure.

    Order of operations:
      1. Apply Glazer tilts (Step C) to in-plane Pb-I bonds.
      2. Apply Pb antiferroelectric z-displacement.
      3. Scale the cell (in-plane) and positions accordingly.
    """
    # 1. Apply tilt
    atoms_t = apply_pnma_tilt(atoms_cubic, theta_a, theta_b, theta_c)

    # 2. Pb displacement
    pb_indices = [i for i, s in enumerate(atoms_t.symbols) if s == 'Pb']
    layer_z = sorted(set(round(atoms_t.positions[i, 2], 3) for i in pb_indices))

    new_pos = atoms_t.positions.copy()
    for pb_i in pb_indices:
        i_a, i_b, i_c = pb_cubic_indices(atoms_t.positions[pb_i], layer_z)
        sign = (-1) ** (i_a + i_b + i_c)
        new_pos[pb_i, 2] += sign * delta_pb_z

    # 3. Cell scaling — scale xy by cell_scale_xy, z by cell_scale_z
    new_cell = atoms_t.cell.array.copy()
    new_cell[0] = new_cell[0] * cell_scale_xy
    new_cell[1] = new_cell[1] * cell_scale_xy
    new_cell[2] = new_cell[2] * cell_scale_z

    # Scale all positions to keep fractional coords
    new_pos[:, 0] *= cell_scale_xy
    new_pos[:, 1] *= cell_scale_xy
    new_pos[:, 2] *= cell_scale_z

    return Atoms(symbols=atoms_t.symbols, positions=new_pos,
                 cell=new_cell, pbc=True)


def verify_step_B(atoms):
    """Verify with detailed bond distribution (especially apical short/long)."""
    from ase.neighborlist import neighbor_list

    ia, ja, D = neighbor_list('ijD', atoms, 4.0)

    apical_dists, eq_dists = [], []
    apical_short, apical_long = [], []
    pb_to_apicals = {}

    for i, j, d in zip(ia, ja, D):
        if atoms[i].symbol == 'I' and atoms[j].symbol == 'Pb':
            dist = np.linalg.norm(d)
            z_frac = abs(d[2]) / dist
            if z_frac > 0.7:
                apical_dists.append(dist)
                pb_to_apicals.setdefault(j, []).append((dist, d[2]))
            else:
                eq_dists.append(dist)

    # For each Pb, split its apicals by sign of d[2] (above vs below)
    # (one should be short, the other long if Pb displacement is correct)
    for pb_idx, ap_list in pb_to_apicals.items():
        if len(ap_list) == 2:
            d1, d2 = ap_list
            if d1[1] > 0 and d2[1] < 0:
                apical_short.append(min(d1[0], d2[0]))
                apical_long.append(max(d1[0], d2[0]))
            elif d1[1] < 0 and d2[1] > 0:
                apical_short.append(min(d1[0], d2[0]))
                apical_long.append(max(d1[0], d2[0]))

    def s(arr, label):
        if not arr: return f'{label}: n/a'
        a = np.array(arr)
        return (f'{label}: n={len(a)}, mean={a.mean():.4f}, '
                f'spread={a.max()-a.min():.4f}')

    return {
        'apical_all':   s(apical_dists, 'apical Pb-I (all)'),
        'apical_short': s(apical_short,  'apical SHORT'),
        'apical_long':  s(apical_long,   'apical LONG'),
        'asymmetry':    (f'asymmetry: {np.mean(apical_long)-np.mean(apical_short):.4f} Å'
                         if apical_short and apical_long else 'asymmetry: n/a'),
        'eq':           s(eq_dists,      'equatorial'),
    }


def main():
    os.makedirs(_OUT, exist_ok=True)
    print(f'Output: {_OUT}\n')

    inputs = [(f'cubic_n{n}', f'cubic_n{n}.vasp') for n in range(1, 6)]

    for label, fname in inputs:
        in_path = os.path.join(_IN, fname)
        if not os.path.exists(in_path):
            continue

        atoms_cubic = read(in_path)
        print(f'\n=== {label} ({len(atoms_cubic)} atoms) ===')

        for theta_a, theta_b, theta_c, dpb, cxy, sce_label in SCENARIOS:
            atoms_b = apply_step_B(atoms_cubic, theta_a, theta_b, theta_c,
                                   dpb, cxy)
            stats = verify_step_B(atoms_b)

            print(f'  [{sce_label}] θ=({theta_a},{theta_b},{theta_c}) '
                  f'δ_Pb={dpb:.2f} cell={cxy:.3f}')
            print(f'      {stats["apical_short"]}')
            print(f'      {stats["apical_long"]}')
            print(f'      {stats["asymmetry"]}')
            print(f'      {stats["eq"]}')

            out_fname = f'{label}_{sce_label}.vasp'
            write(os.path.join(_OUT, out_fname),
                  atoms_b, format='vasp', vasp5=True, sort=True)


if __name__ == '__main__':
    main()
