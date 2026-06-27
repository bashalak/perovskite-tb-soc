"""
parse_geometry.py — Extract Pb-I bond geometry from VASP structures.

For each structure (3D bulk + 2D n=1–5) reads the VASP file, finds all
Pb-I bonds (with PBC), computes direction cosines (l,m,n) and classifies:

  bond type
  ---------
  apical_surface  : apical I bonded to only 1 Pb  → gets Es1_ap / Ep1_ap
  apical_interior : apical I bridging two Pb layers
  equatorial      : in-plane I

  Pb type
  -------
  Pb_surface  : bonded to a surface apical I  → gets Ep0_surf
  Pb_interior : all 6 neighbours are interior

Output: results/data/geometry_{3D,n1,...,n5}.json

Run from src/python/:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 parse_geometry.py
"""

import os
import json
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

_HERE   = os.path.dirname(os.path.abspath(__file__))
VASP    = os.path.join(_HERE, '..', '..', 'data', 'pos3D2Dtilted')
OUT     = os.path.join(_HERE, '..', '..', 'results', 'data')

CUTOFF  = 4.0   # Å  (Pb-I bonds are 3.1–3.3 Å; 4.0 gives clean separation)

STRUCTURES = [
    ('3D', 'POSCAR'),
    ('n1', 'bama_n1.vasp'),
    ('n2', 'bama_n2.vasp'),
    ('n3', 'bama_n3.vasp'),
    ('n4', 'bama_n4.vasp'),
    ('n5', 'bama_n5.vasp'),
]


def slab_normal(atoms):
    """
    Return unit vector of the slab stacking direction.

    Strategy: derive from CELL GEOMETRY, not PCA on Pb positions.

    PCA on Pb positions is unreliable for Pnma slabs because adjacent layers
    are offset in-plane by Pnma glide-plane symmetry. This inter-layer (x,y)
    shift is correlated with z, causing PCA to report a TILTED slab normal
    even when the slab is genuinely along z.

    Cell-based detection:
      (a) "Diamond cell" (n2..n5 standardised): two cell vectors have nearly
          opposite directions (large angle, e.g. 154°). Slab normal is along
          their DIFFERENCE direction. In-plane = (sum)/2 + third vector.
      (b) "Standard cell" (n1, 3D bulk): cell vectors are roughly orthogonal.
          Slab normal = longest cell vector (the vacuum direction).
    """
    cell = atoms.cell.array
    lengths = np.array([np.linalg.norm(cell[k]) for k in range(3)])

    # ── (a) Detect "diamond" cell: any pair of cell vectors at large angle ─
    for i in range(3):
        for j in range(i + 1, 3):
            cos_ij = (np.dot(cell[i], cell[j])
                      / (lengths[i] * lengths[j]))
            # Threshold: cos < -0.5 → angle > 120°
            if cos_ij < -0.5 and abs(lengths[i] - lengths[j]) < 0.1 * lengths[i]:
                # Diamond cell. Slab normal = (cell[i] - cell[j]) / |..|
                diff = cell[i] - cell[j]
                n = diff / np.linalg.norm(diff)
                return n * np.sign(n[np.argmax(np.abs(n))])

    # ── (b) Standard cell: largest cell vector is the slab normal direction
    longest_idx = int(np.argmax(lengths))
    n = cell[longest_idx] / lengths[longest_idx]
    return n * np.sign(n[np.argmax(np.abs(n))])


def parse_structure(label, fpath):
    atoms = read(fpath)
    pos   = atoms.get_positions()
    cell  = atoms.cell.array

    pb_idx = [a.index for a in atoms if a.symbol == 'Pb']
    i_idx  = [a.index for a in atoms if a.symbol == 'I']

    # ── Slab normal ───────────────────────────────────────────────────────────
    if label == '3D':
        normal = np.array([0., 0., 1.])   # unused for 3D (all I-I bonds use II params)
    else:
        normal = slab_normal(atoms)

    # ── All Pb→I bonds with PBC ───────────────────────────────────────────────
    ia, ja, Darr = neighbor_list('ijD', atoms, CUTOFF)
    mask = np.array([atoms[i].symbol == 'Pb' and atoms[j].symbol == 'I'
                     for i, j in zip(ia, ja)])
    ia, ja, Darr = ia[mask], ja[mask], Darr[mask]

    # ── Determine geometric slab extent along stacking direction ───────────────
    # A bond is apical_SURFACE if the bonded I (Pb + dr, accounting for PBC)
    # lies OUTSIDE the Pb stacking range of the central slab.
    #
    # IMPORTANT: many slab POSCARs have some Pb at "extended" Cartesian
    # coordinates (PBC-wrapped images of central-slab atoms placed at large
    # |proj|).  Using raw min/max would put pb_top/pb_bot at these extended
    # positions and miss every real surface bond.
    #
    # Fix: cluster Pb by projection (gap > 8 Å = vacuum); the LARGEST cluster
    # is the central slab.  Use its extent for pb_top, pb_bot.
    # Also compute the cell length along the slab normal so we can wrap
    # bonded-I projections into the central slab's PBC domain.
    if label != '3D':
        pb_proj = np.array([np.dot(pos[idx], normal) for idx in pb_idx])
        sorted_idx  = np.argsort(pb_proj)
        sorted_proj = pb_proj[sorted_idx]
        clusters = [[sorted_idx[0]]]
        for k in range(1, len(sorted_proj)):
            if sorted_proj[k] - sorted_proj[k-1] > 8.0:
                clusters.append([sorted_idx[k]])
            else:
                clusters[-1].append(sorted_idx[k])
        central = max(clusters, key=len)
        central_proj = pb_proj[central]
        pb_top    = float(central_proj.max())
        pb_bot    = float(central_proj.min())
        pb_center = 0.5 * (pb_top + pb_bot)
        L_stack   = float(max(abs(np.dot(cell[i], normal)) for i in range(3)))
    else:
        pb_top, pb_bot, pb_center, L_stack = 0.0, 0.0, 0.0, 1.0

    # ── Build per-Pb bond table ───────────────────────────────────────────────
    pb_data = []
    for pb in pb_idx:
        sel   = ia == pb
        bonds = []
        for j_i, dr in zip(ja[sel], Darr[sel]):
            dist = float(np.linalg.norm(dr))
            lmn  = (dr / dist).tolist()
            proj = float(np.dot(dr / dist, normal))
            bonds.append({
                'I_idx': int(j_i),
                'dr_ang': [round(x, 6) for x in dr.tolist()],
                'lmn':    [round(x, 8) for x in lmn],
                'dist_ang': round(dist, 6),
                'proj_normal': round(proj, 6),
            })

        # sort by |projection| desc: top 2 = apical, remaining 4 = equatorial
        bonds.sort(key=lambda b: abs(b['proj_normal']), reverse=True)

        for k, b in enumerate(bonds):
            if k < 2:
                # Bonded I position (Pb + dr).  Wrap into the central slab's
                # PBC domain [pb_center - L/2, pb_center + L/2] so an I bonded
                # to a wrapped-image Pb gets compared to the right slab edges.
                i_bonded_pos  = [pos[pb][m] + b['dr_ang'][m] for m in range(3)]
                i_bonded_proj = float(np.dot(i_bonded_pos, normal))
                if label != '3D':
                    while i_bonded_proj > pb_center + 0.5 * L_stack:
                        i_bonded_proj -= L_stack
                    while i_bonded_proj < pb_center - 0.5 * L_stack:
                        i_bonded_proj += L_stack
                is_surf = (label != '3D') and (
                    i_bonded_proj > pb_top + 0.01 or i_bonded_proj < pb_bot - 0.01
                )
                b['bond_type'] = 'apical_surface' if is_surf else 'apical_interior'
            else:
                b['bond_type'] = 'equatorial'

        is_surf_pb = any(b['bond_type'] == 'apical_surface' for b in bonds)
        pb_data.append({
            'idx':    int(pb),
            'cart':   [round(x, 6) for x in pos[pb].tolist()],
            'type':   'Pb_surface' if is_surf_pb else 'Pb_interior',
            'n_bonds': len(bonds),
            'bonds':  bonds,
        })

    return {
        'label':       label,
        'lattice':     [[round(x, 8) for x in row] for row in cell.tolist()],
        'slab_normal': [round(x, 8) for x in normal.tolist()],
        'n_Pb':        len(pb_idx),
        'n_I':         len(i_idx),
        'pb_atoms':    pb_data,
    }


def print_summary(data):
    label = data['label']
    n_sp  = sum(1 for a in data['pb_atoms'] if a['type'] == 'Pb_surface')
    n_si  = sum(1 for a in data['pb_atoms']
                for b in a['bonds'] if b['bond_type'] == 'apical_surface')
    n_ai  = sum(1 for a in data['pb_atoms']
                for b in a['bonds'] if b['bond_type'] == 'apical_interior')
    n_eq  = sum(1 for a in data['pb_atoms']
                for b in a['bonds'] if b['bond_type'] == 'equatorial')
    bad   = [a for a in data['pb_atoms'] if a['n_bonds'] != 6]
    print(f"  {label:3s}: {data['n_Pb']:2d} Pb, {data['n_I']:2d} I | "
          f"surface Pb={n_sp}  apical_surf={n_si}  apical_int={n_ai}  eq={n_eq} | "
          f"slab_normal=[{data['slab_normal'][0]:.3f},{data['slab_normal'][1]:.3f},{data['slab_normal'][2]:.3f}]")
    for a in bad:
        print(f"  WARNING: Pb idx={a['idx']} has {a['n_bonds']} bonds (expected 6)")


def main():
    os.makedirs(OUT, exist_ok=True)
    print('parse_geometry.py\n')
    for label, fname in STRUCTURES:
        fpath = os.path.join(VASP, fname)
        data  = parse_structure(label, fpath)
        print_summary(data)
        out = os.path.join(OUT, f'geometry_{label}.json')
        with open(out, 'w') as f:
            json.dump(data, f, indent=2)
    print(f'\nJSON files saved to {OUT}/')


if __name__ == '__main__':
    main()
