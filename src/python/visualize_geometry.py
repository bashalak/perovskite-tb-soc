"""
visualize_geometry.py — Visualise Pb-I bond classification from parse_geometry.

Two outputs per structure:
  1. PNG  : 3-D matplotlib scatter + bond lines, color-coded by type
  2. VASP : POSCAR with Pb/I only, species renamed so VESTA shows colours:
              Pb_interior       → Pb
              Pb_surface        → Bi   (gold in VESTA)
              I_equatorial      → I
              I_apical_interior → Br   (brown in VESTA)
              I_apical_surface  → Cl   (green in VESTA)

Run:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 visualize_geometry.py [label]   # e.g. n1, n2, 3D  (default: all)
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from ase.io import read

_HERE   = os.path.dirname(os.path.abspath(__file__))
DAT     = os.path.join(_HERE, '..', '..', 'results', 'data')
FIG     = os.path.join(_HERE, '..', '..', 'results', 'figures')
VASP_IN = os.path.join(_HERE, '..', '..', 'data', 'pos3D2Dtilted')

VASP_FILES = {
    '3D': 'POSCAR',
    'n1': 'bama_n1.vasp',
    'n2': 'bama_n2.vasp',
    'n3': 'bama_n3.vasp',
    'n4': 'bama_n4.vasp',
    'n5': 'bama_n5.vasp',
}

# ── colour / style maps ───────────────────────────────────────────────────────
PB_STYLE = {
    'Pb_interior': dict(color='steelblue',  s=180, zorder=5, label='Pb interior'),
    'Pb_surface':  dict(color='firebrick',  s=180, zorder=5, label='Pb surface'),
}
BOND_STYLE = {
    'apical_surface':  dict(color='red',        lw=2.5, label='apical surface'),
    'apical_interior': dict(color='darkorange',  lw=2.0, label='apical interior'),
    'equatorial':      dict(color='steelblue',   lw=1.2, label='equatorial'),
}
# VESTA species mapping: original role → dummy element that VESTA colours nicely
VESTA_MAP = {
    'Pb_interior': 'Pb',
    'Pb_surface':  'Bi',
    'equatorial':       'I',
    'apical_interior':  'Br',
    'apical_surface':   'Cl',
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load(label):
    with open(os.path.join(DAT, f'geometry_{label}.json')) as f:
        return json.load(f)


def collect_atoms(data):
    """
    Return arrays of Cartesian positions and type labels for Pb and I atoms.
    I positions are reconstructed from Pb + dr for each bond (unique by I_idx).
    """
    pb_pos, pb_types = [], []
    i_pos,  i_types  = {}, {}   # keyed by I_idx to avoid duplicates

    for pb in data['pb_atoms']:
        pb_pos.append(pb['cart'])
        pb_types.append(pb['type'])
        for b in pb['bonds']:
            if b['I_idx'] not in i_pos:
                cart_i = [pb['cart'][k] + b['dr_ang'][k] for k in range(3)]
                i_pos[b['I_idx']]  = cart_i
                i_types[b['I_idx']] = b['bond_type']

    return (np.array(pb_pos), pb_types,
            np.array(list(i_pos.values())), list(i_types.values()))


# ── PNG plot ──────────────────────────────────────────────────────────────────

def plot_png(label):
    data = load(label)
    pb_pos, pb_types, i_pos, i_types = collect_atoms(data)

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    # I atoms
    icolours = {'equatorial': 'lightsteelblue',
                'apical_interior': 'sandybrown',
                'apical_surface':  'lightgreen'}
    for pos, t in zip(i_pos, i_types):
        ax.scatter(*pos, color=icolours[t], s=80, depthshade=False, zorder=3)

    # Pb atoms
    for pos, t in zip(pb_pos, pb_types):
        kw = PB_STYLE[t].copy()
        kw.pop('label', None)
        ax.scatter(*pos, **kw, depthshade=False)

    # Bond lines
    seen = set()
    for pb in data['pb_atoms']:
        for b in pb['bonds']:
            bt = b['bond_type']
            p0 = np.array(pb['cart'])
            p1 = p0 + np.array(b['dr_ang'])
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                    **{k: v for k, v in BOND_STYLE[bt].items() if k != 'label'})

    # Legend proxy
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(color='steelblue', label='Pb interior'),
        Patch(color='firebrick', label='Pb surface'),
        Line2D([0],[0], color='lightsteelblue', lw=3, label='I equatorial'),
        Line2D([0],[0], color='sandybrown',     lw=3, label='I apical interior'),
        Line2D([0],[0], color='lightgreen',     lw=3, label='I apical surface'),
        Line2D([0],[0], color='steelblue',   lw=1.5, label='bond equatorial'),
        Line2D([0],[0], color='darkorange',  lw=2.0, label='bond apical int.'),
        Line2D([0],[0], color='red',         lw=2.5, label='bond apical surf.'),
    ]
    ax.legend(handles=handles, fontsize=7, loc='upper left', framealpha=0.8)

    normal = data['slab_normal']
    ax.set_title(f'{label} — Pb-I bond classification\n'
                 f'slab normal = ({normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f})',
                 fontsize=10)
    ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)'); ax.set_zlabel('z (Å)')
    fig.tight_layout()

    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, f'geometry_{label}.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  PNG saved  → {out}')


# ── VESTA POSCAR ──────────────────────────────────────────────────────────────

def write_vesta_poscar(label):
    data    = load(label)
    pb_pos, pb_types, i_pos, i_types = collect_atoms(data)

    # Assign VESTA species
    atoms_out = []   # list of (species, cart)
    for pos, t in zip(pb_pos, pb_types):
        atoms_out.append((VESTA_MAP[t], pos))
    for pos, t in zip(i_pos, i_types):
        atoms_out.append((VESTA_MAP[t], pos))

    # Group by species (VASP POSCAR requires species blocks)
    from collections import defaultdict
    groups = defaultdict(list)
    for sp, pos in atoms_out:
        groups[sp].append(pos)

    # Read original lattice
    orig = read(os.path.join(VASP_IN, VASP_FILES[label]))
    cell = orig.cell.array

    lines = [f'geometry_{label} — Pb+I classified (Bi=Pb_surf, Br=ap_int, Cl=ap_surf)',
             '1.0']
    for row in cell:
        lines.append(f'  {row[0]:16.10f}  {row[1]:16.10f}  {row[2]:16.10f}')
    order = [s for s in ['Pb','Bi','I','Br','Cl'] if s in groups]
    lines.append('  ' + '  '.join(order))
    lines.append('  ' + '  '.join(str(len(groups[s])) for s in order))
    lines.append('Cartesian')
    for sp in order:
        for pos in groups[sp]:
            lines.append(f'  {pos[0]:16.10f}  {pos[1]:16.10f}  {pos[2]:16.10f}')

    os.makedirs(os.path.join(DAT, 'vesta'), exist_ok=True)
    out = os.path.join(DAT, 'vesta', f'geometry_{label}.vasp')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  VESTA file → {out}')
    print(f'    Pb=interior  Bi=surface_Pb  I=equatorial  Br=apical_int  Cl=apical_surf')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    labels = sys.argv[1:] if len(sys.argv) > 1 else list(VASP_FILES.keys())
    for label in labels:
        print(f'{label}:')
        plot_png(label)
        write_vesta_poscar(label)

if __name__ == '__main__':
    main()
