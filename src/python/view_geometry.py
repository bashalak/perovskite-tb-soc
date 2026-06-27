"""
view_geometry.py — interactive 3-D viewer for Pb-I bond classification.

Generates one HTML file per structure (or one per label given on command line).
Open the HTML in any browser — rotate/zoom/hover to inspect atom types.

Run:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 view_geometry.py        # all structures
    python3 view_geometry.py n5     # single structure
    python3 view_geometry.py n2 n3  # several
"""

import os, sys, json
import numpy as np
import plotly.graph_objects as go

_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.join(_HERE, '..', '..')
DAT     = os.path.join(_ROOT, 'results', 'data')
OUT     = os.path.join(_ROOT, 'results', 'figures')
_VASP   = os.path.join(_ROOT, 'data', 'pos3D2Dtilted')
_VFILES = {'3D': 'POSCAR', 'n1': 'bama_n1.vasp', 'n2': 'bama_n2.vasp',
           'n3': 'bama_n3.vasp', 'n4': 'bama_n4.vasp', 'n5': 'bama_n5.vasp'}

# ── distance shells shown as bond traces ──────────────────────────────────────
#   shell name              : (min_Å, max_Å, color,                  dash,    width)
_EXTRA_BONDS = {
    'I–I  1st (~4–5 Å)'     : (4.0,  5.15, 'rgba(0,200,220,0.55)',  'solid', 2),
    'I–I  2nd (~5–6.5 Å)'   : (5.15, 6.55, 'rgba(200,0,200,0.50)',  'solid', 2),
    'Pb–Pb 1st (~6–7.5 Å)'  : (6.0,  7.5,  'rgba(255,180,0,0.65)',  'solid', 2),
    'Pb–Pb 2nd (~8–9.5 Å)'  : (8.0,  9.5,  'rgba(255,140,0,0.55)',  'dot',   2),
    'Pb–Pb 3rd (~10–12 Å)'  : (10.0, 12.0, 'rgba(200,100,0,0.45)',  'dot',   2),
    'Pb–Pb 4th (~12–13.5 Å)': (12.0, 13.5, 'rgba(160,80,0,0.40)',   'dash',  1),
}
_PBPB_SHELL_NAMES = [k for k in _EXTRA_BONDS if k.startswith('Pb–Pb')]

LABELS = ['3D', 'n1', 'n2', 'n3', 'n4', 'n5']

# ── visual style per type ─────────────────────────────────────────────────────
ATOM_STYLE = {
    'Pb_interior':       dict(color='royalblue',   size=14, symbol='circle'),
    'Pb_surface':        dict(color='crimson',      size=14, symbol='circle'),
    'equatorial':        dict(color='lightgray',    size=7,  symbol='circle'),
    'apical_interior':   dict(color='orange',       size=9,  symbol='diamond'),
    'apical_surface':    dict(color='limegreen',    size=9,  symbol='diamond'),
}
BOND_COLOR = {
    'equatorial':       'rgba(150,150,200,0.4)',
    'apical_interior':  'rgba(255,160,0,0.7)',
    'apical_surface':   'rgba(50,200,50,0.9)',
}


def load(label):
    with open(os.path.join(DAT, f'geometry_{label}.json')) as f:
        return json.load(f)


def _extra_bond_traces(label):
    """
    Build Plotly line traces for I-I and Pb-Pb bonds, split by distance shell.
    Each trace is hidden by default (visible='legendonly').
    Also returns Pb-Pb distance-label traces at bond midpoints.
    """
    from ase.io import read
    from ase.neighborlist import neighbor_list

    atoms   = read(os.path.join(_VASP, _VFILES[label]))
    pb_set  = {a.index for a in atoms if a.symbol == 'Pb'}
    i_set   = {a.index for a in atoms if a.symbol == 'I'}
    pos     = atoms.get_positions()

    # Cutoff covers the largest shell defined above (Pb-Pb 4th at ~13.5 Å)
    ia, ja, D = neighbor_list('ijD', atoms, 13.6)

    # collect line segments per shell
    segs = {name: dict(x=[], y=[], z=[], n=0)
            for name in _EXTRA_BONDS}
    # midpoint label data, one dict per Pb-Pb shell + one for I-I 1st
    pbpb_mid = {name: dict(x=[], y=[], z=[], txt=[]) for name in _PBPB_SHELL_NAMES}
    ii1_mid  = dict(x=[], y=[], z=[], txt=[])

    seen_pairs = set()   # avoid double-counting i<j and j<i
    for k in range(len(ia)):
        d   = float(np.linalg.norm(D[k]))
        si  = atoms[ia[k]].symbol
        sj  = atoms[ja[k]].symbol
        if not ((si == sj == 'I') or (si == sj == 'Pb')):
            continue
        p0 = pos[ia[k]]
        p1 = p0 + D[k]
        for name, (dlo, dhi, *_) in _EXTRA_BONDS.items():
            if dlo <= d < dhi:
                if 'Pb' in name and si != 'Pb':
                    continue
                if 'I–I' in name and si != 'I':
                    continue
                segs[name]['x'] += [p0[0], p1[0], None]
                segs[name]['y'] += [p0[1], p1[1], None]
                segs[name]['z'] += [p0[2], p1[2], None]
                segs[name]['n'] += 1
                pair = tuple(sorted((int(ia[k]), int(ja[k]))))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    mid = (p0 + p1) / 2.0
                    if name in pbpb_mid:
                        pbpb_mid[name]['x'].append(mid[0])
                        pbpb_mid[name]['y'].append(mid[1])
                        pbpb_mid[name]['z'].append(mid[2])
                        pbpb_mid[name]['txt'].append(f"{d:.3f}")
                    elif 'I–I  1st' in name:
                        ii1_mid['x'].append(mid[0])
                        ii1_mid['y'].append(mid[1])
                        ii1_mid['z'].append(mid[2])
                        ii1_mid['txt'].append(f"{d:.3f}")
                break

    traces = []
    for name, (_, _, color, dash, width) in _EXTRA_BONDS.items():
        s = segs[name]
        if not s['x']:
            continue
        traces.append(go.Scatter3d(
            x=s['x'], y=s['y'], z=s['z'],
            mode='lines',
            name=f'{name} [{s["n"]//2} bonds]',
            line=dict(color=color, width=width, dash=dash),
            hoverinfo='skip',
            visible='legendonly',
        ))

    # Pb-Pb distance labels at midpoints — one trace per shell
    shell_label_colours = {
        _PBPB_SHELL_NAMES[0]: 'rgb(180,120,0)',     # 1st
        _PBPB_SHELL_NAMES[1]: 'rgb(200,100,0)',     # 2nd
        _PBPB_SHELL_NAMES[2]: 'rgb(160,80,0)',      # 3rd
        _PBPB_SHELL_NAMES[3]: 'rgb(120,60,0)',      # 4th
    }
    for name in _PBPB_SHELL_NAMES:
        mid = pbpb_mid[name]
        if not mid['x']:
            continue
        # Shorten shell label for legend ("Pb–Pb 1st (~6–7.5 Å)" → "Pb-Pb 1st")
        short = name.split(' (')[0].replace('–', '-')
        traces.append(go.Scatter3d(
            x=mid['x'], y=mid['y'], z=mid['z'],
            mode='text',
            name=f'dist {short} (Å)  [{len(mid["x"])}]',
            text=mid['txt'],
            textposition='middle center',
            textfont=dict(size=9, color=shell_label_colours[name]),
            hoverinfo='skip',
            visible='legendonly',
        ))

    # I-I 1st-neighbour distance labels at midpoints
    if ii1_mid['x']:
        traces.append(go.Scatter3d(
            x=ii1_mid['x'], y=ii1_mid['y'], z=ii1_mid['z'],
            mode='text',
            name=f'dist I-I 1st (Å)  [{len(ii1_mid["x"])}]',
            text=ii1_mid['txt'],
            textposition='middle center',
            textfont=dict(size=9, color='rgb(0,160,170)'),
            hoverinfo='skip',
            visible='legendonly',
        ))
    return traces


def _basis_vector_traces(label):
    """Draw cell basis vectors a, b, c from origin, with letter labels at tips."""
    data = load(label)
    cell = np.array(data['lattice'])      # shape (3, 3), rows = a, b, c
    origin = np.zeros(3)
    colours = ['#E74C3C', '#27AE60', '#3498DB']    # a=red, b=green, c=blue
    names   = ['a', 'b', 'c']

    traces = []
    for k in range(3):
        end = origin + cell[k]
        # line from origin to end-of-vector
        traces.append(go.Scatter3d(
            x=[origin[0], end[0]], y=[origin[1], end[1]], z=[origin[2], end[2]],
            mode='lines',
            name=f'basis {names[k]} (|.|={np.linalg.norm(cell[k]):.2f} Å)',
            line=dict(color=colours[k], width=6),
            hoverinfo='skip',
            visible='legendonly',
            legendgroup='basis',
        ))
        # letter label at the tip
        traces.append(go.Scatter3d(
            x=[end[0]], y=[end[1]], z=[end[2]],
            mode='text',
            text=[f'<b>{names[k]}</b>'],
            textposition='top right',
            textfont=dict(size=14, color=colours[k]),
            hoverinfo='skip',
            visible='legendonly',
            legendgroup='basis',
            showlegend=False,
        ))
    return traces


# Per-element style for atoms in neighbour cells.
# Only Pb and I are drawn — organic spacer atoms (C, N, H) are omitted to keep
# the neighbour-cell view clean and focused on the inorganic Pb-I framework.
_NEIGHBOUR_ELEM_STYLE = {
    'Pb': dict(color='royalblue', size=10),
    'I':  dict(color='lightgray', size=6),
}


def _neighbour_cell_traces(label, opacity=0.35):
    """
    Build 6 toggleable traces showing atoms in face-adjacent neighbour cells
    (±a, ±b, ±c). All atoms shown with semi-transparent markers.

    Each trace is hidden by default — click in the legend to toggle on.
    Useful for visualising the PBC structure and checking that the slab
    extends correctly along chosen lattice vectors.
    """
    from ase.io import read
    poscar_path = os.path.join(_VASP, _VFILES[label])
    atoms = read(poscar_path)
    cell = atoms.cell.array
    pos = atoms.get_positions()
    syms = [a.symbol for a in atoms]

    shifts = [
        ('+a', (1, 0, 0)),  ('-a', (-1, 0, 0)),
        ('+b', (0, 1, 0)),  ('-b', (0, -1, 0)),
        ('+c', (0, 0, 1)),  ('-c', (0, 0, -1)),
    ]

    traces = []
    for name, (ia, ib, ic) in shifts:
        delta = ia * cell[0] + ib * cell[1] + ic * cell[2]
        # Group atoms by element so each gets the right size/colour
        for elem, st in _NEIGHBOUR_ELEM_STYLE.items():
            mask = [s == elem for s in syms]
            if not any(mask):
                continue
            xs = [pos[k][0] + delta[0] for k in range(len(pos)) if mask[k]]
            ys = [pos[k][1] + delta[1] for k in range(len(pos)) if mask[k]]
            zs = [pos[k][2] + delta[2] for k in range(len(pos)) if mask[k]]
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='markers',
                name=f'nbr cell {name}' if elem == 'Pb' else None,
                marker=dict(size=st['size'], color=st['color'],
                            opacity=opacity,
                            line=dict(width=0, color='black')),
                hoverinfo='skip',
                visible='legendonly',
                legendgroup=f'neighbour_{name}',
                showlegend=(elem == 'Pb'),   # only one legend entry per shift
            ))
    return traces


def build_figure(label):
    data = load(label)
    traces = []

    # ── collect atom positions grouped by type ────────────────────────────────
    pb_groups   = {}   # type → list of (x,y,z,hover)
    i_groups    = {}   # bond_type → list of (x,y,z,hover)
    i_seen      = {}   # I_idx → cart  (avoid duplicates)

    for pb in data['pb_atoms']:
        t = pb['type']
        pb_groups.setdefault(t, []).append(
            (*pb['cart'], f"<b>{t}</b><br>idx={pb['idx']}<br>"
                          f"xyz=({pb['cart'][0]:.2f},{pb['cart'][1]:.2f},{pb['cart'][2]:.2f})")
        )
        for b in pb['bonds']:
            if b['I_idx'] not in i_seen:
                cart_i = [pb['cart'][k] + b['dr_ang'][k] for k in range(3)]
                i_seen[b['I_idx']] = cart_i
                bt = b['bond_type']
                i_groups.setdefault(bt, []).append(
                    (*cart_i, f"<b>{bt}</b><br>idx={b['I_idx']}<br>"
                              f"dist={b['dist_ang']:.3f} Å<br>"
                              f"lmn=({b['lmn'][0]:.3f},{b['lmn'][1]:.3f},{b['lmn'][2]:.3f})<br>"
                              f"proj={b['proj_normal']:.3f}")
                )

    # ── I atom traces ─────────────────────────────────────────────────────────
    all_i_xs, all_i_ys, all_i_zs, all_i_labels = [], [], [], []
    for bt, pts in sorted(i_groups.items()):
        xs, ys, zs, hs = zip(*pts)
        st = ATOM_STYLE[bt]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            name=f'I {bt}',
            marker=dict(size=st['size'], color=st['color'],
                        symbol=st['symbol'], opacity=0.85,
                        line=dict(width=0.5, color='black')),
            hovertext=hs, hoverinfo='text',
        ))
        all_i_xs += list(xs)
        all_i_ys += list(ys)
        all_i_zs += list(zs)
        all_i_labels += [f'({x:.2f}, {y:.2f}, {z:.2f})'
                         for x, y, z in zip(xs, ys, zs)]

    # ── Pb atom traces ────────────────────────────────────────────────────────
    all_pb_xs, all_pb_ys, all_pb_zs, all_pb_labels = [], [], [], []
    for pt, pts in sorted(pb_groups.items()):
        xs, ys, zs, hs = zip(*pts)
        st = ATOM_STYLE[pt]
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            name=pt,
            marker=dict(size=st['size'], color=st['color'],
                        symbol=st['symbol'], opacity=1.0,
                        line=dict(width=1, color='black')),
            hovertext=hs, hoverinfo='text',
        ))
        all_pb_xs += list(xs)
        all_pb_ys += list(ys)
        all_pb_zs += list(zs)
        all_pb_labels += [f'({x:.2f}, {y:.2f}, {z:.2f})'
                          for x, y, z in zip(xs, ys, zs)]

    # ── Pb coordinate labels (toggleable in legend, default off) ─────────────
    if all_pb_xs:
        traces.append(go.Scatter3d(
            x=all_pb_xs, y=all_pb_ys, z=all_pb_zs,
            mode='text',
            name='Pb coordinates (x,y,z)',
            text=all_pb_labels,
            textposition='top right',
            textfont=dict(size=9, color='midnightblue'),
            hoverinfo='skip',
            visible='legendonly',
        ))

    # ── I coordinate labels (toggleable in legend, default off) ───────────────
    if all_i_xs:
        traces.append(go.Scatter3d(
            x=all_i_xs, y=all_i_ys, z=all_i_zs,
            mode='text',
            name='I coordinates (x,y,z)',
            text=all_i_labels,
            textposition='bottom right',
            textfont=dict(size=8, color='gray'),
            hoverinfo='skip',
            visible='legendonly',
        ))

    # ── bond lines (one trace per bond type for legend) ───────────────────────
    bond_lines  = {bt: dict(x=[], y=[], z=[]) for bt in BOND_COLOR}
    label_pts   = {bt: dict(x=[], y=[], z=[], txt=[]) for bt in BOND_COLOR}
    for pb in data['pb_atoms']:
        for b in pb['bonds']:
            bt   = b['bond_type']
            p0   = pb['cart']
            p1   = [p0[k] + b['dr_ang'][k] for k in range(3)]
            mid  = [(p0[k] + p1[k]) / 2.0 for k in range(3)]
            bond_lines[bt]['x'] += [p0[0], p1[0], None]
            bond_lines[bt]['y'] += [p0[1], p1[1], None]
            bond_lines[bt]['z'] += [p0[2], p1[2], None]
            label_pts[bt]['x'].append(mid[0])
            label_pts[bt]['y'].append(mid[1])
            label_pts[bt]['z'].append(mid[2])
            label_pts[bt]['txt'].append(f"{b['dist_ang']:.3f}")

    for bt, xyz in bond_lines.items():
        if not xyz['x']:
            continue
        traces.append(go.Scatter3d(
            x=xyz['x'], y=xyz['y'], z=xyz['z'],
            mode='lines',
            name=f'bond {bt}',
            line=dict(color=BOND_COLOR[bt], width=3),
            hoverinfo='skip',
        ))

    # ── distance labels at bond midpoints (toggle in legend) ─────────────────
    for bt, lp in label_pts.items():
        if not lp['x']:
            continue
        traces.append(go.Scatter3d(
            x=lp['x'], y=lp['y'], z=lp['z'],
            mode='text',
            name=f'dist {bt} (Å)',
            text=lp['txt'],
            textposition='middle center',
            textfont=dict(size=10, color=BOND_COLOR[bt].replace('0.4', '1.0').replace('0.7', '1.0').replace('0.9', '1.0')),
            hoverinfo='skip',
            visible='legendonly',     # hidden until clicked; toggle in legend
        ))

    # ── slab normal arrow (from centre of Pb cloud) ───────────────────────────
    if label != '3D':
        pb_pos = np.array([pb['cart'] for pb in data['pb_atoms']])
        centre = pb_pos.mean(axis=0)
        n = np.array(data['slab_normal'])
        proj  = pb_pos @ n
        span  = proj.max() - proj.min()     # spread of Pb along normal
        arrow_len = span * 0.3
        tip = centre + n * arrow_len
        traces.append(go.Scatter3d(
            x=[centre[0], tip[0]], y=[centre[1], tip[1]], z=[centre[2], tip[2]],
            mode='lines+text',
            name='slab normal',
            line=dict(color='black', width=4),
            text=['', 'n̂'],
            textposition='top center',
            hoverinfo='skip',
        ))

    # ── 2nd / 3rd neighbour bonds + Pb-Pb / I-I 1st distance labels ─────────
    traces += _extra_bond_traces(label)

    # ── Cell basis vectors a, b, c (toggleable in legend) ────────────────────
    traces += _basis_vector_traces(label)

    # ── Face-adjacent neighbour cells ±a, ±b, ±c (toggleable, hidden) ────────
    traces += _neighbour_cell_traces(label)

    layout = go.Layout(
        title=dict(text=f'<b>{label}</b> — Pb-I bond classification<br>'
                        f'<sup>Blue=Pb_int  Red=Pb_surf  Gray=I_eq  '
                        f'Orange=I_ap_int  Green=I_ap_surf  '
                        f'| toggle in legend: Cyan=I-I 1st  Magenta=I-I 2nd  Gold=Pb-Pb</sup>',
                   x=0.5),
        scene=dict(
            xaxis_title='x (Å)', yaxis_title='y (Å)', zaxis_title='z (Å)',
            aspectmode='data',
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(l=0, r=0, t=60, b=0),
        width=900, height=700,
    )
    return go.Figure(data=traces, layout=layout)


def _print_pb_i_distances(label):
    """Print summary statistics of 1st-neighbour Pb-I bond lengths."""
    data = load(label)
    by_type = {}     # bond_type → list of distances
    all_d   = []
    for pb in data['pb_atoms']:
        for b in pb['bonds']:
            by_type.setdefault(b['bond_type'], []).append(b['dist_ang'])
            all_d.append(b['dist_ang'])
    all_d = np.array(all_d)
    print(f'  Pb-I bond distances [Å]  (n_bonds = {len(all_d)})')
    print(f'    overall : min={all_d.min():.4f}  max={all_d.max():.4f}  '
          f'mean={all_d.mean():.4f}  std={all_d.std():.4f}')
    for bt in sorted(by_type):
        d = np.array(by_type[bt])
        print(f'    {bt:<16}: count={len(d):>3}  '
              f'min={d.min():.4f}  max={d.max():.4f}  mean={d.mean():.4f}')


def main():
    labels = sys.argv[1:] if len(sys.argv) > 1 else LABELS
    os.makedirs(OUT, exist_ok=True)
    for label in labels:
        print(f'\n=== {label} ===')
        _print_pb_i_distances(label)
        print(f'  Building HTML ...', end=' ', flush=True)
        fig = build_figure(label)
        out = os.path.join(OUT, f'geometry_{label}.html')
        fig.write_html(out, include_plotlyjs='cdn')
        print(f'→ {out}')
    print('\nOpen the HTML files in your browser.')
    print('In the plot legend, toggle "dist <bond_type> (Å)" entries to show '
          'bond-length labels at bond midpoints.')


if __name__ == '__main__':
    main()
