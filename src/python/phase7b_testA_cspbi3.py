"""
phase7b_testA_cspbi3.py — Apply Gen 140 to γ-CsPbI3 Pnma WITHOUT refit.

Test A of Phase 7B: A-cation transferability.
The TB Gen 140 model was fit only to MAPbI3 data. In our sp³+SOC basis the
A-cation (Cs vs MA) doesn't enter the parameters — only the geometry changes.
This test verifies that hypothesis directly.

Source structure: Materials Project mp-1120768 (Pnma γ-CsPbI3).

Reference values to compare against:
  Jiang DFTB:      1.70 eV (β-CsPbI3 Pnma → same as γ in their notation)
  Jiang ss-GW:     1.57 eV
  Marronnier exp:  1.72 eV (ACS Nano 12, 3477 (2018))
  Traoré DFT-1/2:  2.11 eV (known to overshoot for Cs phases)
"""
import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import parse_geometry as pg

# Gen 140 parameters
GEN140 = {
    'Es0':        -2.79,  'Ep0':       +3.29,
    'Es1':       -10.37,  'Ep1':       -4.06,
    'Vss':        -0.67,  'Vsp':       -1.20, 'Vps': -0.22,
    'Vppsig':     -1.75,  'Vpppi':     -1.53,
    'deltaSO0':   +1.30,  'deltaSO1':  +0.90,
    'Vss_II':     -0.76,  'Vsp_II':    -0.22,
    'Vppsig_II':  -0.14,  'Vpppi_II':  +0.10,
    'Vss_IIz':    -0.05,  'Vsp_IIz':   +0.33,
    'Vppsig_IIz': +0.19,  'Vpppi_IIz': -0.49,
    'Vss_PbPb':    0.0,   'Vsp_PbPb':   0.0,
    'Vppsig_PbPb': 0.0,   'Vpppi_PbPb': 0.0,
    'Es1_ap':     -9.38,  'Ep1_ap':    -2.49,  'Ep0_surf': +3.18,
    'n_harrison': +2.36,
}

LABEL = '3D_cspbi3'
# Use b-long Pnma convention (b↔c permutation of MP-1120768) so the band
# structure plot is directly comparable to MAPbI3 (Whitfield b-long setting).
# The physical crystal is identical; only axis labels change.
POSCAR_PATH = os.path.join(_HERE, '..', '..', 'data',
                            'CsPbI3_pnma_mp1120768_blong.vasp')
JSON_PATH = os.path.join(_HERE, '..', '..', 'results', 'data',
                          f'geometry_{LABEL}.json')

# References for the corrected Test A (3D bulk fit was done at ss-GW level,
# so ss-GW is the natural comparison; DFT-1/2 overshoots Cs phases and is
# kept here only for completeness, with a flag).
REFS = {
    # (Eg in eV, μ_reduced in m₀ or None, note)
    'Jiang 2024 ss-GW (Pnma)':       (1.57, 0.11, 'level-of-theory match'),
    'Jiang 2024 DFTB (Pnma)':        (1.70, 0.12, 'cross-check'),
    'Marronnier 2018 exp (Pnma, 90K)': (1.72, None, 'low-T optical gap'),
    'Yang & Sutton 2017 exp (Pnma)':   (None, 0.114, 'magneto-abs μ'),
    'Traoré 2022 DFT-1/2 (Pnma)':    (2.79, None, 'DFT-1/2 overshoots Cs (not a TB target)'),
}


def main():
    print("=" * 78)
    print("Phase 7B Test A — γ-CsPbI3 Pnma transferability test")
    print("=" * 78)

    # ── Step 1: Generate geometry JSON from POSCAR ────────────────────────────
    print(f"\n[1] Parsing geometry from {os.path.basename(POSCAR_PATH)} ...")
    data = pg.parse_structure(label='3D', fpath=POSCAR_PATH)
    # Override the label and add vasp_path so ham_ortho_harrison can locate
    # the structure file for second-neighbour bond finding.
    data['label'] = LABEL
    data['vasp_path'] = os.path.relpath(POSCAR_PATH,
                                        os.path.join(_HERE, '..', '..'))
    pg.print_summary(data)

    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {JSON_PATH}")

    # ── Step 2: Patch ham_ortho_harrison to honour vasp_path field ────────────
    # We monkey-patch the second-neighbour bond finder to consult vasp_path.
    import ham_ortho_harrison as ham
    from ase.io import read as ase_read
    from ase.neighborlist import neighbor_list as ase_neighbor_list

    _orig_finder = ham._find_second_neighbour_bonds
    _BOND_CACHE = ham._BOND_CACHE

    def _patched_finder(geom, cutoff_PbPb=7.5, cutoff_II=6.5):
        key = (geom['label'], round(cutoff_PbPb, 3), round(cutoff_II, 3))
        if key in _BOND_CACHE:
            return _BOND_CACHE[key]

        # If a vasp_path is present in the JSON, use it. Else fall back to
        # the original hard-coded path logic.
        if 'vasp_path' in geom:
            v = geom['vasp_path']
            if not os.path.isabs(v):
                v = os.path.join(_HERE, '..', '..', v)
            vasp_path = v
        elif geom['label'] == '3D':
            vasp_path = os.path.join(_HERE, '..', '..', 'data',
                                      'pos3D2Dtilted', 'POSCAR')
        else:
            vasp_path = os.path.join(_HERE, '..', '..', 'data',
                                      'pos3D2Dpnma_xyz',
                                      f'pnma_xyz_{geom["label"]}.vasp')
        atoms = ase_read(vasp_path)
        pb_set = {a.index for a in atoms if a.symbol == 'Pb'}
        i_set  = {a.index for a in atoms if a.symbol == 'I'}
        ia, ja, Darr = ase_neighbor_list('ijD', atoms, max(cutoff_PbPb, cutoff_II))
        pb_pb_bonds = [(int(i), int(j), D)
                       for i, j, D in zip(ia, ja, Darr)
                       if i in pb_set and j in pb_set and i < j
                       and np.linalg.norm(D) < cutoff_PbPb]
        ii_bonds = [(int(i), int(j), D)
                    for i, j, D in zip(ia, ja, Darr)
                    if i in i_set and j in i_set and i < j
                    and np.linalg.norm(D) < cutoff_II]
        _BOND_CACHE[key] = (pb_pb_bonds, ii_bonds)
        return pb_pb_bonds, ii_bonds

    ham._find_second_neighbour_bonds = _patched_finder

    # ── Step 3: Run TB Gen 140 on CsPbI3 ──────────────────────────────────────
    print(f"\n[2] Computing Eg(Γ) for CsPbI3 Pnma with Gen 140 (zero refit) ...")
    geom = data
    nf = 2 * geom['n_Pb'] + 8 * geom['n_I']
    ivb, icb = nf - 1, nf

    # Γ point
    # Pass '3D' (not LABEL='3D_cspbi3') so ham.hamiltonian uses the 3D-bulk
    # branch: all I-I bonds use II (not IIz) parameters. LABEL is kept on
    # geom['label'] only for caching, not for code-path selection.
    H_G = ham.hamiltonian('3D', [0., 0., 0.], GEN140, geom=geom)
    evals = np.sort(np.linalg.eigvalsh(H_G).real)
    Eg_G = float(evals[icb] - evals[ivb])
    print(f"  Eg(Γ) = {Eg_G:.4f} eV")

    # ── Effective masses via central FD around Γ ──────────────────────────────
    print(f"\n[3] Effective masses (central FD, dk=0.05 Å⁻¹) ...")
    DK = 0.05
    HBAR2_M = 7.6200
    # 7-point batch around Γ
    G = np.array([0., 0., 0.])
    kp = np.array([
        G,
        G + DK * np.array([1, 0, 0]), G - DK * np.array([1, 0, 0]),
        G + DK * np.array([0, 1, 0]), G - DK * np.array([0, 1, 0]),
        G + DK * np.array([0, 0, 1]), G - DK * np.array([0, 0, 1]),
    ])
    ev_batch = ham.eigenvalues_batch('3D', kp, GEN140, geom=geom)
    e0v = ev_batch[0, ivb]
    e0c = ev_batch[0, icb]
    d2v = np.array([(ev_batch[1, ivb] + ev_batch[2, ivb] - 2*e0v) / DK**2,
                    (ev_batch[3, ivb] + ev_batch[4, ivb] - 2*e0v) / DK**2,
                    (ev_batch[5, ivb] + ev_batch[6, ivb] - 2*e0v) / DK**2])
    d2c = np.array([(ev_batch[1, icb] + ev_batch[2, icb] - 2*e0c) / DK**2,
                    (ev_batch[3, icb] + ev_batch[4, icb] - 2*e0c) / DK**2,
                    (ev_batch[5, icb] + ev_batch[6, icb] - 2*e0c) / DK**2])
    mv = [HBAR2_M / d for d in d2v]
    mc = [HBAR2_M / d for d in d2c]
    mu_avg = abs(np.mean(mc) * np.mean(mv) / (np.mean(mc) + np.mean(mv)))
    print(f"  mv [xx,yy,zz] = [{mv[0]:+.3f}, {mv[1]:+.3f}, {mv[2]:+.3f}]")
    print(f"  mc [xx,yy,zz] = [{mc[0]:+.3f}, {mc[1]:+.3f}, {mc[2]:+.3f}]")
    # Per-direction reduced mass μ = |mv|·mc / (|mv| + mc) — proper definition
    mu_dir = [abs(mv[k]) * mc[k] / (abs(mv[k]) + mc[k]) for k in range(3)]
    mu_inplane = 0.5 * (mu_dir[0] + mu_dir[1])
    print(f"  μ [xx,yy,zz]  = [{mu_dir[0]:.3f}, {mu_dir[1]:.3f}, {mu_dir[2]:.3f}]")
    print(f"  μ_in-plane    = {mu_inplane:.3f}")

    # ── Compare to references ─────────────────────────────────────────────────
    # 3D bulk Gen 140 was fit primarily to Jiang ss-GW for MAPbI₃. The natural
    # comparison for Test A is therefore ss-GW for CsPbI₃, not DFT-1/2.
    print(f"\n[4] Comparison to literature references (3D fit was on ss-GW):")
    print(f"  {'Source':<38} {'Eg':>7}  {'ΔEg':>7}  {'μ':>7}  {'Δμ':>7}  note")
    for src, (Eg_r, mu_r, note) in REFS.items():
        Eg_s = f"{Eg_r:.3f}" if Eg_r is not None else "  —  "
        dEg  = f"{Eg_G - Eg_r:+.3f}" if Eg_r is not None else "  —  "
        mu_s = f"{mu_r:.3f}" if mu_r is not None else "  —  "
        dmu  = f"{mu_inplane - mu_r:+.3f}" if mu_r is not None else "  —  "
        print(f"  {src:<38} {Eg_s:>7}  {dEg:>7}  {mu_s:>7}  {dmu:>7}  {note}")

    # ── Save numerical result ──────────────────────────────────────────────────
    out_path = os.path.join(_HERE, '..', '..', 'results', 'data',
                             'phase7b_testA_cspbi3.npz')
    np.savez(out_path,
             Eg_G=Eg_G,
             mv=mv, mc=mc, mu_avg=mu_avg,
             mu_dir=mu_dir, mu_inplane=mu_inplane,
             refs_keys=np.array(list(REFS.keys())))
    print(f"\nSaved {out_path}")

    print("\n" + "=" * 78)
    print(f"SUMMARY: Gen 140 → CsPbI3 Pnma Eg(Γ) = {Eg_G:.3f} eV  (zero refit)")
    print(f"  Primary comparison: Jiang 2024 ss-GW = 1.57 eV  "
          f"(Δ = {Eg_G - 1.57:+.3f})  ← level-of-theory match")
    print(f"  μ_in-plane = {mu_inplane:.3f} m₀  vs  Jiang ss-GW 0.11 / exp 0.114")
    print("=" * 78)


if __name__ == '__main__':
    main()
