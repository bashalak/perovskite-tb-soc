# data/ — Band-structure reference files

## Canonical reference: ss-GW band structure for Pnma γ-MAPbI₃

Provided by **Junke Jiang** (private communication).

| File | Purpose |
|---|---|
| `ss-gw.band.dat` | Raw band data: 96 bands × 393 k-points, two-column (k_dist, E_eV) per band, blank-line separated |
| `plot.gp` | **Authoritative gnuplot script** — defines the k-path labels and tabulates GW effective masses |
| `jiang_gw_pnma_targets.csv` | Extracted high-symmetry-point gaps used as fit targets (this work) |
| `KPOINTS` | **⚠ STALE** — see warning below |

## ⚠ KPOINTS file is stale — do NOT trust its labels

`KPOINTS` says the path is `Z-Γ-X-U-Γ-Y-T-Z-Γ-S`, but the actual band data follows a different path:

### Canonical path: **Γ-X-S-Y-Γ-Z-U-R-T-Z**

Sources (all agree):
- `plot.gp` xtics (line 27 of that file)
- `src/python/plot_bands_interactive_harrison.py:249`
- git commit `2bc121d` "fix: match TB 3D k-path to GW path (G-X-S-Y-G-Z-U-R-T-Z)"

If you regenerate the band data, do it along the canonical path and either replace `KPOINTS` or delete it to avoid confusion.

## High-symmetry-point gap values (from ss-GW)

Extracted with `report/tables/table2_3D_observables.md` workflow; saved verbatim in `jiang_gw_pnma_targets.csv`:

| Label | k_index | Eg (eV) |
|---|---|---|
| Γ | 0 (and 172) | 1.5725 |
| X | 50 | 3.4800 |
| S | 85 | 4.8254 |
| Y | 133 | 2.5457 |
| Z | 222 (and 392) | 3.4801 |
| U | 274 | 2.8645 |
| R | 306 | 3.7946 |
| T | 357 | 4.8722 |

## ss-GW effective masses (from plot.gp header — not fit targets)

| | xx | yy | zz |
|---|---|---|---|
| Hole \|m_v\| | 0.209 | 0.234 | 0.207 |
| Electron m_c | 0.236 | 0.199 | 0.245 |
| Reduced μ | 0.111 | 0.107 | 0.112 |

These are independent of Traoré 2022's DFT-1/2 masses (μ = 0.14–0.16) and are cited in the paper as a cross-check, not a fit target.

## Other contents of `data/`

- `original2DPOSCAR/`, `pos3D2D*/` — geometry files (3D Pnma + 2D RP slabs n=1..5 in various conventions)
- `optuna_study.db`, `surrogate_*.{csv,joblib}` — old optimization artifacts (pre-NSGA-II)

---

*README written 2026-06-11 after resolving the KPOINTS / plot.gp label discrepancy.*
