# Perovskite TB–SOC

An **sp³ + spin–orbit-coupling tight-binding framework with Harrison distance
scaling** for 3D and 2D Ruddlesden–Popper halide perovskites.

A single 16-parameter Hamiltonian (14 free + 2 atomic SOC values fixed from
Boyer-Richard 2016), plus seven slab-specific corrections that vanish in the 3D
limit, describes:

- **3D bulk MAPbI₃ (Pnma)** at the self-consistent-scissor-GW level — mean gap
  offset +22 meV across the eight high-symmetry *k*-points;
- the **2D (BA)₂(MA)ₙ₋₁PbₙI₃ₙ₊₁ series (n = 1–5)** — the canonical 2/n band-gap
  scaling, *E*g = 1.94 + 0.46·(2/n) eV (R² = 0.984);
- **excitons**, through a Rytova–Keldysh + Fröhlich-polaron + Bethe–Salpeter
  chain built on the same single-particle inputs — optical-absorption peaks of
  Blancon *et al.* reproduced to ~100 meV RMS (n = 1 to 11 meV);
- **A-cation transferability**: zero-refit application to γ-CsPbI₃ (Pnma)
  preserves the ss-GW level of theory within ~60 meV.

Harrison distance scaling, V(d) = V₀·(d₀/d)ⁿ on the Pb–I integrals, removes the
need to refit hopping integrals when the geometry changes between phases,
slab thicknesses, or A-cations.

## Canonical parameters

The published parameter set (**Gen 140**) is in
[`params/gen140.json`](params/gen140.json) — energies and hoppings in eV,
n_harrison dimensionless, reference bond length d₀ = 3.14 Å.

## Installation

```bash
git clone https://github.com/bashalak/perovskite-tb-soc.git
cd perovskite-tb-soc
pip install numpy scipy matplotlib ase pymoo
```

## Quick start

```python
import numpy as np, json
from src.python.ham_ortho_harrison import hamiltonian, eigenvalues

params = {k: v for k, v in json.load(open("params/gen140.json"))["parameters"].items()}
# build H(k) for the 3D MAPbI3 Pnma structure and diagonalize at Gamma
ev = eigenvalues("3D", np.zeros(3), params)
print("Gamma-point gap (eV):", ...)
```

## Reproducing the paper figures

The figure scripts read the Gen 140 parameters and the structures in `data/`:

| Script | Output |
|---|---|
| `plot_bands_interactive_harrison.py` | 3D MAPbI₃ Pnma band structure vs ss-GW |
| `plot_2n_scaling.py` | 2/n band-gap scaling of the RP series |
| `plot_mu_scaling.py` | reduced effective masses vs n (bare-band + polaron) |
| `plot_exciton_chain.py` | exciton binding / optical-peak chain |
| `plot_cspbi3_bands.py` | zero-refit transfer to γ-CsPbI₃ |
| `plot_basin_convergence.py` | NSGA-II Pareto-front basin diagnostic |

The NSGA-II fit itself is in `fit_ortho_sanity_harrison_physical.py`
(uses [pymoo](https://pymoo.org/); ~hours on a laptop — not needed to use the
released parameters).

## Repository layout

```
src/python/   framework (ham_ortho_harrison.py), fitting, exciton chain, figure scripts
params/       canonical Gen 140 parameters (gen140.json)
data/         crystal structures (POSCAR) and fit-target data
tests/        unit tests
```

## Citation

A manuscript describing this framework is in preparation
(Shalak, Thébaud, Volonakis, Kepenekian, Katan, Even). A citation / DOI will be
added here on publication.

The atomic SOC values and the sp³+SOC formulation follow
S. Boyer-Richard *et al.*, *Symmetry-Based Tight-Binding Modeling of Halide
Perovskite Semiconductors*, J. Phys. Chem. Lett. **7**, 3833 (2016).

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
