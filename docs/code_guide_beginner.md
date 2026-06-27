# Beginner's Code Guide — Perovskite TB-SOC

This document explains every script in this project line by line, for someone who knows basic Python but has never worked with scientific computing or machine learning before.

**How to read this:** Each script has a short explanation of what it does, then each important block of code is shown with an explanation in plain English underneath.

---

## Table of Contents

1. [ctypesroutines.py — The bridge between Python and Fortran](#1-ctypesroutinespy)
2. [absorption_2D_fortran.py — Computing optical absorption spectra](#2-absorption_2d_fortranpy)
3. [effective_mass_tensor.py — How heavy are the electrons and holes?](#3-effective_mass_tensorpy)
4. [exciton_binding.py — Bound electron-hole pairs](#4-exciton_bindingpy)
5. [comparison_table.py — How well does the model match experiment?](#5-comparison_tablepy)
6. [convergence_plot.py — Did the optimizer converge?](#6-convergence_plotpy)
7. [tb_optimize.py — Finding the best parameters](#7-tb_optimizepy)
8. [generate_surrogate_data.py — Building the ML training dataset](#8-generate_surrogate_datapy)
9. [train_surrogate.py — Teaching the ML model](#9-train_surrogatepy)
10. [optuna_optimize.py — Smart parameter search](#10-optuna_optimizepy)
11. [vqe_tb.py — Quantum algorithm for band energies](#11-vqe_tbpy)
12. [rytova_keldysh.py — Exciton binding via real-space solver (Phase 5 Level-1)](#12-rytova_keldyshpy)
13. [bse_2d.py — k-space Bethe-Salpeter Equation (Phase 5 Level-2)](#13-bse_2dpy)
14. [absorption_elliott.py — Optical absorption spectrum with excitonic peaks (Phase 5)](#14-absorption_elliottpy)
15. [parse_geometry.py — Extract and classify Pb-I bonds from VASP structures (Phase 6)](#15-parse_geometrypy)
16. [ham_ortho.py — Orthorhombic TB+SOC Hamiltonian (Phase 6)](#16-ham_orthopy)
17. [plot_bands_interactive.py — Interactive 6-panel band-structure viewer (Phase 6)](#17-plot_bands_interactivepy)
18. [fit_ortho.py — NSGA2 fitting for the orthorhombic model (Phase 6)](#18-fit_orthopy)
19. [view_geometry.py — Interactive 3-D HTML structure viewer (Phase 6)](#19-view_geometrypy)
20. [visualize_geometry.py — Static PNG and VESTA output (Phase 6)](#20-visualize_geometrypy)

---

## Background: What is the project doing?

A **perovskite** is a type of crystal used in solar cells. Its electronic properties (how well it conducts electricity, what color of light it absorbs) depend on its **band gap** — the minimum energy needed to excite an electron.

This project builds a **mathematical model** of the perovskite's electrons. The model has 26 adjustable numbers called **parameters**. The goal is to find the parameter values that make the model match experimental measurements.

The model is written in **Fortran** (a fast programming language for numerical computation). Python is used to call that Fortran code, analyze results, and optimize the parameters.

---

## 1. ctypesroutines.py

**What it does:** Acts as a translator between Python and the compiled Fortran code. Python cannot call Fortran directly, so this file builds a bridge using a Python module called `ctypes`.

Think of it like a plug adapter: the Fortran code is a European plug, Python expects a US socket, and this file is the adapter.

---

```python
import numpy as np
import ctypes
from ctypes import c_double, c_int, c_void_p, POINTER
```

**Explanation:**
- `numpy` (shortened to `np`) is the standard Python library for working with arrays of numbers.
- `ctypes` is a Python module that lets you call functions from compiled code (like the Fortran `.so` file).
- `c_double`, `c_int`, `c_void_p` are C data types. Fortran and C use the same memory layout, so ctypes uses C types to talk to Fortran. `c_double` = a decimal number (64-bit float), `c_int` = an integer, `c_void_p` = a pointer (a memory address).

---

```python
lib_3rd = ctypes.CDLL('./libhamiltonian_3rd.so')
```

**Explanation:**
- This loads the compiled Fortran library file (`libhamiltonian_3rd.so`) into memory. The `.so` extension means "shared object" — a compiled library file on Linux (like a `.dll` on Windows).
- `./` means "in the current directory". This is why you need to run scripts from `src/python/` or copy the `.so` there.
- After this line, `lib_3rd` is an object that represents the loaded library. You can call functions from it.

---

```python
hamiltonian_func_3rd = lib_3rd.hamiltonian_cubic3d_2nd
hamiltonian_func_3rd.argtypes = [
    c_double, c_double, c_double,   # kx, ky, kz
    c_double, c_double, ...
    c_void_p,                       # ham array pointer
    c_int                           # n
]
hamiltonian_func_3rd.restype = None
```

**Explanation:**
- `lib_3rd.hamiltonian_cubic3d_2nd` gets the function named `hamiltonian_cubic3d_2nd` from the library.
- `.argtypes` tells Python exactly what types of arguments the Fortran function expects. This is important: Fortran is strict about data types. If you pass the wrong type, the program crashes or gives wrong results.
- The last two arguments are special: `c_void_p` is a pointer to where the output matrix should be stored (Python passes the memory address of a numpy array), and `c_int` is the matrix size.
- `.restype = None` means the function doesn't return a value — instead it writes results directly into the array you pass to it (common in Fortran/C style programming).

---

```python
def hamiltonian_cubic3D_fortran(kx, ky, kz, Es0, Ep0, ...):
    n = 32
    ham = np.zeros((n, n), dtype=np.complex128, order='F')
    hamiltonian_func_3rd(kx, ky, kz, Es0, Ep0, ..., ham.ctypes.data, n)
    return ham
```

**Explanation:**
- This is a Python wrapper function — a friendly interface that hides the complexity of ctypes.
- `n = 32`: the Hamiltonian matrix is 32×32. Why 32? The unit cell of the 3D perovskite has 5 atoms × 4 orbitals × 2 spin states per atom-orbital = 32 basis states.
- `np.zeros((n, n), dtype=np.complex128, order='F')`: creates a 32×32 matrix of complex numbers, initialized to zero. `complex128` means each number has a 64-bit real part and a 64-bit imaginary part. `order='F'` means the memory is arranged in Fortran column-major order (Fortran stores columns first, Python stores rows first — they must match or the matrix comes out transposed).
- `ham.ctypes.data`: gets the memory address (pointer) of the numpy array so Fortran can write directly into it.
- After the Fortran function runs, `ham` contains the completed Hamiltonian matrix.

---

```python
def eigenvalues_only_fortran(ham):
    n = ham.shape[0]
    eigenvalues = np.zeros(n, dtype=np.complex128)
    eigen_func(ham.ctypes.data_as(c_void_p),
               eigenvalues.ctypes.data_as(c_void_p), n)
    return eigenvalues.real
```

**Explanation:**
- Takes a Hamiltonian matrix as input and returns its eigenvalues.
- **What are eigenvalues?** For a Hamiltonian matrix, eigenvalues are the allowed energy levels of the system. Each eigenvalue is one energy level an electron can occupy. Sorted from lowest to highest, they form the band structure.
- The function calls Fortran's LAPACK ZHEEV routine (a standard library for diagonalizing complex Hermitian matrices).
- `.real` at the end discards the imaginary part — for a correct Hermitian matrix the imaginary part is always zero (or very close to it due to floating-point arithmetic).

---

## 2. absorption_2D_fortran.py

**What it does:** Computes the optical absorption coefficient α(ω) — how strongly the material absorbs light at each photon energy ω. Runs for both 3D bulk and 2D slabs of thickness n=1 through n=5.

---

```python
import sys, os
sys.path.insert(0, '/home/baker/newparam5')
os.chdir('/home/baker/newparam5')
```

**Explanation:**
- `sys.path.insert(0, ...)` adds a directory to Python's search path for modules. This is needed so Python can find `ctypesroutines.py` in that directory.
- `os.chdir(...)` changes the current working directory. This is needed because `ctypesroutines.py` loads `./libhamiltonian_3rd.so` using a relative path — if the working directory is wrong, the library won't be found.

---

```python
PARAMS = np.array([
    -2.75, +3.42, -9.42, +2.58, +1.95, +0.74, -3.64, -1.86, -0.90, +1.30, +0.90,
    ...
])
```

**Explanation:**
- This is an array of the 26 best-fit parameter values in electron-volts (eV). These are the numbers that were found by the optimizer to best reproduce experimental band gaps.
- They represent energy levels (Es0, Ep0, Es1, Ep1), hopping integrals (how easily electrons jump between atoms: Vss, Vsp, etc.), spin-orbit coupling strengths (deltaSO0, deltaSO1), and surface corrections for 2D slabs.

---

```python
NK = 30
k_vals = np.linspace(0, np.pi / A_LAT, NK)
kx_grid, ky_grid = np.meshgrid(k_vals, k_vals)
```

**Explanation:**
- The Brillouin zone is a mathematical box in "k-space" (momentum space). To compute absorption, we need to sum contributions from all electron momenta.
- `NK = 30` means we sample 30 points along each axis.
- `np.linspace(0, π/a, 30)` creates 30 evenly spaced points from 0 to π/a (the edge of the Brillouin zone).
- `np.meshgrid` takes two 1D arrays and creates a 2D grid — every combination of kx and ky values. For 2D slabs, kz=0, so we only need kx and ky.

---

```python
N_OMEGA = 6000
OMEGA_MIN, OMEGA_MAX = 0.5, 6.0
energies = np.linspace(OMEGA_MIN, OMEGA_MAX, N_OMEGA)
alphas = np.zeros(N_OMEGA)
```

**Explanation:**
- `N_OMEGA = 6000`: the absorption spectrum is computed at 6000 photon energies, from 0.5 eV to 6.0 eV.
- `alphas` starts as all zeros; each k-point's contribution gets added in.

---

```python
delta_k = 1e-5
H_plus  = build_hamiltonian(kx + delta_k, ky, kz)
H_minus = build_hamiltonian(kx - delta_k, ky, kz)
dH_dkx  = (H_plus - H_minus) / (2 * delta_k)
```

**Explanation:**
- This computes the **derivative** of the Hamiltonian with respect to kx using the central finite difference formula: `df/dx ≈ (f(x+h) - f(x-h)) / (2h)`.
- Why do we need this derivative? The absorption formula requires **momentum matrix elements** — overlaps between electron wavefunctions — which are proportional to ∂H/∂kx.
- We build the Hamiltonian at kx+δ and kx−δ, then take the difference divided by 2δ. This is called "numerical differentiation."

---

```python
for ic, iv in product(range(n_cb), range(n_vb)):
    dE = E_cb[ic] - E_vb[iv]
    matel = abs(psi_cb[ic] @ dH_dkx @ psi_vb[iv])**2
    gauss = np.exp(-0.5 * ((energies - dE) / ETA)**2) / (ETA * np.sqrt(2*np.pi))
    alphas += matel * gauss
```

**Explanation:**
- For each pair of (conduction band, valence band) states, this computes the contribution to absorption.
- `dE = E_cb - E_vb`: energy difference between upper and lower state — the photon energy needed to excite this transition.
- `psi_cb @ dH_dkx @ psi_vb`: the momentum matrix element — how strongly light couples these two states. `@` is the matrix multiplication operator in Python.
- `gauss`: a Gaussian function centered at energy `dE` with width `ETA`. This models that each transition has some energy broadening (due to temperature, disorder). The Gaussian integrates to 1, so it correctly distributes the oscillator strength across nearby energies.
- The sum over all transitions builds up the full absorption spectrum.

---

## 3. effective_mass_tensor.py

**What it does:** Computes the effective mass of electrons (at the conduction band minimum) and holes (at the valence band maximum). The effective mass tells you how easily carriers accelerate under an electric field.

---

```python
def effective_mass(band_idx, k0, direction, delta_k=0.05):
    E0 = get_eigenvalue(k0, band_idx)
    E_plus  = get_eigenvalue(k0 + delta_k * direction, band_idx)
    E_minus = get_eigenvalue(k0 - delta_k * direction, band_idx)
    inv_mass = (E_plus - 2*E0 + E_minus) / (delta_k**2)
    return 1.0 / inv_mass
```

**Explanation:**
- The effective mass is related to the **curvature** of the energy band: `1/m* = (1/ħ²) d²E/dk²`.
- The second derivative is approximated numerically: `d²E/dk² ≈ (E(k+δ) - 2E(k) + E(k-δ)) / δ²`. This is the standard finite difference formula for a second derivative.
- A small curvature (flat band) → large effective mass → heavy carrier → slow in electric field.
- A large curvature (steep band) → small effective mass → light carrier → fast in electric field.
- The result is in units of the free electron mass m₀.

---

## 4. exciton_binding.py

**What it does:** Computes the exciton binding energy — how strongly an excited electron and the hole it left behind attract each other.

---

```python
RYD = 13.6056   # eV — the Rydberg constant
eps_well    = 5.75
eps_barrier = 1.0

eps_eff_2d = (n_vals * eps_well + 2 * eps_barrier) / (n_vals + 2)
eb_2d_model = RYD * mu_2d / eps_eff_2d**2
```

**Explanation:**
- **Exciton:** When light excites an electron, it leaves behind a positively charged "hole." The electron and hole attract each other (opposite charges), forming a bound pair called an exciton. To produce a free electron-hole pair (useful for solar cells), you need to overcome this binding energy.
- **Wannier-Mott formula:** `Eb = Ry × μ/m₀ / εeff²`
  - `Ry = 13.6 eV` — the hydrogen atom binding energy (same physics: electron orbiting positive charge)
  - `μ` — reduced mass (μ = me × mh / (me + mh)), computed from the TB effective masses
  - `εeff` — effective dielectric constant (how well the surrounding material screens the attraction)
- **Slab-average dielectric:** In 2D slabs, part of the electric field passes through the organic barrier material (low ε) and part through the perovskite well (high ε). The formula `εeff(n) = (n·εwell + 2·εbarrier)/(n+2)` averages them proportional to thickness.
- As n decreases (thinner slab), εeff decreases, so Eb increases — thinner slabs have larger exciton binding energies. This is the quantum confinement effect.

---

## 5. comparison_table.py

**What it does:** Creates a table comparing three columns of band gaps: Experiment, DFT/Wannier reference, and this TB model. Highlights errors in green (good), yellow (acceptable), or red (bad).

---

```python
errors = (tb_values - exp_values) / exp_values * 100   # percent error
rms = np.sqrt(np.mean(errors**2))
mae = np.mean(np.abs(errors))
r2  = 1 - np.sum((tb_values - exp_values)**2) / np.sum((exp_values - exp_values.mean())**2)
```

**Explanation:**
- **Percent error:** `(predicted - actual) / actual × 100`. A value of 2% means the model is off by 2%.
- **RMS (Root Mean Square error):** `√(mean(errors²))`. Squaring before averaging penalizes large errors more than small ones. Taking the square root puts it back in the same units as the data (percent in this case).
- **MAE (Mean Absolute Error):** `mean(|errors|)`. Simpler than RMS — just the average size of the error, regardless of direction.
- **R² (R-squared, coefficient of determination):** Measures how much of the variation in the data the model explains. The formula compares the model's error (numerator) to the total variation in the data (denominator). R²=1 means perfect fit; R²=0 means the model is no better than just predicting the mean.

---

```python
colors = []
for err in errors:
    if abs(err) < 5:
        colors.append('lightgreen')
    elif abs(err) < 15:
        colors.append('lightyellow')
    else:
        colors.append('lightcoral')
```

**Explanation:**
- This assigns a color to each cell based on how large the error is.
- Green = error < 5% (good), Yellow = 5–15% (acceptable), Red = > 15% (poor).
- The colors are passed to matplotlib's table function to draw a colored HTML-style table.

---

## 6. convergence_plot.py

**What it does:** Shows how the optimizer improved over generations — did it converge to a solution or is it still searching?

---

```python
history = np.load('optimization_history.npz')
best_per_gen = history['best_f_per_gen']

plt.plot(best_per_gen, lw=2)
plt.xlabel('Generation')
plt.ylabel('Best objective value')
plt.yscale('log')
```

**Explanation:**
- `np.load`: loads a `.npz` file — a compressed numpy archive that stores arrays.
- `best_per_gen`: an array where entry `i` is the best objective function value found up to generation `i`.
- `plt.yscale('log')`: plots the y-axis on a logarithmic scale. This is useful when values decrease rapidly at first and then slowly — a log scale spreads out the small values so you can see the fine structure of convergence.
- A converged optimizer shows a curve that flattens out — it has stopped improving.

---

## 7. tb_optimize.py

**What it does:** Finds the 26 parameter values that make the TB model best match 8 experimental band gaps and 6 effective masses simultaneously. Uses a genetic algorithm (NSGA2) followed by gradient-based refinement.

---

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
```

**Explanation:**
- `pymoo` is a Python library for multi-objective optimization.
- **NSGA2** (Non-dominated Sorting Genetic Algorithm II): a genetic algorithm that can optimize multiple objectives at the same time. It mimics biological evolution — starts with a population of random parameter sets, selects the best ones (parents), creates new ones by combining and mutating parents (offspring), and repeats for many generations.
- `ElementwiseProblem`: a pymoo base class where you define the problem. "Elementwise" means the objective function is evaluated one parameter set at a time.

---

```python
TARGET_EG = np.array([1.65, 2.75, 3.016, 2.432, 2.206, 2.078, 1.971, 3.4])
```

**Explanation:**
- These are the 8 target band gaps in eV that the model should reproduce:
  - 1.65 eV: 3D bulk at the R-point (experimental value for MAPbI₃)
  - 2.75 eV: 3D bulk at the M-point
  - 3.016, 2.432, 2.206, 2.078, 1.971 eV: 2D slabs n=1 through 5 at the M-point
  - 3.4 eV: 3D bulk at the X-point
- All values come from published experimental or DFT literature — they are not from this project's own calculations.

---

```python
XL = np.array([-15.0, -7.0, -15.0, ...])
XU = np.array([ +0.0, +7.0,   0.0, ...])
```

**Explanation:**
- `XL` = lower bounds, `XU` = upper bounds for each of the 26 parameters.
- These define the "search space" — the region where the optimizer is allowed to look.
- The bounds come from physical reasoning: for example, `Es0` (the on-site energy of the s orbital on the metal atom) must be negative (below vacuum level) and within a physically reasonable range for lead iodide.

---

```python
N_CONSTR = 44
GAP_WINDOWS = np.array([
    [1.5, 1.75],   # eg3DR = 1.65
    ...
])
```

**Explanation:**
- The optimizer has 44 **constraints** — conditions the solution must satisfy to be considered valid.
- Example constraint: `GAP_WINDOWS[0] = [1.5, 1.75]` means the 3D R-point gap must be between 1.5 and 1.75 eV. A parameter set that gives a gap of 0.5 eV is not physically acceptable.
- In pymoo, constraints are expressed as `g ≤ 0`. For a window constraint `[lo, hi]`, this means two inequalities: `lo - gap ≤ 0` (gap must be ≥ lo) and `gap - hi ≤ 0` (gap must be ≤ hi). So each window becomes 2 constraints.
- 8 gaps × 2 = 16 gap constraints + 6 masses × 2 = 12 mass constraints + 8 dispersion/curvature constraints + 8 others = 44 total.

---

```python
class TightBindingOptimization(ElementwiseProblem):
    def _evaluate(self, x, out, *args, **kwargs):
        eg, mu, evbs, ecbs = objectivesfunctions(x)
        
        f1 = np.sum(((mu - TARGET_MU) / TARGET_MU)**2)
        f2 = np.sum(band_ordering_penalty(evbs, ecbs))
        f3 = np.sum(((eg[:6] - TARGET_EG[:6]) / TARGET_EG[:6])**2)
        f4 = np.sum(eigenvalue_spread(evbs, ecbs))
        
        out["F"] = [f1, f2, f3, f4]
        out["G"] = build_constraints(eg, mu)
```

**Explanation:**
- This class defines the optimization problem. The key method is `_evaluate`, which receives one set of 26 parameters (`x`) and must return the objective values (`F`) and constraint violations (`G`).
- **4 objectives** (things to minimize simultaneously):
  - `f1`: mean squared relative error on effective masses
  - `f2`: penalty for bands being in the wrong order
  - `f3`: mean squared relative error on band gaps
  - `f4`: penalty for eigenvalue spreading (keeping bands tight)
- `out["F"]` and `out["G"]` are pymoo's way of receiving results — you write to these dictionaries.

---

```python
sampling = np.random.uniform(problem.xl, problem.xu, (pop_size, N_VAR))
algorithm = NSGA2(pop_size=pop_size, sampling=sampling, ...)
```

**Explanation:**
- `sampling`: the initial population — 200 random parameter sets uniformly distributed within the bounds. This is "generation 0."
- Each generation, NSGA2: (1) evaluates all individuals, (2) ranks them by how well they satisfy objectives/constraints, (3) selects the best as parents, (4) creates offspring by crossover (mixing two parents) and mutation (randomly perturbing one parameter), (5) keeps the best combined population.

---

```python
with mp.Pool(processes=N_SEEDS) as pool:
    results = pool.map(run_seed, range(N_SEEDS))
```

**Explanation:**
- `mp.Pool` creates a pool of parallel worker processes.
- `pool.map(run_seed, range(N_SEEDS))`: runs `run_seed(0)`, `run_seed(1)`, ..., `run_seed(N_SEEDS-1)` simultaneously across multiple CPU cores.
- Each seed starts from a different random population. Running multiple seeds improves the chance of finding the global minimum (genetic algorithms can get stuck in local minima).

---

## 8. generate_surrogate_data.py

**What it does:** Runs the TB Fortran model on 5,000 random parameter sets to create a dataset for training the ML surrogate model.

---

```python
rng = np.random.default_rng(42)
X = rng.uniform(XL, XU, size=(N_SAMPLES, len(XL)))
```

**Explanation:**
- `np.random.default_rng(42)`: creates a random number generator with seed 42. The seed is just a starting number — using the same seed always produces the same sequence of random numbers. This makes experiments reproducible.
- `rng.uniform(XL, XU, size=(5000, 26))`: generates a 5000×26 array. Each row is one parameter set. For each parameter, the value is chosen uniformly at random between its lower bound (XL) and upper bound (XU).

---

```python
def compute_gaps(x):
    try:
        ...
        return np.array([eg3DR, eg3DM, eg2D[0], ..., eg3DX])
    except Exception:
        return None
```

**Explanation:**
- `try/except`: attempts to compute the gaps, but if anything goes wrong (e.g., the Fortran LAPACK diagonalization fails for a numerically bad parameter set), it catches the error and returns `None` instead of crashing the whole script.
- Returning `None` for failed samples is safe — we filter them out before saving.

---

```python
results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(compute_gaps)(X[i]) for i in range(N_SAMPLES)
)
```

**Explanation:**
- `joblib.Parallel`: runs tasks in parallel across CPU cores.
- `n_jobs=-1`: use all available cores.
- `delayed(compute_gaps)(X[i])`: this is joblib's syntax for "call `compute_gaps(X[i])` but don't run it yet — schedule it for parallel execution."
- The result is a list of 5,000 items, each being either a numpy array of 8 gaps or `None`.
- `verbose=10`: prints progress every 10% — useful because 5,000 calculations takes a while.

---

```python
df = pd.DataFrame(
    np.hstack([np.array(rows_x), np.array(rows_y)]),
    columns=PARAM_NAMES + GAP_NAMES,
)
df.to_csv(out_path, index=False)
```

**Explanation:**
- `np.hstack`: stacks arrays horizontally (side by side). `rows_x` has shape (N, 26) and `rows_y` has shape (N, 8), so `hstack` gives shape (N, 34).
- `pd.DataFrame`: converts the numpy array to a pandas DataFrame — a table with named columns (like an Excel spreadsheet in Python).
- `.to_csv(out_path, index=False)`: saves the table as a CSV file. `index=False` means don't save the row numbers as a column.

---

## 9. train_surrogate.py

**What it does:** Loads the 5,000-row CSV, trains a machine learning model to predict band gaps from parameters, and saves the trained model.

---

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
```

**Explanation:**
- Splits the 5,000 rows randomly into:
  - 4,000 rows for **training** (the model sees these and learns from them)
  - 1,000 rows for **testing** (the model never sees these during training — used only to measure accuracy)
- `test_size=0.2` means 20% goes to the test set.
- `random_state=42`: fixed seed for reproducibility — the same split every time.
- **Why split?** If you test the model on data it was trained on, you only measure how well it memorized the data, not how well it generalizes to new inputs. The test set simulates new, unseen parameter sets.

---

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
```

**Explanation:**
- `StandardScaler` rescales each input column to have mean=0 and standard deviation=1.
- Why? The 26 parameters have very different scales: `Es0` ranges from −15 to 0 (range of 15), while `Vss_IIz` ranges from −0.5 to +0.5 (range of 1). Without scaling, the gradient boosting algorithm might treat large-range parameters as more important just because their numbers are bigger.
- `fit_transform(X_train)`: computes the mean and std of each column from the training data, then scales it.
- `transform(X_test)`: applies the **same** scaling (using training mean and std) to the test data. Important: you must never fit the scaler on the test data — that would "leak" information.

---

```python
base = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, ...)
model = MultiOutputRegressor(base, n_jobs=-1)
model.fit(X_train_s, Y_train)
```

**Explanation:**
- **Gradient Boosting Regressor:** Builds 300 decision trees sequentially. Each tree tries to correct the errors of all previous trees. This "boosting" approach makes the combined model very accurate.
  - `n_estimators=300`: use 300 trees.
  - `max_depth=5`: each tree is at most 5 levels deep (controls complexity).
  - `learning_rate=0.05`: each new tree contributes only 5% of its prediction (small steps prevent overfitting).
  - `subsample=0.8`: each tree is trained on a random 80% of the data (reduces overfitting).
- **MultiOutputRegressor:** Since GradientBoosting predicts one number at a time, this wrapper trains 8 separate models — one for each of the 8 band gap targets.
- `.fit(X_train_s, Y_train)`: trains the model. This is where the learning happens.

---

```python
r2  = r2_score(Y_test[:, j], Y_pred[:, j])
mae = mean_absolute_error(Y_test[:, j], Y_pred[:, j]) * 1000
```

**Explanation:**
- `Y_test[:, j]`: all rows, column j — the true values for gap j from the test set.
- `Y_pred[:, j]`: the model's predictions for those same samples.
- `* 1000`: converts from eV to meV (millielectronvolts) for easier reading.

---

```python
joblib.dump({"model": model, "scaler": scaler}, model_path)
```

**Explanation:**
- Saves both the trained model and the scaler to a single file.
- Why save the scaler too? When you later load the model to make predictions, you must apply the same scaling to the inputs. The scaler contains the means and standard deviations it learned from the training data — without it, you'd get wrong predictions.
- `joblib.dump` uses Python's object serialization — it converts the Python objects to bytes and writes them to disk. `joblib.load` reverses this.

---

## 10. optuna_optimize.py

**What it does:** Uses the trained surrogate model to quickly search for parameter values that minimize the gap prediction error. Runs 3,000 "trials" — each trial is one parameter set evaluated by the surrogate.

---

```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
```

**Explanation:**
- `optuna` is a Python library for hyperparameter optimization and Bayesian search.
- `set_verbosity(WARNING)`: tells Optuna to only print warnings, not every trial's details (which would flood the screen with 3,000 lines).

---

```python
def objective(trial):
    x = np.zeros(len(PARAM_NAMES))
    for i, (name, lo, hi) in enumerate(zip(PARAM_NAMES, XL, XU)):
        if i in FIXED_IDX:
            x[i] = FIXED_IDX[i]
        else:
            x[i] = trial.suggest_float(name, lo, hi)
    
    x_s = scaler.transform(x.reshape(1, -1))
    gaps = np.array([est.predict(x_s)[0] for est in model.estimators_])
    return float(np.mean((gaps - TARGET_EG) ** 2))
```

**Explanation:**
- The objective function is what Optuna tries to minimize. It receives a `trial` object and must return a number (the "score" — lower is better).
- `trial.suggest_float(name, lo, hi)`: asks Optuna to suggest a value for this parameter. Optuna picks the value based on what it has learned from previous trials — not randomly, but guided by its internal model of where good values are likely to be.
- `FIXED_IDX`: some parameters (deltaSO0, deltaSO1) are fixed at known physical values — we don't optimize them.
- `x.reshape(1, -1)`: changes the shape from (26,) to (1, 26) because the scaler and model expect a 2D array (a table with one row).
- `[est.predict(x_s)[0] for est in model.estimators_]`: calls each of the 8 sub-models directly, avoiding joblib overhead for a single sample.
- `np.mean((gaps - TARGET_EG)**2)`: Mean Squared Error — the average squared difference between predicted and target gaps. MSE=0 means perfect match.

---

```python
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42, multivariate=False),
    storage=f"sqlite:///{db_path}",
    load_if_exists=True,
)
study.optimize(objective, n_trials=3000, show_progress_bar=True)
```

**Explanation:**
- `create_study`: creates an optimization "study" — a named collection of trials.
- `direction="minimize"`: we want to minimize the MSE (smaller = better).
- **TPESampler** (Tree-structured Parzen Estimator): Optuna's default algorithm. It maintains two probability distributions — one for parameter values with good scores, one for bad scores — and suggests values more likely to be in the "good" region.
- `multivariate=False`: treats each parameter independently (simpler, faster than modeling correlations).
- `storage=f"sqlite:///{db_path}"`: saves every trial to a SQLite database file. This means if the script is interrupted, you can resume from where you left off by running it again (because `load_if_exists=True`).
- `study.optimize(objective, n_trials=3000)`: runs 3,000 trials. Each trial calls `objective(trial)` once.

---

```python
best = study.best_trial
print(f"Best MSE: {best.value:.6f} eV²")
```

**Explanation:**
- After optimization, `study.best_trial` contains the trial with the lowest MSE.
- `best.value`: the MSE score of the best trial.
- `best.params`: a dictionary of the 26 parameter values that gave the best score.
- These parameters can then be used as a starting point for the NSGA2 optimizer (as `WARM_START`) or directly verified using the full Fortran TB model.

---

---

## 11. vqe_tb.py

**What it does:** Implements a Variational Quantum Eigensolver (VQE) to find the ground-state energy of a 4-band subspace of the TB Hamiltonian using a quantum circuit simulation (PennyLane).

---

```python
H_R = hamiltonian_cubic3D_fortran(kR, kR, kR, *PARAMS_3D)
_, evecs_R = np.linalg.eigh(H_R)
```

**Explanation:**
- Builds the full 32×32 TB Hamiltonian at k = R (the corner of the Brillouin zone, where the band gap is).
- `np.linalg.eigh`: diagonalises a Hermitian matrix — returns eigenvalues and eigenvectors. The `h` in `eigh` means Hermitian (guarantees real eigenvalues and orthogonal eigenvectors).
- `evecs_R`: a 32×32 matrix where each column is one eigenvector (one quantum state). Column 25 is the state of the top valence band; column 26 is the bottom conduction band.

---

```python
sub_idx = [IVB - 1, IVB, ICB, ICB + 1]
U4 = evecs_R[:, sub_idx]   # shape (32, 4)
```

**Explanation:**
- `sub_idx = [24, 25, 26, 27]`: the 4 bands nearest the gap — 2 from the valence band, 2 from the conduction band.
- `U4`: a 32×4 matrix. Each of its 4 columns is a 32-component quantum state. These 4 states form the "near-gap subspace" — the physics that matters most for optical and electronic properties.
- This step is necessary because a 32-qubit simulation would require 2³² ≈ 4 billion numbers. A 4-state (2-qubit) simulation needs only 4 numbers.

---

```python
H_k = hamiltonian_cubic3D_fortran(kR + DELTA_K, kR, kR, *PARAMS_3D)
H_eff = U4.conj().T @ H_k @ U4   # shape (4, 4)
```

**Explanation:**
- We evaluate the Hamiltonian at a k-point slightly away from R. At R exactly, the 4 bands are well-separated and don't mix — the problem would be trivially diagonal. Moving away from R mixes the bands, creating off-diagonal elements.
- `DELTA_K = 0.05 * π/a`: 5% of the zone boundary — a small shift.
- `U4.conj().T @ H_k @ U4`: this is a matrix projection. It computes H_eff[i,j] = ⟨φᵢ|H(k+δk)|φⱼ⟩ — the energy matrix element between eigenstates φᵢ and φⱼ in the new Hamiltonian.
- `.conj().T`: conjugate transpose (also called †, the Hermitian adjoint). Required because the eigenstates are complex-valued.
- The result is a 4×4 complex Hermitian matrix — this is what the VQE will solve.

---

```python
dev = qml.device("default.qubit", wires=2)
```

**Explanation:**
- `qml.device("default.qubit", wires=2)`: creates a simulated 2-qubit quantum computer in memory.
- `default.qubit` is PennyLane's built-in classical simulator — it doesn't run on real quantum hardware, but produces exactly the same results (no noise).
- `wires=2`: 2 qubits. A 2-qubit system has 2² = 4 basis states: |00⟩, |01⟩, |10⟩, |11⟩ — matching our 4×4 H_eff.

---

```python
@qml.qnode(dev)
def circuit(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)
    qml.CNOT(wires=[1, 0])
    qml.RY(params[4], wires=0)
    qml.RY(params[5], wires=1)
    return qml.expval(qml.Hermitian(H_eff, wires=[0, 1]))
```

**Explanation:**
- `@qml.qnode(dev)`: this decorator turns the function into a "quantum node" — a function that runs on the quantum device `dev`. Every time you call `circuit(params)`, it runs the quantum gates and returns a measurement.
- **RY gate**: rotates a qubit around the Y-axis of the Bloch sphere by an angle `params[i]`. This is the quantum equivalent of a trainable weight.
- **CNOT gate**: a 2-qubit gate — flips qubit 1 if qubit 0 is |1⟩, otherwise does nothing. This creates **entanglement** — correlations between the two qubits that can't be represented by two independent qubits. Entanglement is essential for representing complex quantum states.
- The pattern (RY, RY, CNOT) repeated twice is a "hardware-efficient ansatz" — a parameterised circuit that is both expressive (can represent many states) and practical (uses only common gates).
- `qml.Hermitian(H_eff, wires=[0, 1])`: this is the observable — it tells PennyLane "measure the expected energy of the circuit state under H_eff."
- `qml.expval(...)`: returns the expected value ⟨ψ(θ)|H_eff|ψ(θ)⟩ — the average energy if you ran this circuit many times and measured.

---

```python
opt = qml.AdamOptimizer(stepsize=STEPSIZE)
for step in range(N_STEPS):
    params, energy = opt.step_and_cost(circuit, params)
```

**Explanation:**
- `qml.AdamOptimizer`: PennyLane's built-in Adam optimizer — the same algorithm used to train neural networks.
- `opt.step_and_cost(circuit, params)`: computes the gradient of `circuit(params)` with respect to `params` (using the **parameter-shift rule** — a quantum-specific version of backpropagation), takes one Adam step, and returns the updated params and the current energy.
- The gradient is exact (not numerical approximation): PennyLane runs the circuit twice with ±π/2 shifts for each parameter, and the gradient is the difference. This is called the parameter-shift rule.
- After 300 steps, the energy converges to the exact ground state of H_eff.

---

```python
E_exact = np.linalg.eigvalsh(H_eff)
```

**Explanation:**
- `eigvalsh`: computes eigenvalues of a Hermitian matrix — this is the classical exact solution.
- The VQE result should match `E_exact[0]` (the lowest eigenvalue) to within < 1 meV after convergence.
- This comparison is the key validation: the quantum algorithm gives the same answer as classical exact diagonalization.

---

## Summary: How all the scripts connect

```
Fortran (.f90 files)
    ↓ compiled by gfortran
libhamiltonian_3rd.so
    ↓ loaded by
ctypesroutines.py  ←──────────────────────────────────────────┐
    ↓ used by                                                   │
absorption_2D_fortran.py  → results/figures/absorption*.png    │
effective_mass_tensor.py  → results/figures/eff_mass*.png      │
exciton_binding.py        → results/figures/exciton*.png       │
comparison_table.py       → results/figures/comparison*.png    │
convergence_plot.py       → results/figures/convergence*.png   │
tb_optimize.py            → finds best 26 parameters           │
    ↓                                                           │
generate_surrogate_data.py → data/surrogate_dataset.csv        │
    ↓                                                           │
train_surrogate.py → data/surrogate_model.joblib               │
    ↓                                                           │
optuna_optimize.py → best parameters (fast, 3000 trials ~1 min)│
    ↓                                                           │
vqe_tb.py → quantum ground-state energy ←─────────────────────┘
    └→ verify vs ctypesroutines.py (numpy exact diag)
```

Every script in the left column uses the Fortran model via `ctypesroutines.py`. The Phase 2 scripts form a self-contained ML pipeline. Phase 3 (vqe_tb.py) adds a quantum algorithm that cross-checks with the same Fortran Hamiltonian.

---

## Phase 5 Background: What is an exciton, and why is it special in 2D?

Before reading the Phase 5 scripts, it helps to understand the core physics they calculate.

### What is an exciton?

When light hits a semiconductor, it can give an electron enough energy to jump from the valence band (where electrons are normally found) to the conduction band (where electrons can move freely). This is the process that makes solar cells and LEDs work.

But there is a subtlety: when the electron leaves, it leaves behind a positively charged "hole" — an absence of electron. The electron and hole are attracted to each other by the electrostatic force between opposite charges (the same force that holds hydrogen atoms together). If the attraction is strong enough, the electron and hole form a bound pair called an **exciton**.

Think of an exciton like a mini hydrogen atom floating inside the crystal: a negative electron orbiting a positive hole. The energy it takes to break apart this pair and create truly free carriers is the **exciton binding energy** E_B (measured in meV).

### Why are 2D perovskites special?

In a standard 3D semiconductor (like silicon), the electric field between the electron and hole is uniformly screened by the surrounding dielectric material. The exciton binding energy is typically small (a few meV), so excitons fall apart at room temperature.

In a 2D perovskite slab, the situation is very different:

1. **Geometric confinement:** The electron and hole are squeezed into a thin layer (as thin as 0.6 nm for n=1). A confined particle has less room to escape, which makes the bound state tighter and increases E_B.

2. **Dielectric contrast:** The thin inorganic slab (high dielectric constant ε ≈ 6) is sandwiched between organic layers (low dielectric constant ε ≈ 2). When the electric field between electron and hole extends outside the slab, it passes through the low-ε organic layer. Low ε means *less* screening, which means *stronger* attraction, which means *larger* E_B.

The result: exciton binding energies of 125–470 meV in 2D perovskites, compared to ~4 meV in GaAs. Excitons survive at room temperature and dominate the optical properties.

### The Rytova-Keldysh potential

In 3D, the electron-hole interaction is just the Coulomb potential: V(r) = -e²/(4πε₀ εr). In a 2D slab with dielectric contrast, the exact potential was worked out by Rytova (1967) and Keldysh (1979) and re-derived for atomically thin materials by Chernikov et al. (2014). In real space it reads:

```
V_RK(r) = -(π e²/4πε₀) / (2 ε_env r₀) × [H₀(r/r₀) − Y₀(r/r₀)]
```

where:
- **r** is the electron-hole separation
- **ε_env** is the dielectric constant of the surrounding (organic) environment
- **r₀ = ε_slab × d / (2 ε_env)** is the "screening length" — proportional to the slab thickness d. Thicker slabs screen more, so E_B decreases with n.
- **H₀** is the Struve function (related to a Bessel function), **Y₀** is the Neumann function (Bessel function of the second kind, order 0). Together H₀ − Y₀ describes the spatial profile of the screened potential.

Limits: at large r, V_RK → −e²/(4πε₀ ε_env r) (normal 2D Coulomb, screened only by the environment). At small r, V_RK → −e²/(4πε₀ ε_env r₀) × const (the short-range potential is softer than 1/r because the slab screens the interaction at distances smaller than r₀).

In k-space (Fourier transform), the same potential becomes much simpler:

```
V_RK(q) = −π e²/(4πε₀) / [ε_env q (1 + r₀ q)]    [eV·nm²]
```

This form is used in the BSE (Phase 5 Level-2). At small q (long wavelengths) it diverges as 1/q — the same as the 2D Coulomb potential. At large q it falls as 1/q² (screened). This 1/q behavior at small q is important: it makes the BSE integral logarithmically divergent, so a fine k-grid is needed for convergence.

### Two levels of calculation

Phase 5 has two levels of increasing sophistication:

**Level 1 (rytova_keldysh.py):** Uses the effective-mass approximation (EMA). The electron and hole are treated as if they have a single effective mass μ (from experiment). The Schrödinger equation with V_RK(r) is solved directly in real space on a 1D radial grid — equivalent to solving a hydrogen atom with a modified potential.

**Level 2 (bse_2d.py):** Uses the full tight-binding (TB) band structure. Instead of a single scalar mass μ, the electron and hole have momentum-dependent energies Ec(k) and Ev(k) computed by the TB Hamiltonian. The Schrödinger equation becomes a matrix equation in k-space (the Bethe-Salpeter Equation), where each row/column corresponds to one k-point and the interaction is V_RK(|k−k'|) times an overlap factor (form factor) between the TB wavefunctions at k and k'.

---

## 12. rytova_keldysh.py

**What it does:** Computes the exciton binding energy for each slab n=1–5 by solving the 2D radial Schrödinger equation with the Rytova-Keldysh potential. This is Phase 5 Level-1 — no k-space matrix, just a one-dimensional eigenvalue problem on a real-space grid.

---

```python
HBAR2_OVER_2ME = 0.038099821   # ℏ²/(2mₑ)  [eV·nm²]
E2_OVER_4PIEPS0 = 1.4399644    # e²/(4πε₀) [eV·nm]
```

**Explanation:**
- These are universal physical constants expressed in the units used throughout: energies in eV, lengths in nm.
- `HBAR2_OVER_2ME = ℏ²/(2mₑ)`: this combination appears in the kinetic energy term of the Schrödinger equation. Explicitly: ℏ = 1.0546×10⁻³⁴ J·s, mₑ = 9.109×10⁻³¹ kg. Converting to eV·nm² gives 0.03810. When we divide by the reduced mass ratio μ/mₑ, we get ℏ²/(2μ).
- `E2_OVER_4PIEPS0 = e²/(4πε₀)`: the strength of the Coulomb force, in eV·nm. The full Coulomb potential between charges ±e separated by r nm is −E2_OVER_4PIEPS0 / r eV.

---

```python
EPS_SLAB = 6.1
EPS_ENV  = 2.1
D_NM   = {1: 0.629, 2: 1.258, 3: 1.887, 4: 2.516, 5: 3.145}
MU_M0  = {1: 0.221, 2: 0.210, 3: 0.200, 4: 0.195, 5: 0.186}
EL_EXP = {1: 470,   2: 300,   3: 200,   4: 160,   5: 125}
```

**Explanation:**
- `EPS_SLAB = 6.1`: the dielectric constant of the inorganic PbI₄ layer. A dielectric constant of 6 means electric fields inside the material are weakened by a factor of 6 compared to vacuum. Source: Blancon et al. (2018) supplementary material.
- `EPS_ENV = 2.1`: the dielectric constant of the organic BA spacer. Its small value (close to 1 = vacuum) means the spacer barely screens the electric field, making the exciton tighter.
- `D_NM`: slab thicknesses in nanometres. The single inorganic layer (n=1) is 0.629 nm thick. Each additional layer adds 0.629 nm.
- `MU_M0`: reduced mass μ/mₑ for each n, measured experimentally in 60-Tesla pulsed magneto-optics (Blancon 2018). Reduced mass μ = (mₑ × mₕ) / (mₑ + mₕ) is the effective inertia of the electron-hole pair relative motion.
- `EL_EXP`: experimental binding energies in meV — the numbers this calculation is trying to reproduce.

---

```python
def r0_nm(n, eps_slab=EPS_SLAB, eps_env=EPS_ENV):
    return eps_slab * D_NM[n] / (2.0 * eps_env)
```

**Explanation:**
- The screening length r₀ = ε_slab × d / (2 ε_env).
- Physically: r₀ is the length scale at which the dielectric screening "turns on." For distances r ≪ r₀ the electron sees the slab screening (ε_slab). For r ≫ r₀ the field has leaked into the environment (ε_env).
- Larger n → thicker slab → larger r₀ → more screening → smaller binding energy. This formula encodes the key physics of why E_B decreases with n.
- For n=1: r₀ = 6.1 × 0.629 / (2 × 2.1) = 0.91 nm.

---

```python
def V_RK_q(q_nm, r0, eps_env=EPS_ENV):
    q = np.asarray(q_nm, dtype=float)
    safe = q > 1e-12
    out  = np.zeros_like(q)
    out[safe] = (-np.pi * E2_OVER_4PIEPS0
                 / (eps_env * q[safe] * (1.0 + r0 * q[safe])))
    out[~safe] = -np.inf
    return out
```

**Explanation:**
- This computes the R-K potential in k-space: V_RK(q) = −π e²/(4πε₀) / [ε_env q (1 + r₀ q)], result in eV·nm².
- `safe = q > 1e-12`: guards against division by zero at q = 0. At q = 0 the potential diverges to −∞ (the 2D Fourier transform of any attractive potential diverges at q=0 — it represents the average value of the potential, which is the total attractive energy). We set these entries to −inf explicitly.
- The formula is separating: at large q (short distances), the denominator gains a factor r₀ q so V_RK ∝ 1/q², meaning the slab screens away the short-distance interaction. At small q (long distances), V_RK ≈ −π e²/(ε_env q), which is the standard unscreened 2D Coulomb.
- This function is imported and used by `bse_2d.py` (Level-2).

---

```python
def V_RK_r(r_nm, r0, eps_env=EPS_ENV):
    r = np.asarray(r_nm, dtype=float)
    x = r / r0
    H0 = sp.struve(0, x)
    Y0 = sp.y0(x)
    return -E2_OVER_4PIEPS0 * (np.pi / (2.0 * eps_env * r0)) * (H0 - Y0)
```

**Explanation:**
- This computes the R-K potential in real space (Eq. 2 of Chernikov 2014).
- `x = r / r0`: the key point — the argument of the Struve and Neumann functions is r/r₀ (not ε_env × r/r₀, which was a bug in an earlier version). This is a critical detail: the correct formula came from carefully converting the original SI expression.
- `sp.struve(0, x)`: the Struve function H₀(x), from `scipy.special`. It is a close relative of the Bessel function J₀, but doesn't oscillate — it grows logarithmically for small x.
- `sp.y0(x)`: the Neumann function Y₀(x) (Bessel function of the second kind). It also diverges logarithmically at x=0.
- The combination H₀(x) − Y₀(x): both functions diverge as log(x) at small x, but their difference is finite — the logarithmic divergences cancel. This gives a potential that is softer than 1/r at short distances, which is physically correct for a thin slab (there is no point charge, just a smeared-out 2D distribution).

---

```python
def solve_binding_energy(n, r_max=120.0, Nr=4000):
    r0        = r0_nm(n)
    hbar2_2mu = HBAR2_OVER_2ME / MU_M0[n]
    dr        = r_max / (Nr + 1)
    r         = np.arange(1, Nr + 1) * dr
```

**Explanation:**
- `r_max=120.0 nm`: the grid extends to 120 nm — large enough that the wavefunction has decayed to zero at the boundary.
- `Nr=4000`: 4000 grid points. With dr = 120/4001 ≈ 0.030 nm, this is dense enough to resolve the wavefunction near r=0 where the potential changes quickly.
- `hbar2_2mu = HBAR2_OVER_2ME / MU_M0[n]`: ℏ²/(2μ) in eV·nm². This is the "stiffness" — larger reduced mass → smaller ℏ²/(2μ) → the electron-hole pair is "heavier" and more easily bound.
- `r = np.arange(1, Nr+1) * dr`: creates the grid [dr, 2dr, 3dr, ..., Nr×dr]. We start from dr (not zero) because V_RK(0) diverges. The boundary condition g(0) = 0 is imposed implicitly by not including r=0.

---

```python
    V_eff = V_RK_r(r, r0) + hbar2_2mu / (4.0 * r**2)
```

**Explanation:**
- This is the key step: the effective potential in the transformed equation.
- **Why add hbar2_2mu/(4r²)?** The 2D radial Schrödinger equation for the m=0 (lowest angular momentum) state is:
  ```
  −ℏ²/(2μ) × [ψ'' + (1/r)ψ'] + V_RK(r) ψ = E ψ
  ```
  The `(1/r)ψ'` term makes this awkward to solve numerically (it causes a singularity at r=0). The standard trick is to substitute `g(r) = √r × ψ(r)`. Then ψ = g/√r, and after substitution, the equation transforms to:
  ```
  −ℏ²/(2μ) g'' + [V_RK(r) + ℏ²/(8μr²)] g = E g
  ```
  Note: ℏ²/(8μr²) = hbar2_2mu / (4r²) since hbar2_2mu = ℏ²/(2μ).
- This transformed equation has no first derivative — it looks like a standard 1D Schrödinger equation in a potential V_eff = V_RK + ℏ²/(8μr²). The extra term ℏ²/(8μr²) is the "centrifugal barrier" — it pushes the wavefunction away from r=0, which is correct physical behavior for the 2D m=0 state.
- The boundary conditions are simply g(0) = g(r_max) = 0, which is easy to enforce on a grid.

---

```python
    hopping   = -hbar2_2mu / dr**2
    main_diag = -2.0 * hopping + V_eff
    H = diags([np.full(Nr-1, hopping), main_diag, np.full(Nr-1, hopping)],
              [-1, 0, 1], shape=(Nr, Nr), format='csr')
```

**Explanation:**
- **Finite differences:** The second derivative g''(rᵢ) ≈ [g(rᵢ₊₁) − 2g(rᵢ) + g(rᵢ₋₁)] / dr². This turns the differential equation into a matrix equation: Hg = Eg.
- The kinetic energy operator −ℏ²/(2μ) g'' becomes a tridiagonal matrix: the diagonal is −ℏ²/(2μ) × (−2/dr²) = +2×hbar2_2mu/dr², and the off-diagonals are −ℏ²/(2μ) × (1/dr²) = hopping. The sign of hopping is negative.
- `main_diag = -2.0 * hopping + V_eff`: diagonal = kinetic + potential = 2×hbar2_2mu/dr² + V_eff.
- `diags(...)`: creates a **sparse** tridiagonal matrix — only the non-zero elements are stored. With Nr=4000, a full matrix would be 4000×4000 = 16 million complex numbers; the sparse format stores only 3×4000 = 12,000 numbers.
- `format='csr'`: Compressed Sparse Row — the best format for matrix-vector multiplications, which is what the eigenvalue solver does repeatedly.

---

```python
    eigenvalues, _ = eigsh(H, k=3, which='SA')
    eigenvalues = np.sort(eigenvalues)
    binding_meV = -eigenvalues[0] * 1000.0
    return binding_meV, eigenvalues
```

**Explanation:**
- `eigsh`: scipy's sparse Hermitian eigenvalue solver. It uses the Lanczos algorithm — an iterative method that finds a few eigenvalues without storing or computing the full matrix.
- `k=3`: find the 3 lowest eigenvalues.
- `which='SA'` (Smallest Algebraic): find the eigenvalues with the smallest numerical value. Since the ground state has the most negative energy, this finds the bound states.
- The eigenvalues are in eV. A bound state has negative energy (the electron-hole pair is more stable than two free particles). The ground state energy E₁ < 0 gives the binding energy:
  - `E_B = −E₁ × 1000` meV (the minus sign makes it a positive number).

---

## 13. bse_2d.py

**What it does:** Phase 5 Level-2. Computes exciton binding energies using the Bethe-Salpeter Equation (BSE) in k-space. Instead of using a single effective mass μ, it uses the full TB band structure: a 2D grid of k-points around the band edge, and an interaction matrix V_RK(|k−k'|) weighted by the overlap between the TB wavefunctions (the "form factor"). This is a many-body calculation: every k-point is coupled to every other k-point by the potential.

---

### Why not just use Level-1?

Level-1 uses the effective mass μ — a single number that approximates how the TB bands curve near the M-point. This is convenient but approximate. The TB bands are not perfect parabolas; they flatten, split due to spin-orbit coupling, and mix with other bands away from the M-point. Level-2 captures all of this: the band energies Ec(k) and Ev(k) at each k-point are computed exactly by diagonalizing the full 26-parameter TB Hamiltonian.

---

### What is the Bethe-Salpeter Equation?

The BSE is a quantum many-body equation for the electron-hole pair. In k-space it looks like this:

```
H_exc × A = Ω × A

H_exc[k, k'] =  (Ec(k) − Ev(k)) × δ(k,k')        [diagonal: free e-h pair energy]
              + V_RK(|k−k'|) × F(k,k') × (dk)²/(2π)²  [off-diagonal: Coulomb interaction]
```

Here A is a vector (one component per k-point) representing how each k-point contributes to the exciton wavefunction, and Ω is the excitation energy.

- The **diagonal** gives the energy of a free electron at k in the conduction band minus a free hole at k in the valence band.
- The **off-diagonal** terms couple k to k': the R-K potential scatters the electron (or hole) from k to k'. The **form factor** F(k,k') = |⟨c,k|c,k'⟩| × |⟨v,k|v,k'⟩| accounts for the fact that the TB wavefunctions at k and k' are not identical (they rotate in the spinor space as k changes).
- `(dk)²/(2π)²` is the phase-space factor converting the potential from eV·nm² to eV (it is the area element in k-space).

---

```python
A_LAT = 6.28          # in-plane lattice constant [Å]
KM    = np.pi / A_LAT  # M-point kx = ky [Å⁻¹]

IVB = {1: 33, 2: 59, 3: 85, 4: 111, 5: 137}
ICB = {1: 34, 2: 60, 3: 86, 4: 112, 5: 138}
```

**Explanation:**
- `A_LAT = 6.28 Å`: the in-plane lattice constant of the perovskite crystal. The Brillouin zone runs from −π/a to +π/a in each direction.
- `KM = π/A_LAT ≈ 0.5 Å⁻¹`: the k-coordinate of the M-point (corner of the square Brillouin zone). The band gap in these 2D perovskites occurs at M, not at Γ. This is a key difference from many common semiconductors (like GaAs which has its gap at Γ). Working at M instead of Γ means all k-grids must be centered at (KM, KM), not at (0, 0).
- `IVB, ICB`: the 0-based indices of the valence band maximum (VBM) and conduction band minimum (CBM) at M for each slab thickness n. These come from manually inspecting the eigenvalue spectrum at M. For n=1: the TB Hamiltonian is 32×8 = (32×1 + 8) = 40 × 40 ... wait let me re-read: n_orb = 32*n + 8. For n=1, n_orb = 40. VBM is band index 33 (0-based), CBM is 34.

---

```python
def _ham(n_layers, kx, ky):
    return hamiltonian_cubic2D_apical_fortran(
        n_layers, kx, ky, 0.0,
        Es0, Ep0, Es1, Ep1, Vss, Vsp, Vps, Vppsig, Vpppi,
        deltaSO0, deltaSO1,
        Vss_II, Vsp_II, Vppsig_II, Vpppi_II,
        Vss_IIz, Vsp_IIz, Vppsig_IIz, Vpppi_IIz,
        Vss_PbPb, Vsp_PbPb, Vppsig_PbPb, Vpppi_PbPb,
        Es1_ap, Ep1_ap, Ep0_surf,
    )
```

**Explanation:**
- A thin wrapper that calls the Fortran function `hamiltonian_cubic2D_apical_fortran` with the 26 best-fit parameters.
- `hamiltonian_cubic2D_apical_fortran` (not `hamiltonian_cubic2D_general_fortran`): this is the version that includes the "apical correction" — a modification to the surface iodine atoms at the top and bottom of the slab. These apical atoms sit in a different chemical environment than the bulk iodines, so they have different on-site energies (Es1_ap, Ep1_ap, Ep0_surf). Using the wrong Hamiltonian function produces wrong band gaps.
- `kz=0.0`: 2D slabs have no periodicity in z, so kz is always zero.

---

```python
    ik_center = Nk2 // 2
    ham_M = _ham(n, kx_flat[ik_center], ky_flat[ik_center])
    evals_M, evecs_M = diagonalize_hamiltonian_fortran(ham_M)
    psi_c_ref = evecs_M[:, icb].copy()
    psi_v_ref = evecs_M[:, ivb].copy()
```

**Explanation:**
- `ik_center = Nk²/2`: the index of the grid centre, which is the exact M-point.
- We diagonalize the TB Hamiltonian at M to get the reference wavefunctions: `psi_c_ref` (the conduction band spinor) and `psi_v_ref` (the valence band spinor).
- These are complex vectors of length n_orb = 32n + 8 — the quantum state of the electron in the CB (or hole in the VB) at M.
- We will use these as references to identify the CB and VB at other k-points.

---

```python
    for ik in range(Nk2):
        if ik == ik_center:
            continue
        ham = _ham(n, kx_flat[ik], ky_flat[ik])
        evals, evecs = diagonalize_hamiltonian_fortran(ham)
        olap_c = np.abs(psi_c_ref.conj() @ evecs)
        ic_k   = np.argmax(olap_c)
        olap_v = np.abs(psi_v_ref.conj() @ evecs)
        iv_k   = np.argmax(olap_v)
        Ec[ik] = evals[ic_k]
        Ev[ik] = evals[iv_k]
        psi_c[ik] = evecs[:, ic_k]
        psi_v[ik] = evecs[:, iv_k]
```

**Explanation:**
- This loop is the most critical part of the code. It solves the TB Hamiltonian at every k-point and identifies the correct CB and VB bands using **band tracking by wavefunction overlap**.
- **Why is tracking necessary?** The bands are labeled by their energies — band 33 at M is the VBM because it has the highest energy among occupied bands *at M*. But at k-points away from M, the energy order can change: two bands can cross (their energies become equal and then swap). If this happens, "band 33" at k≠M might actually be the conduction band — the label no longer identifies the physical band we want.
- **The overlap trick:** Instead of using the index, we ask: which band at k has the largest wavefunction overlap with the VBM wavefunction at M? If the bands vary smoothly, the VBM at k is the one most similar (highest overlap) to the VBM at M. `psi_c_ref.conj() @ evecs` computes the overlap of `psi_c_ref` with every column of `evecs`. `argmax` finds the column with the largest overlap.
- Without this trick, the form factors F[k,k'] come out near zero for k≠M (because fixed-index states are orthogonal after a band crossing), giving a near-zero BSE interaction and a near-zero binding energy — the code silently produces a wrong answer of ~0 meV.

---

```python
    S_c = psi_c.conj() @ psi_c.T
    S_v = psi_v.conj() @ psi_v.T
    F   = np.abs(S_c) * np.abs(S_v)
```

**Explanation:**
- `S_c[ik, ik'] = ⟨c,k|c,k'⟩`: overlap of the conduction-band wavefunction at k with that at k'. This is an Nk²×Nk² complex matrix.
- `S_v[ik, ik'] = ⟨v,k|v,k'⟩`: same for the valence band.
- `F = |S_c| × |S_v|`: taking absolute values makes F real and non-negative. On the diagonal (k=k'), F[k,k] = |⟨c,k|c,k⟩|² × |⟨v,k|v,k⟩|² = 1×1 = 1 (each wavefunction has norm 1). Off-diagonal, F decreases as k and k' become more different.
- **Why intraband and not interband?** The BSE direct kernel physically describes the electron-hole Coulomb interaction, which conserves the band character of each particle separately — the electron stays in the CB while the hole stays in the VB. The correct overlap factor is therefore ⟨c,k|c,k'⟩ × ⟨v,k'|v,k⟩ (each band with itself). The interband overlap ⟨c,k|v,k'⟩ is the optical matrix element — it describes light absorption, not the Coulomb interaction. At k=k', ⟨c|v⟩=0 because CB and VB are orthogonal, so the interband form factor would be zero on the diagonal, giving no BSE interaction at all.

---

```python
    q_nm   = np.sqrt(dkx_nm**2 + dky_nm**2)
    Vrk    = V_RK_q(q_nm, r0)
    np.fill_diagonal(Vrk, 0.0)
    scale  = dk_step_nm**2 / (2.0 * np.pi)**2
    H_bse += Vrk * F * scale
```

**Explanation:**
- `q_nm`: the momentum transfer |k−k'| in nm⁻¹. This is the q in V_RK(q). `dkx_nm` and `dky_nm` are Nk²×Nk² matrices of k-point separations.
- `Vrk = V_RK_q(q_nm, r0)`: an Nk²×Nk² matrix where entry [ik, ik'] is the R-K potential at momentum transfer q = |k−k'|. The values are negative (attractive).
- `np.fill_diagonal(Vrk, 0.0)`: the q=0 term (k=k') is the "self-interaction" of the electron with itself. This is physically excluded from the Coulomb interaction. Mathematically, V_RK(q=0) = −∞ (the integral diverges), so we set it to zero.
- `scale = dk²/(2π)²`: the phase-space factor. In the BSE, the sum over k-points approximates an integral: Σ_k → ∫ d²k/(2π)². The factor (2π)² comes from the 2D Fourier convention. `dk_step_nm` is the grid spacing in nm⁻¹.
- `H_bse += Vrk * F * scale`: adds the interaction matrix to the diagonal free-pair Hamiltonian. Since Vrk < 0, this lowers all eigenvalues — bound states appear below the free-pair threshold (E_gap).

---

```python
    omega = evals_bse - E_gap
    E_bind_meV = -omega[0] * 1000.0
    osc = np.abs(evecs_bse.sum(axis=0))**2
```

**Explanation:**
- `omega = evals_bse - E_gap`: excitation energies relative to the free-pair threshold. Bound states have omega < 0 (below the gap).
- `E_bind_meV = −omega[0] × 1000`: the binding energy is the magnitude of the lowest (most negative) eigenvalue, converted to meV.
- `osc = |Σ_k A_s(k)|²`: the oscillator strength of exciton state s. `evecs_bse[:, s]` is the exciton wavefunction in k-space — it gives the amplitude A_s(k) for each k-point. Summing over all k gives the total "brightness" of the exciton: how strongly it couples to light. The 1s ground state (A_1(k) = positive everywhere, no nodes) has the largest oscillator strength. Higher states have nodes (positive and negative regions cancel), making them optically dark.

---

## 14. absorption_elliott.py

**What it does:** Computes the optical absorption spectrum α(E) from the BSE exciton energies and oscillator strengths. The spectrum consists of two parts: (1) sharp Lorentzian peaks from the bound excitons below the band gap, and (2) a step-like continuum above the band gap enhanced by the Sommerfeld factor.

---

### The Elliott formula

In a 2D system, the optical absorption near the band edge follows the Elliott formula (1957):

```
α(E) ∝  Σ_s f_s × L(E − Ω_s, Γ)          [exciton peaks below gap]
       + θ(E − E_gap) × S_2D(E, E_B)        [free-carrier continuum above gap]
```

where:
- **Ω_s = E_gap + ω_s** is the energy of exciton state s (ω_s < 0 for bound states)
- **f_s** is the oscillator strength from the BSE calculation
- **L(E − Ω_s, Γ)** is a Lorentzian with width Γ (broadening due to phonons, disorder)
- **θ(E − E_gap)** is the step function (1 above gap, 0 below)
- **S_2D(E, E_B)** is the 2D Sommerfeld factor — a quantum mechanical enhancement of the free-carrier absorption due to residual Coulomb attraction above the gap

The Sommerfeld factor is: S(E) = 2πξ / (1 − exp(−2πξ)), where ξ = √(E_B / (E − E_gap)).

- Far above the gap (E ≫ E_gap + E_B): ξ → 0, S → 1 (no enhancement, normal 2D absorption)
- Just above the gap (E → E_gap⁺): ξ → ∞, S → 2πξ → ∞ (divergent — the free-carrier states are strongly modified by Coulomb attraction, approaching the exciton poles)
- This divergence is integrable — when the exciton peaks and the continuum are plotted together, the Sommerfeld enhancement connects smoothly to the exciton ladder

---

```python
def lorentzian(E, E0, Gamma):
    return (Gamma / np.pi) / ((E - E0)**2 + Gamma**2)
```

**Explanation:**
- A Lorentzian (also called a Cauchy distribution) is a bell-shaped curve that decays as 1/(E−E₀)² far from the center. It models the natural linewidth of a quantum transition: an excited state with lifetime τ has an energy uncertainty ΔE = ℏ/τ, giving a Lorentzian profile with width Γ = ℏ/τ.
- The `Gamma/pi` prefactor normalizes the function: ∫ L(E) dE = 1. So multiplying by the oscillator strength f preserves the total spectral weight.
- Typical broadening: Γ = 20 meV = 0.020 eV (about 160 cm⁻¹ or 50 ps lifetime).

---

```python
def sommerfeld_2d(E, E_gap, E_B):
    out = np.ones_like(E)
    above = E > E_gap
    xi = np.sqrt(np.maximum(E_B / (E[above] - E_gap), 0.0))
    twopix = 2.0 * np.pi * xi
    safe = twopix < 700.0
    S = np.where(safe, twopix / (1.0 - np.exp(-twopix)), twopix)
    out[above] = S
    return out
```

**Explanation:**
- `above = E > E_gap`: the Sommerfeld factor only applies above the gap (free carriers).
- `xi = √(E_B / (E − E_gap))`: the dimensionless parameter. Close to the gap edge, E − E_gap is small and ξ is large. Far above the gap, ξ → 0.
- `twopix = 2πξ`: the argument of the formula S = 2πξ / (1 − exp(−2πξ)).
- `safe = twopix < 700.0`: if twopix is very large (near the gap edge), exp(−twopix) underflows to zero on a computer, so the denominator becomes 1 − 0 = 1, and S = twopix. The `np.where` handles both cases: for moderate ξ, use the full formula; for large ξ (very close to gap), use the asymptotic S ≈ 2πξ.
- `out = np.ones_like(E)`: below the gap, the Sommerfeld factor is 1 (no enhancement; no free carriers).

---

```python
def absorption_spectrum(bse_result, Gamma_meV=20.0, n_E=2000):
    E_gap = bse_result["E_gap"]
    omega = bse_result["omega"]
    osc   = bse_result["osc"]
    E_B   = bse_result["E_bind_meV"] / 1000.0
    Gamma = Gamma_meV / 1000.0

    E_lo = E_gap - 4.0 * E_B
    E_hi = E_gap + 3.0 * E_B
    E    = np.linspace(E_lo, E_hi, n_E)
```

**Explanation:**
- The function takes the dictionary returned by `bse_2d.solve_bse` and computes the spectrum.
- `E_B`: binding energy in eV (BSE result in meV divided by 1000).
- `Gamma`: broadening in eV (input in meV divided by 1000).
- `E_lo, E_hi`: the energy window is set relative to E_B. We go 4×E_B below the gap (to capture the exciton peaks, which sit at approximately −E_B, −E_B/4, ... below the gap following the Rydberg series) and 3×E_B above the gap (to see the Sommerfeld-enhanced continuum).

---

```python
    alpha_exc = np.zeros(n_E)
    for s, (om, f) in enumerate(zip(omega, osc)):
        if om < 0.5 * E_B:
            E_s = E_gap + om
            alpha_exc += f * lorentzian(E, E_s, Gamma)
    peak = alpha_exc.max()
    if peak > 0:
        alpha_exc /= peak
```

**Explanation:**
- `om < 0.5 * E_B`: we only include states that are clearly below the gap (bound excitons). `om` is the excitation energy relative to E_gap — bound states have om < 0. The condition `om < 0.5 * E_B` means "at least 0.5 × E_B below the gap" — this excludes resonances that barely qualify as bound.
- `E_s = E_gap + om`: absolute energy of exciton state s. Since om < 0, this is below E_gap.
- `alpha_exc += f * lorentzian(E, E_s, Gamma)`: each exciton state contributes a Lorentzian peak at energy E_s, with area proportional to its oscillator strength f.
- `alpha_exc /= peak`: normalize the maximum exciton peak to height 1. This is arbitrary normalization for plotting — the absolute scale of absorption depends on the sample thickness and other factors not computed here.

---

```python
    alpha_cont = np.where(E >= E_gap, 1.0, 0.0) * sommerfeld_2d(E, E_gap, E_B)
    cont_far   = alpha_cont[E > E_gap + 0.5 * E_B]
    if cont_far.size > 0 and cont_far.max() > 0:
        alpha_cont /= cont_far.max()
    return E, alpha_exc, alpha_cont
```

**Explanation:**
- `np.where(E >= E_gap, 1.0, 0.0)`: a step function — 1 above the gap, 0 below. This is the 2D joint density of states for parabolic bands (constant above threshold).
- `* sommerfeld_2d(...)`: multiplies by the Sommerfeld enhancement. At E slightly above E_gap, this gives a large value (strong Coulomb enhancement). Far above E_gap, it approaches 1.
- `cont_far = alpha_cont[E > E_gap + 0.5 * E_B]`: the continuum far above the gap (where Sommerfeld ≈ 1). We normalize by this value so the continuum settles to ~1 far above the gap.
- The function returns three arrays: energy axis E, exciton contribution alpha_exc, and free-carrier contribution alpha_cont. The total absorption is alpha_exc + alpha_cont.

---

```python
def plot_all(Nk=21, Gamma_meV=20.0, savepath=None):
    fig, axes = plt.subplots(5, 1, figsize=(7, 12), sharex=False)
    for i, n in enumerate(range(1, 6)):
        res = solve_bse(n, Nk=Nk)
        E, alpha_exc, alpha_cont = absorption_spectrum(res, Gamma_meV=Gamma_meV)
        E_rel = (E - res["E_gap"]) * 1000   # meV relative to gap
        ...
        E_b_str = f"E_B={res['E_bind_meV']:.0f} meV"
        ax.text(0.97, 0.85, E_b_str, transform=ax.transAxes, ha="right", fontsize=9)
```

**Explanation:**
- `sharex=False`: each panel covers a different energy window (because E_B varies with n, so 4×E_B spans different ranges). Having separate x-axes makes each panel use its own optimal range.
- `E_rel = (E − E_gap) × 1000`: converts the x-axis to meV relative to the gap edge. This centers the plot on the gap edge (x=0) and shows the exciton peaks at negative x and the continuum at positive x.
- The text annotation shows the binding energy in each panel's top-right corner for quick reading.

---

## Summary: How all the scripts connect (updated with Phase 5)

```
Fortran (.f90 files)
    ↓ compiled by gfortran
libhamiltonian_3rd.so  /  libhamiltonian_2d_apical.so
    ↓ loaded by
ctypesroutines.py  ←────────────────────────────────────────────────┐
    ↓ used by                                                         │
absorption_2D_fortran.py  → results/figures/absorption*.png          │
effective_mass_tensor.py  → results/figures/eff_mass*.png            │
exciton_binding.py        → results/figures/exciton*.png             │
comparison_table.py       → results/figures/comparison*.png          │
convergence_plot.py       → results/figures/convergence*.png         │
tb_optimize.py            → finds best 26 parameters                 │
    ↓                                                                 │
generate_surrogate_data.py → data/surrogate_dataset.csv              │
    ↓                                                                 │
train_surrogate.py → data/surrogate_model.joblib                     │
    ↓                                                                 │
optuna_optimize.py → best parameters (fast, 3000 trials ~1 min)      │
    ↓                                                                 │
vqe_tb.py → quantum ground-state energy ←──────────────────────────┘
    │
    │  Phase 5 — Excitonic physics
    │
rytova_keldysh.py  ← V_RK_r, V_RK_q, solve_binding_energy
    │   Level-1: solve 2D radial SE with g=√r·ψ substitution
    │   → results/figures/rk_potential.png
    │   → results/figures/rk_binding_energies.png
    │
    ↓  (imports V_RK_q, r0_nm, material constants)
bse_2d.py  ← ctypesroutines.py (hamiltonian_cubic2D_apical_fortran)
    │   Level-2: k-space BSE matrix, band tracking, form factors
    │   → results/figures/bse_binding_energies.png
    │
    ↓  (imports solve_bse, EL_EXP)
absorption_elliott.py
        Level-2+Elliott: Lorentzian exciton peaks + 2D Sommerfeld continuum
        → results/figures/absorption_bse_all.png
        → results/figures/absorption_bse_n{1..5}.png
```

**What each Phase adds:**

| Phase | Script(s) | What it computes | Key technique |
|-------|-----------|-----------------|---------------|
| 0–1 | ctypesroutines.py, absorption_2D_fortran.py | Band structure, optical absorption | Fortran TB model via ctypes |
| 1 | effective_mass_tensor.py, exciton_binding.py | Effective masses, simple binding energy | Numerical derivatives, Wannier-Mott formula |
| 1 | comparison_table.py, convergence_plot.py | Model accuracy vs. experiment | Statistics (RMS, R²) |
| 1 | tb_optimize.py | Best-fit 26 parameters | Multi-objective NSGA2 genetic algorithm |
| 2 | generate_surrogate_data.py, train_surrogate.py, optuna_optimize.py | Fast parameter search | ML surrogate (gradient boosting) + Bayesian search |
| 3 | vqe_tb.py | Quantum ground-state energy | Variational Quantum Eigensolver (PennyLane) |
| 5-L1 | rytova_keldysh.py | Exciton binding via real-space SE | Rytova-Keldysh potential, finite-difference eigenvalue |
| 5-L2 | bse_2d.py | Exciton binding via k-space matrix | BSE with TB form factors and band tracking |
| 5-L2 | absorption_elliott.py | Full absorption spectrum | Elliott formula + 2D Sommerfeld enhancement |
| 6 | parse_geometry.py | Bond classification from VASP structures | PCA slab normal, PBC neighbor list, projection sorting |
| 6 | ham_ortho.py | Orthorhombic Hamiltonian H(k) + batch diagonalization | General Slater-Koster direction cosines, vectorized phases |
| 6 | plot_bands_interactive.py | Interactive 6-panel band viewer with GW overlay | Matplotlib TextBox/Button, eigenvalues_batch |
| 6 | fit_ortho.py | Fit 26 orthorhombic parameters | NSGA2 multi-objective, k-stencil masses, 8 parallel seeds |
| 6 | view_geometry.py | HTML 3-D structure viewer + bond shells | Plotly Scatter3d, togglable legend traces |
| 6 | visualize_geometry.py | Static PNG + VESTA POSCAR | matplotlib 3D, ASE, dummy-element remapping |

---

---

## Phase 6 — Orthorhombic TB+SOC: Tilted Pnma Structure

### What is the orthorhombic structure, in plain English?

The earlier TB model used the **cubic** (ideal) crystal structure: all Pb-I-Pb angles are exactly 180° and all bond lengths are the same. Real MAPbI₃ at room temperature adopts the **orthorhombic Pnma** phase: the PbI₆ octahedra are tilted, which lowers the symmetry and changes the electronic structure (band gap, effective masses, band ordering).

Phase 6 rebuilds the entire pipeline — geometry parsing, Hamiltonian, fitting, and visualization — for these tilted structures.

---

## 15. parse_geometry.py

**What it does:** Reads each VASP structure file, finds every Pb-I bond (using periodic boundary conditions), measures the bond direction, and classifies each bond and atom into one of five types. The result is saved as a JSON file used by `ham_ortho.py`.

**The five types:**

| Type | Meaning |
|------|---------|
| `Pb_interior` | Pb atom surrounded only by interior I atoms |
| `Pb_surface` | Pb atom with at least one surface apical I |
| `equatorial` | I atom bonded to two Pb atoms in the slab plane |
| `apical_interior` | I atom bridging two Pb layers (bonded to two Pb atoms along the stacking direction) |
| `apical_surface` | I atom on the outer surface of the slab, bonded to only one Pb |

---

```python
CUTOFF = 4.0   # Å  (Pb-I bonds are 3.1–3.3 Å; 4.0 gives clean separation)
ia, ja, Darr = neighbor_list('ijD', atoms, CUTOFF)
```

**Explanation:**
- `neighbor_list` is from the ASE library. It finds all pairs of atoms (i, j) closer than the cutoff distance, respecting periodic boundary conditions (PBC). PBC means it also considers atoms in neighbouring copies of the unit cell.
- `ia, ja`: arrays of atom-index pairs. `Darr`: the 3-D displacement vector from atom i to atom j (in Ångström), computed through the nearest PBC image.
- The cutoff 4.0 Å cleanly captures all Pb-I bonds (3.1–3.3 Å) and excludes Pb-Pb and I-I contacts (> 6 Å).

---

```python
def slab_normal(atoms):
    pb_pos = np.array([pos[a.index] for a in atoms if a.symbol == 'Pb'])
    cov = np.cov(pb_pos.T)
    vals, vecs = np.linalg.eigh(cov)
    n = vecs[:, np.argmax(vals)]
```

**Explanation:**
- This finds the stacking direction of the slab automatically.
- `np.cov(pb_pos.T)`: builds a 3×3 covariance matrix of the Pb atom positions. Each entry [i,j] measures how much the x/y/z coordinates vary together.
- `np.linalg.eigh`: diagonalizes the covariance matrix. The eigenvector with the largest eigenvalue points in the direction of maximum spread — which is the stacking direction, because the Pb layers are separated along it.
- This is called PCA (Principal Component Analysis). It discovers the most important direction in the data without being told what to look for.

---

```python
bonds.sort(key=lambda b: abs(b['proj_normal']), reverse=True)
for k, b in enumerate(bonds):
    if k < 2:
        is_surf = (label != '3D') and (pb_count[b['I_idx']] == 1)
        b['bond_type'] = 'apical_surface' if is_surf else 'apical_interior'
    else:
        b['bond_type'] = 'equatorial'
```

**Explanation:**
- Each Pb atom has exactly 6 bonds. The direction cosine along the slab normal (`proj_normal`) measures how aligned the bond is with the stacking direction.
- The two bonds most aligned with the stacking direction are "apical" (top and bottom). The remaining four are "equatorial" (in-plane), regardless of how tilted the octahedra are.
- `pb_count[b['I_idx']] == 1`: if this I atom is bonded to only one Pb, it has no neighbour above or below — it is on the surface.

---

**Output format (`geometry_n2.json`):**

```json
{
  "label": "n2",
  "slab_normal": [0.016, -0.015, 0.9998],
  "n_Pb": 8, "n_I": 20,
  "pb_atoms": [
    {
      "idx": 2, "cart": [3.14, 0.05, 3.22],
      "type": "Pb_interior",
      "bonds": [
        {"I_idx": 18, "dr_ang": [0.0, 0.0, 3.15],
         "lmn": [0.0, 0.0, 1.0], "dist_ang": 3.15,
         "proj_normal": 0.9998, "bond_type": "apical_interior"},
        ...
      ]
    }
  ]
}
```

The `lmn` list is the direction cosine (unit vector) from Pb to I — the key input to `ham_ortho.py`.

---

## 16. ham_ortho.py

**What it does:** Builds the tight-binding + SOC Hamiltonian H(k) for the orthorhombic structure and computes its eigenvalues (band energies). Unlike the cubic model (which assumes all bonds are the same), this model uses the exact direction cosines from `parse_geometry.py` for every bond in the crystal.

**The key idea — Slater-Koster integrals:**

A tight-binding Hamiltonian describes how electrons "hop" between orbitals on neighbouring atoms. For sp³ orbitals, the hopping amplitude depends on the direction of the bond. The Slater-Koster formulas express this as a combination of two parameters (Vσ and Vπ) multiplied by functions of the direction cosines (l, m, n):

```
H_ss = Vss
H_sp = l × Vsp   (if p is the px orbital)
H_ppσ = l² × Vppσ + (1-l²) × Vppπ   (px-px hopping along bond direction l,m,n)
...
```

The full 4×4 Pb-I hopping matrix `T(l,m,n)` contains all combinations of s, px, py, pz.

---

```python
PARAM_KEYS = [
    'Es0','Ep0','Es1','Ep1',
    'Vss','Vsp','Vps','Vppsig','Vpppi',
    'deltaSO0','deltaSO1',
    'Vss_II','Vsp_II','Vppsig_II','Vpppi_II',
    'Vss_IIz','Vsp_IIz','Vppsig_IIz','Vpppi_IIz',
    'Vss_PbPb','Vsp_PbPb','Vppsig_PbPb','Vpppi_PbPb',
    'Es1_ap','Ep1_ap','Ep0_surf',
]
```

**Explanation:**
- `Es0, Ep0`: on-site energies for Pb s and p orbitals.
- `Es1, Ep1`: on-site energies for I s and p orbitals.
- `Vss, Vsp, Vps, Vppsig, Vpppi`: Pb-I hopping parameters.
- `deltaSO0, deltaSO1`: spin-orbit coupling strength for Pb and I.
- `Vss_II, ...`: I-I hopping (second-nearest-neighbour).
- `Vss_IIz, ...`: I-I hopping in the apical direction (potentially different from in-plane).
- `Vss_PbPb, ...`: Pb-Pb hopping (third-nearest-neighbour).
- `Es1_ap, Ep1_ap`: separate on-site energies for apical I (different environment from equatorial I).
- `Ep0_surf`: separate Pb p on-site energy for surface Pb atoms.

26 parameters total: 4 on-site + 5 Pb-I hopping + 2 SOC + 8 I-I hopping + 4 Pb-Pb hopping + 3 surface corrections.

---

```python
def sk_block(l, m, n, Vss, Vsp, Vps, Vppsig, Vpppi):
    """4x4 Slater-Koster hopping block: Pb(s,px,py,pz) → I(s,px,py,pz)."""
    T = np.zeros((4, 4))
    T[0,0] = Vss
    T[0,1] = l * Vsp;  T[0,2] = m * Vsp;  T[0,3] = n * Vsp
    T[1,0] = -l * Vps; T[2,0] = -m * Vps; T[3,0] = -n * Vps
    T[1,1] = l*l*Vppsig + (1-l*l)*Vpppi
    T[2,2] = m*m*Vppsig + (1-m*m)*Vpppi
    T[3,3] = n*n*Vppsig + (1-n*n)*Vpppi
    T[1,2] = T[2,1] = l*m*(Vppsig - Vpppi)
    T[1,3] = T[3,1] = l*n*(Vppsig - Vpppi)
    T[2,3] = T[3,2] = m*n*(Vppsig - Vpppi)
    return T
```

**Explanation:**
- This computes the hopping between Pb (s, px, py, pz) and I (s, px, py, pz) for a bond pointing in direction (l, m, n).
- `l*l*Vppsig + (1-l*l)*Vpppi`: the component of the px-px hopping along the bond direction (σ) adds Vppsig; the perpendicular component (π) adds Vpppi. If the bond is along x (l=1, m=n=0), this gives pure σ character.
- `l*m*(Vppsig - Vpppi)`: mixing between px and py comes only from the σ contribution — at 90° to the bond there is no σ mixing, but at intermediate angles there is.

---

```python
def eigenvalues_batch(label, kvecs, params, geom=None):
    ...
    phases = np.exp(1j * (kv @ D_arr.T))            # (nk, n_hops)
    contrib = phases[:, :, None, None] * T_arr[None, :, :, :]
    H0_all = np.tile(H0_base, (nk, 1, 1))
    for ih, (r0, c0) in enumerate(zip(hops_r, hops_c)):
        H0_all[:, r0:r0+4, c0:c0+4] += contrib[:, ih, :, :]
        H0_all[:, c0:c0+4, r0:r0+4] += contrib[:, ih, :, :].conj().swapaxes(-2,-1)
    ...
    return np.linalg.eigvalsh(H_all)   # (nk, dim), sorted real
```

**Explanation:**
- The tight-binding Hamiltonian at wavevector k is built by adding phase factors to each hopping term: `H(k) = Σ_R T_R × exp(i k·R)`, where R is the displacement vector to the neighbour and T_R is the hopping matrix.
- Instead of computing this one k at a time (slow), `eigenvalues_batch` computes all k-points simultaneously.
- `kv @ D_arr.T`: matrix multiplication that computes the dot product k·R for every combination of k-point and hopping term in one step. Result is shape (nk, n_hops).
- `np.exp(1j * ...)`: applies the complex phase factor.
- `phases[:, :, None, None] * T_arr[None, :, :, :]`: multiplies each k-point × each hop by the corresponding 4×4 matrix, using broadcasting (no Python loops).
- The final `eigvalsh` call diagonalizes all `nk` Hermitian matrices in one batch — this is where most computation happens, and it parallelizes automatically across CPU cores.
- Speed: ~300 ms for 6 structures × 30 k-points, vs ~18 seconds with a Python loop.

---

**Spin doubling and SOC:**

The orbital Hamiltonian H₀ (size n_orb × n_orb) acts on spinless orbitals. To include spin-orbit coupling, the basis is doubled to (2 × n_orb) with alternating spin-up and spin-down entries:

```
H_full =  |↑⟩  |↓⟩
    |↑⟩  H₀    L·S
    |↓⟩  L·S†  H₀
```

The SOC block `L·S` for a p-orbital involves angular momentum matrices Lx, Ly, Lz scaled by the SOC constant δ/3. For lead (Pb), the large SOC (δ ≈ 1.5–3 eV) splits the p bands significantly and is the main reason perovskites have a direct band gap at the zone boundary.

---

## 17. plot_bands_interactive.py

**What it does:** Opens a Matplotlib window showing band structures for all 6 structures (3D bulk + 2D n=1–5) in a 2×3 grid. TextBox widgets let you type in parameter values; ^ and v buttons step each parameter by 0.01 eV. All panels update immediately when you press Enter or click a button.

---

```python
self.fig, axes = plt.subplots(2, 3, figsize=(16, 13))
plt.subplots_adjust(left=0.05, right=0.98, top=0.96,
                    bottom=0.35, wspace=0.24, hspace=0.38)
```

**Explanation:**
- `bottom=0.35`: the bottom 35% of the figure is reserved for the parameter control panel (26 TextBoxes + 52 buttons). The top 65% is the 6 band-structure panels.
- `wspace, hspace`: horizontal and vertical spacing between panels.

---

```python
from matplotlib.widgets import TextBox, Button

for row, key in enumerate(col):
    ax_tb = self.fig.add_axes([x, y, w, h])
    tb = TextBox(ax_tb, key, initial=f'{self.params[key]:.4f}')
    tb.on_submit(self.update)
    self.textboxes[key] = tb
```

**Explanation:**
- `TextBox` is a Matplotlib widget that shows an editable text field. Pressing Enter calls `self.update`.
- Each TextBox is positioned precisely using axes coordinates [left, bottom, width, height] in the range 0–1 (fraction of the figure).
- `on_submit`: registers the callback. Every time you press Enter in any TextBox, `self.update` is called with the new value — which triggers a full recompute of all 6 band structures.

---

```python
def _load_gw(ticks_3D):
    ...
    scale = ticks_3D[-1] / bands[0][0][-1]
    lines = [(b[0]*scale, b[1]-vbm) for b in bands if np.any(np.abs(b[1]-vbm) < 4.0)]
```

**Explanation:**
- The GW reference data (`ss-gw.band.dat`) stores k-distances in units where the full path ends at a specific value. The TB model uses Ångström⁻¹ units.
- `scale = ticks_3D[-1] / bands[0][0][-1]`: rescales the GW k-axis so that its endpoint matches the endpoint of the TB k-path (both go from Γ to Z along the same path Γ-X-S-Y-Γ-Z-U-R-T-Z).
- The GW bands are displayed in gray over the 3D panel only, as a reference target for the fitting.

---

```python
gi  = int(np.argmin(np.linalg.norm(kp, axis=1)))
vbm = bands[gi, self.nf[lbl] - 1]
bands -= vbm
```

**Explanation:**
- `np.linalg.norm(kp, axis=1)`: computes the length of each k-vector (distance from the Γ point). `argmin` finds the index of Γ.
- `self.nf[lbl]`: number of filled bands = n_Pb × 2 + n_I × 8. The VBM is the highest filled band at Γ (index nf-1), and the CBM is the lowest empty band (index nf).
- `bands -= vbm`: shifts all bands so the VBM at Γ is at 0 eV. This makes it easy to compare the band gap across structures.

---

## 18. fit_ortho.py

**What it does:** Fits the 26 orthorhombic TB parameters to match GW band-structure targets (3D bulk) and DFT gap + effective-mass targets (2D slabs n=1–5). Uses the same NSGA2 genetic algorithm as `tb_optimize.py` but with the new Hamiltonian and a wider set of physics constraints.

---

**Why NSGA2 (genetic algorithm) and not gradient descent?**

Gradient descent (like L-BFGS-B) needs to compute ∂(objective)/∂(parameter) — the gradient. The TB eigenvalue problem is not differentiable with respect to parameters in a simple way. Also, the objective landscape has many local minima: a gradient method starting from the wrong place gets stuck. NSGA2 is a genetic algorithm — it maintains a large "population" of candidate solutions, combines and mutates them, and over many generations gravitates toward good solutions without needing gradients.

"Multi-objective" means we minimize 4 objectives simultaneously (band gap errors, effective mass errors, etc.) instead of combining everything into a single weighted sum. The result is a Pareto front — a set of solutions where improving one objective requires worsening another.

---

```python
EG_3D_GA = 1.5725    # gap at Γ (GW)
TARGET_MV = np.array([0.209, 0.234, 0.207])   # vxx, vyy, vzz
TARGET_MC = np.array([0.236, 0.199, 0.245])   # cxx, cyy, czz
TARGET_EG_2D = np.array([3.016, 2.432, 2.206, 2.078, 1.971])   # n=1..5
TARGET_MU_2D = np.array([0.221, 0.217, 0.201, 0.196, 0.186])   # reduced masses
```

**Explanation:**
- The 3D targets come from GW calculations (ss-gw.band.dat): gap at Γ = 1.5725 eV, band positions at X/Y/Z high-symmetry points.
- The effective mass targets (mvxx, etc.) tell the model how quickly bands curve near the Γ point — a large mass means a flat band, a small mass means a steep band.
- The 2D targets come from DFT: the n=1 slab has a gap of 3.016 eV (large, because quantum confinement pushes the bands apart), shrinking toward bulk as n increases.

---

```python
# 5-point k-stencil for gap and mass
KPTS = np.vstack([[0,0,0],
                  [DK,0,0], [-DK,0,0],   # ±e1
                  [0,DK,0], [0,-DK,0]])  # ±e2

gap_2d = min(CBM_vals) - max(VBM_vals)
mu_e1 = HBAR2_M / (1/mc_e1 + 1/mv_e1)
mu_e2 = HBAR2_M / (1/mc_e2 + 1/mv_e2)
mu_2d  = 0.5 * (mu_e1 + mu_e2)
```

**Explanation:**
- The 2D gap is slightly indirect in some slabs (the VBM or CBM may not be exactly at Γ). Instead of only evaluating at Γ, we evaluate at 5 k-points (Γ and ±0.05 Å⁻¹ in both in-plane directions) and take `min(CBM) - max(VBM)` — this catches the case where the extremum is slightly off Γ.
- `HBAR2_M = 7.62 eV·Å²` is ℏ²/m_e in convenient units. The effective mass is computed from the finite difference: `m = ℏ²/(2 × d²E/dk²)`, where `d²E/dk² = (E(+dk) - 2E(0) + E(-dk)) / dk²`.
- The reduced mass μ combines the electron and hole masses: `1/μ = 1/mc + 1/mv`. For exciton binding (Phase 5), the relevant quantity is the reduced mass, not the individual masses.

---

```python
class OrthoFitProblem(Problem):
    def _evaluate(self, X, out, *args, **kwargs):
        F_all = []; G_all = []
        with mp.Pool(processes=N_WORKERS) as pool:
            results = pool.map(_eval_one, X)
        for f, g in results:
            F_all.append(f); G_all.append(g)
        out['F'] = np.array(F_all)
        out['G'] = np.array(G_all)
```

**Explanation:**
- `Problem` is a pymoo base class. NSGA2 calls `_evaluate` once per generation with a matrix `X` of shape (pop_size, 26) — each row is one candidate parameter set.
- `mp.Pool(processes=N_WORKERS)`: runs each candidate in a separate CPU process in parallel. With 8 processes, 100 candidates per generation takes ~37 ms instead of 300 ms.
- `out['F']`: objectives (4 numbers per candidate). `out['G']`: constraints (50 numbers per candidate; satisfied when ≤ 0).

---

## 19. view_geometry.py

**What it does:** Creates an interactive HTML file for each structure. Open it in any web browser to rotate, zoom, and hover over atoms to see their type and position. The 3-D view lets you inspect the tilt pattern of the PbI₆ octahedra.

---

```python
_EXTRA_BONDS = {
    'I–I  1st (~4–5 Å)' : (4.0, 5.15, 'rgba(0,200,220,0.55)',   'solid', 2),
    'I–I  2nd (~5–6.5 Å)': (5.15, 6.55, 'rgba(200,0,200,0.50)', 'dash',  2),
    'Pb–Pb  (~6–7.5 Å)' : (6.0, 7.55, 'rgba(255,180,0,0.55)',   'dot',   2),
}
```

**Explanation:**
- These are distance shells for second- and third-nearest neighbours. I-I bonds at 4–5 Å are the first shell (equatorial I atoms next to each other in the plane). At 5–6.5 Å, a second shell appears. Pb-Pb bonds at 6–7.5 Å are the nearest-Pb neighbours.
- These bonds are hidden by default (`visible='legendonly'`) — you click them in the legend to toggle them on. This lets you check the bond network without cluttering the initial view.

---

```python
traces.append(go.Scatter3d(
    x=s['x'], y=s['y'], z=s['z'],
    mode='lines',
    name=f'{name} [{s["n"]//2} bonds]',
    line=dict(color=color, width=width, dash=dash),
    hoverinfo='skip',
    visible='legendonly',
))
```

**Explanation:**
- `go.Scatter3d`: a Plotly 3-D line trace. Bond lines are represented as lists of (x, y, z) coordinate pairs separated by `None` — when Plotly encounters `None` it lifts the pen, so you can put many disconnected line segments in one trace.
- `s["n"]//2`: each bond is stored twice (once from each end), so the actual number of unique bonds is half the count.
- `visible='legendonly'`: the trace appears in the legend but is not drawn until the user clicks it. This is a standard Plotly pattern for optional overlays.

---

```python
if label != '3D':
    pb_pos = np.array([pb['cart'] for pb in data['pb_atoms']])
    centre = pb_pos.mean(axis=0)
    n = np.array(data['slab_normal'])
    proj  = pb_pos @ n
    span  = proj.max() - proj.min()
    arrow_len = span * 0.3
    tip = centre + n * arrow_len
    traces.append(go.Scatter3d(
        x=[centre[0], tip[0]], y=[centre[1], tip[1]], z=[centre[2], tip[2]],
        mode='lines+text', text=['', 'n̂'], ...
    ))
```

**Explanation:**
- A short arrow drawn from the centroid of the Pb cloud in the direction of the slab normal, labelled n̂. This shows the stacking direction visually.
- `pb_pos @ n`: projects each Pb position onto the slab normal — this gives the "height" of each Pb layer. `span` is the total thickness of the Pb cloud along the normal. The arrow length is 30% of this span — long enough to be visible but not dominant.

---

## 20. visualize_geometry.py

**What it does:** Generates two outputs per structure: a static PNG figure (for the paper or README) and a VASP POSCAR file with renamed species so VESTA (crystal structure viewer software) shows each bond type in a different color.

---

```python
VESTA_MAP = {
    'Pb_interior': 'Pb',
    'Pb_surface':  'Bi',
    'equatorial':       'I',
    'apical_interior':  'Br',
    'apical_surface':   'Cl',
}
```

**Explanation:**
- VESTA colours atoms by chemical element. By substituting dummy elements (Bismuth for surface Pb, Bromine for apical interior I, Chlorine for apical surface I), each bond type gets a distinct VESTA colour without modifying the actual crystal geometry.
- This is a common trick in computational crystallography: relabel atoms for visualization purposes while keeping the structure physically correct.

---

```python
order = [s for s in ['Pb','Bi','I','Br','Cl'] if s in groups]
lines.append('  ' + '  '.join(order))
lines.append('  ' + '  '.join(str(len(groups[s])) for s in order))
lines.append('Cartesian')
for sp in order:
    for pos in groups[sp]:
        lines.append(f'  {pos[0]:16.10f}  {pos[1]:16.10f}  {pos[2]:16.10f}')
```

**Explanation:**
- VASP POSCAR format requires all atoms of the same species grouped together, with a header line listing species names and a count line listing how many atoms of each species.
- We rebuild the file from scratch using the classified positions from `parse_geometry.py` — the original POSCAR is only used for the lattice vectors (`orig.cell.array`).
- `16.10f`: 16 characters wide, 10 decimal places. VASP requires high-precision coordinates.

---

## Summary: How all the scripts connect (updated with Phase 6)

```
Fortran (.f90 files)
    ↓ compiled by gfortran
libhamiltonian_3rd.so  /  libhamiltonian_2d_apical.so
    ↓ loaded by
ctypesroutines.py  ←────────────────────────────────────────────────┐
    ↓ used by                                                         │
absorption_2D_fortran.py  → results/figures/absorption*.png          │
effective_mass_tensor.py  → results/figures/eff_mass*.png            │
exciton_binding.py        → results/figures/exciton*.png             │
comparison_table.py       → results/figures/comparison*.png          │
convergence_plot.py       → results/figures/convergence*.png         │
tb_optimize.py            → finds best 26 parameters (cubic model)   │
    ↓                                                                 │
generate_surrogate_data.py → data/surrogate_dataset.csv              │
    ↓                                                                 │
train_surrogate.py → data/surrogate_model.joblib                     │
    ↓                                                                 │
optuna_optimize.py → best parameters (fast, 3000 trials ~1 min)      │
    ↓                                                                 │
vqe_tb.py → quantum ground-state energy ←──────────────────────────┘
    │
    │  Phase 5 — Excitonic physics
    │
rytova_keldysh.py  ← V_RK_r, V_RK_q, solve_binding_energy
    │   Level-1: solve 2D radial SE with g=√r·ψ substitution
    │   → results/figures/rk_potential.png
    │   → results/figures/rk_binding_energies.png
    │
    ↓  (imports V_RK_q, r0_nm, material constants)
bse_2d.py  ← ctypesroutines.py (hamiltonian_cubic2D_apical_fortran)
    │   Level-2: k-space BSE matrix, band tracking, form factors
    │   → results/figures/bse_binding_energies.png
    │
    ↓  (imports solve_bse, EL_EXP)
absorption_elliott.py
        Level-2+Elliott: Lorentzian exciton peaks + 2D Sommerfeld continuum
        → results/figures/absorption_bse_all.png
        → results/figures/absorption_bse_n{1..5}.png

    Phase 6 — Orthorhombic TB+SOC (tilted Pnma structure)

data/pos3D2Dtilted/*.vasp   ← VASP structure files (3D + n1..n5)
    ↓
parse_geometry.py
    │   PBC neighbor list, PCA slab normal, direction cosines
    │   → results/data/geometry_{3D,n1..n5}.json
    │
    ├──→ visualize_geometry.py
    │       PNG bond classification + VESTA POSCAR
    │       → results/figures/geometry_{3D,n1..n5}.png
    │       → results/data/vesta/geometry_{3D,n1..n5}.vasp
    │
    ├──→ view_geometry.py
    │       Interactive HTML (Plotly): Pb/I bond types + togglable shells
    │       → results/figures/geometry_{3D,n1..n5}.html
    │
    ↓
ham_ortho.py   (no standalone output — library used by other scripts)
    │   26-parameter orthorhombic TB+SOC Hamiltonian
    │   eigenvalues_batch: vectorized k-loop, batch eigvalsh
    │
    ├──→ plot_bands_interactive.py
    │       2×3 interactive band viewer, GW overlay, TextBox controls
    │       (interactive window only, no saved file)
    │
    └──→ fit_ortho.py
            NSGA2 multi-objective fitting (4 objectives, 50 constraints)
            8 parallel seeds, ~1 hr wall time
            → prints best-fit parameters (not committed)
```
