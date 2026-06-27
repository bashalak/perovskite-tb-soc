"""
phase7a_excitons.py — Predict exciton binding + polaronic masses from Gen 140.

Pipeline:
    Step 1: Run Rytova-Keldysh Wannier-Mott on the Gen 140 bare-band μ → E_b(n)
    Step 2: Compute Fröhlich polaron coupling α and predict polaronic μ
    Step 3: Compare predictions to Blancon 2018 experimental values
    Step 4: Save numeric results for Figure 5

The bare-band μ_2D values come from `best_gen140_canonical.npz` (Gen 140 fit
to DFT-1/2 + Jiang ss-GW targets — see Figure 4).

Uses the existing `rytova_keldysh.py` module by overriding its parameter dicts
with Gen 140-specific values.

Output: results/data/phase7a_excitons.npz  (predictions + experiment values)

Run:
    source ~/TBT-TS-sisl-workshop/setup.sh
    python3 phase7a_excitons.py
"""
import os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import rytova_keldysh as rk


# ── 1. Gen 140 inputs (the values fixed at the end of fit) ────────────────────

GEN140 = {
    'mu_2d':  np.array([0.200, 0.159, 0.139, 0.139, 0.137]),  # bare-band reduced mass
    'eg_2d':  np.array([2.840, 2.457, 2.205, 2.181, 2.110]),  # quasiparticle gap at Γ
}

# ── 2. Blancon 2018 experimental values (Nat. Commun. 9, 2254) ────────────────

BLANCON = {
    'mu_polaron': np.array([0.221, 0.217, 0.201, 0.196, 0.186]),  # polaronic μ
    'E_b_meV':    np.array([476.0, 252.0, 176.0, 158.0, 121.0]),   # exciton binding
    'E_elec':     np.array([3.016, 2.432, 2.206, 2.078, 1.971]),   # electronic gap
    'E_opt':      np.array([2.540, 2.180, 2.030, 1.920, 1.850]),   # optical (PL) gap
}

# ── 3. Fröhlich polaron material constants (MAPbI3 literature) ────────────────

# Wright et al. 2016, Nat. Commun. 7, 11755 — values used throughout paper §5.2
EPS_INF       = 6.50      # high-frequency (electronic) dielectric
EPS_STATIC    = 25.7      # static (ionic + electronic) dielectric
OMEGA_LO_meV  = 11.5      # LO phonon energy in meV  (Wright 2016 B3u Pb-I stretching mode)
OMEGA_LO_eV   = OMEGA_LO_meV / 1000.0

# Conversion constants
HBAR_eV_s     = 6.582119569e-16       # eV·s
ME_kg         = 9.1093837015e-31      # kg
E_C           = 1.602176634e-19       # C
EPS0          = 8.8541878128e-12      # F/m
HARTREE_eV    = 27.211386245988       # eV/Ha


def frohlich_alpha(mu_in_m0, omega_LO_eV, eps_inf, eps_0):
    """
    Fröhlich coupling constant:
        α = (1/ε∞ − 1/ε₀) × √( e²/4πε₀ · √(m_bare · m_e / 2ℏ ω_LO) ) / (ℏω_LO)

    Simpler dimensionless form (from Mahan, Many-Particle Physics):
        α = (1/ε∞ − 1/ε₀) × √( m_bare · Ry / ω_LO )
                                                    [Ry in Hartree units]
    With ω_LO in eV and m_bare in units of m_e:
        α = (1/ε∞ − 1/ε₀) × (Ry_eV / ω_LO_eV) × √( m_bare · ω_LO_eV / Ry_eV )
    """
    Ry_eV = 13.605693      # Rydberg in eV
    return (1.0/eps_inf - 1.0/eps_0) * np.sqrt(mu_in_m0 * Ry_eV / omega_LO_eV)


def polaron_mass_enhancement(alpha, order='first'):
    """
    m_polaron / m_bare in the weak-to-intermediate coupling limit.
        first-order  (Fröhlich 1954):  1 + α/6
        higher-order (Lee-Low-Pines):  1 + α/6 + 0.0236 α² + ...
    """
    if order == 'first':
        return 1.0 + alpha / 6.0
    elif order == 'second':
        return 1.0 + alpha / 6.0 + 0.0236 * alpha**2
    else:
        raise ValueError("order must be 'first' or 'second'")


def main():
    print("=" * 78)
    print("Phase 7A — Exciton binding + polaronic mass predictions from Gen 140")
    print("=" * 78)

    # ── Step 1: Run R-K with Gen 140 μ ────────────────────────────────────────
    # The existing rytova_keldysh.solve_binding_energy reads from MU_M0 dict.
    # We override it with our bare-band Gen 140 μ values.
    print("\n[Step 1] Solving Rytova-Keldysh Wannier-Mott with Gen 140 μ ...")
    print(f"         ε_slab = {rk.EPS_SLAB},  ε_env = {rk.EPS_ENV} (Blancon 2018)")

    # Override MU_M0 dict with Gen 140 bare-band values (in m_e)
    mu_override = {i + 1: float(GEN140['mu_2d'][i]) for i in range(5)}
    print(f"         Gen 140 μ values: {mu_override}")
    rk.MU_M0 = mu_override

    eb_rk = np.zeros(5)
    for n in range(1, 6):
        E_b_meV, _ = rk.solve_binding_energy(n, r_max=120.0, Nr=4000)
        eb_rk[n - 1] = E_b_meV

    print("\n  R-K results (Gen 140 μ):")
    print(f"  {'n':>3}  {'r0(nm)':>7}  {'μ_TB':>7}  {'E_b(predict, meV)':>18}  "
          f"{'E_b(Blancon, meV)':>18}  {'Δ(meV)':>8}")
    for i, n in enumerate(range(1, 6)):
        delta = eb_rk[i] - BLANCON['E_b_meV'][i]
        print(f"  {n:>3}  {rk.r0_nm(n):>7.3f}  {GEN140['mu_2d'][i]:>7.3f}  "
              f"{eb_rk[i]:>18.1f}  {BLANCON['E_b_meV'][i]:>18.1f}  {delta:>+8.1f}")

    # ── Step 2: Fröhlich polaron correction to masses ─────────────────────────
    print("\n[Step 2] Fröhlich polaron coupling and mass enhancement ...")
    print(f"         ε∞ = {EPS_INF},  ε_static = {EPS_STATIC},  "
          f"ω_LO = {OMEGA_LO_meV} meV  [Sendner 2016 / Schlipf 2018]")

    mu_polaron_pred_1st = np.zeros(5)
    mu_polaron_pred_2nd = np.zeros(5)
    alpha_arr = np.zeros(5)
    for i, n in enumerate(range(1, 6)):
        mu = GEN140['mu_2d'][i]
        alpha = frohlich_alpha(mu, OMEGA_LO_eV, EPS_INF, EPS_STATIC)
        alpha_arr[i] = alpha
        # Use the same ALPHA_FACTOR as the R-K solver (rk.ALPHA_FACTOR = 1.34)
        # so that Fig 5 Panel A (E_b via R-K) and Panel B (μ_polaron prediction)
        # are based on the same Fröhlich correction.
        mu_polaron_pred_1st[i] = mu * rk.ALPHA_FACTOR
        mu_polaron_pred_2nd[i] = mu * polaron_mass_enhancement(alpha, order='second')

    print("\n  Polaron mass enhancement:")
    print(f"  {'n':>3}  {'μ_bare':>7}  {'α':>6}  {'1+α/6':>7}  "
          f"{'μ_pol(1st)':>11}  {'μ_pol(2nd)':>11}  {'μ_Blancon':>10}")
    for i, n in enumerate(range(1, 6)):
        a = alpha_arr[i]
        print(f"  {n:>3}  {GEN140['mu_2d'][i]:>7.3f}  {a:>6.3f}  {1+a/6:>7.3f}  "
              f"{mu_polaron_pred_1st[i]:>11.3f}  {mu_polaron_pred_2nd[i]:>11.3f}  "
              f"{BLANCON['mu_polaron'][i]:>10.3f}")
    print(f"  mean α = {alpha_arr.mean():.3f}  "
          f"⟨μ_pol_pred / μ_bare⟩ (1st) = {np.mean(mu_polaron_pred_1st/GEN140['mu_2d']):.3f}")

    # ── Step 3: Predicted optical gap from TB Eg − E_b ────────────────────────
    print("\n[Step 3] Predicted optical gap = E_TB − E_b ...")
    E_opt_pred = GEN140['eg_2d'] - eb_rk / 1000.0
    print(f"  {'n':>3}  {'E_TB(eV)':>9}  {'E_b(eV)':>9}  "
          f"{'E_opt_pred':>11}  {'E_opt_Blancon':>14}  {'Δ(eV)':>7}")
    for i, n in enumerate(range(1, 6)):
        delta = E_opt_pred[i] - BLANCON['E_opt'][i]
        print(f"  {n:>3}  {GEN140['eg_2d'][i]:>9.3f}  {eb_rk[i]/1000.0:>9.3f}  "
              f"{E_opt_pred[i]:>11.3f}  {BLANCON['E_opt'][i]:>14.3f}  {delta:>+7.3f}")

    # ── Step 4: Save results for Figure 5 ─────────────────────────────────────
    out_path = os.path.join(_HERE, '..', '..', 'results', 'data',
                            'phase7a_excitons.npz')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
             n_vals             = np.arange(1, 6),
             mu_TB              = GEN140['mu_2d'],
             eg_TB              = GEN140['eg_2d'],
             E_b_RK_meV         = eb_rk,
             E_opt_pred         = E_opt_pred,
             alpha_Frohlich     = alpha_arr,
             mu_polaron_pred_1st= mu_polaron_pred_1st,
             mu_polaron_pred_2nd= mu_polaron_pred_2nd,
             E_b_exp_meV        = BLANCON['E_b_meV'],
             E_opt_exp          = BLANCON['E_opt'],
             E_elec_exp         = BLANCON['E_elec'],
             mu_polaron_exp     = BLANCON['mu_polaron'],
             eps_inf            = EPS_INF,
             eps_static         = EPS_STATIC,
             omega_LO_meV       = OMEGA_LO_meV,
             eps_slab           = rk.EPS_SLAB,
             eps_env            = rk.EPS_ENV,
             )
    print(f"\nSaved {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SUMMARY (vs Blancon 2018):")
    print("=" * 78)
    rms_eb = np.sqrt(np.mean((eb_rk - BLANCON['E_b_meV'])**2))
    rms_mu = np.sqrt(np.mean((mu_polaron_pred_1st - BLANCON['mu_polaron'])**2))
    rms_opt = np.sqrt(np.mean((E_opt_pred - BLANCON['E_opt'])**2))
    print(f"  E_b RMS error (R-K vs Blancon)     : {rms_eb:.1f} meV")
    print(f"  μ_polaron RMS error (1st order)    : {rms_mu:.4f} m_e")
    print(f"  E_opt RMS error (predicted vs exp) : {rms_opt:.3f} eV")
    print()


if __name__ == '__main__':
    main()
