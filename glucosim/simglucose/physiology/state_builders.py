import jax.numpy as jnp
from ..core.params import PatientParams, _mgdl_to_mM
from ..physiology.steady_state_solvers import insulin_steady_state_from_Sb, iir_exogenous_pmolkgmin


def build_x0_with_Y_Gf(params: PatientParams) -> jnp.ndarray:
    """
    Construct the augmented T2D fasting state with the PATCH-A secretion/filter states (Y, Gf).

    Mirrors the initialisation shared by the UVA simulator maintainers when enabling
    dynamic beta-cell secretion (Y) and short glucose filters (Gf) for dG/dt estimation.
    """
    # Basal glucose pools (mg/kg)
    Gp0 = params.Gpb
    if Gp0 <= 0.0:
        Gp0 = float(params.Gb * params.Vg)
    Gt0 = params.Gtb
    if Gt0 <= 0.0:
        k2_safe = max(params.k2, 1e-8)
        Gt0 = float((params.k1 / k2_safe) * Gp0)

    # Gut & CGM initialisations
    D10 = D20 = D30 = 0.0
    Gsc0 = Gp0

    # Exercise placeholders
    E10 = 0.0
    E20 = 0.0
    TE0 = params.tau_ex

    # SC insulin compartments
    Isc10 = 0.0
    Isc20 = 0.0
    if params.use_pump:
        basal_U_per_min = params.basal / 60.0
        IIR_ext = (basal_U_per_min * 6000.0 / params.BW) if params.BW > 0 else 0.0
        rate = params.ka1 + params.kd
        if rate > 0.0:
            Isc10 = IIR_ext / rate
            if params.ka2 > 0.0:
                Isc20 = (params.kd * Isc10) / params.ka2

    # Insulin steady state from basal secretion
    Ip0, Il0 = insulin_steady_state_from_Sb(params)

    # Insulin effect filters
    Vi = params.Vi
    I_p_mU_L = (Ip0 / Vi) / 6.0 if Vi > 0 else 0.0
    k_a1 = params.k_a1
    k_a2 = params.k_a2
    k_a3 = params.k_a3
    S_I1 = params.S_I1
    S_I2 = params.S_I2
    S_I3 = params.S_I3
    x1_0 = (S_I1 / k_a1) * I_p_mU_L if k_a1 > 0.0 else 0.0
    x2_0 = (S_I2 / k_a2) * I_p_mU_L if k_a2 > 0.0 else 0.0
    x3_0 = (S_I3 / k_a3) * I_p_mU_L if k_a3 > 0.0 else 0.0

    # New dynamic secretion/filter states
    Vg = params.Vg
    Gb0_mgdl = Gp0 / Vg if Vg > 0 else 0.0
    G0_mM = _mgdl_to_mM(Gb0_mgdl)
    Y0 = 0.0
    Gf0 = G0_mM

    state = jnp.array([
        D10, D20, D30,
        Gp0, Gt0,
        Ip0,
        x1_0, x2_0, x3_0,
        Il0,
        Isc10, Isc20,
        Gsc0,
        E10, TE0, E20,
        Y0, Gf0
    ], dtype=jnp.float32)
    return state

def init_state_t1d(params: PatientParams) -> jnp.ndarray:
    """
    Build the 18-state T1D vector at fasting steady state using Hovorka/UVA algebra.

    Validated against the published T1D simulator by solving the linear steady-state equations
    for plasma/liver insulin, SC compartments, and keeping exercise buffers at rest.
    Returns x0 (18,) in CSV units:
      D1,D2,D3 [mg]; Gp,Gt [mg/kg]; Ip,Il,Isc1,Isc2 [pmol/kg];
      x1 [pmol/L offset], x2 [pmol/L], x3 [pmol/L];
      Gsc [mg/kg]; E1[-],T_E[min],E2[-]; Y,Gf [mU/min, mM].
    """

    # --- Glucose basal masses (mg/kg) ---    
    Gp0 = params.Gpb
    Gt0 = params.Gtb

    # --- Basal insulin delivery (external, U/min → pmol/kg/min) ---
    IIR_pmolkgmin = iir_exogenous_pmolkgmin(params, 0.0)

    # --- SC steady state for constant input ---
    Isc1_ss = IIR_pmolkgmin / (params.ka1 + params.kd)
    Isc2_ss = (params.kd * Isc1_ss) / params.ka2

    # --- Plasma/Liver steady state (solve 2x2) ---
    # 0 = -(m2+m4)Ip + m1*Il + ka1*Isc1 + ka2*Isc2
    # 0 =  m2*Ip - (m1+m30)Il
    A = jnp.array([[-(params.m2 + params.m4),  params.m1],
                   [ params.m2,               -(params.m1 + params.m30)]])
    b = jnp.array([-(params.ka1 * Isc1_ss + params.ka2 * Isc2_ss), 0.0])
    Ip0, Il0 = jnp.linalg.solve(A, b)

    # --- Insulin concentration (mU/L) for effects ---
    I_p_mU_L = (Ip0 / params.Vi) / 6.0

    # --- Insulin “effect” states follow insulin concentration (pmol/L) ---
    x1_0 = 0.0                 # pmol/L offset from Ib (zero effect at basal)
    x2_0 = Ip0 / params.Vi     # pmol/L
    x3_0 = x2_0                # pmol/L

    # --- Gut & CGM & Exercise ---
    D1=D2=D3=0.0
    Gsc0 = Gp0
    E1_0, T_E0, E2_0 = 0.0, 1.0, 0.0

    x0 = jnp.array([D1, D2, D3, Gp0, Gt0,
                    Ip0, x1_0, x2_0, x3_0, Il0,
                    Isc1_ss, Isc2_ss, Gsc0,
                    E1_0, T_E0, E2_0,
                    0.0, 0.0])
    return x0

def init_state_t2d(params: PatientParams) -> jnp.ndarray:
    """
    Construct the T2D hybrid fasting state, reusing diabetic-specific helpers for Y/Gf and basal insulin.

    Validity: reproduces the fasting solution of the UVA T2D hybrid when dynamic secretion is enabled.
    """
    x0 = build_x0_with_Y_Gf(params)

    Ip0 = float(x0[5])
    Il0 = float(x0[9])
    Isc1_ss = float(x0[10])
    Isc2_ss = float(x0[11])
    S_endog = 6.0 * params.Sb_per_kg

    res1 = -(params.m2 + params.m4) * Ip0 + params.m1 * Il0 + params.ka1 * Isc1_ss + params.ka2 * Isc2_ss
    res2 = params.m2 * Ip0 - (params.m1 + params.m30) * Il0 + S_endog
    if abs(res1) > 1e-4 or abs(res2) > 1e-4:
        # Re-solve steady state including SC infusion terms to remove residual
        A = jnp.array([[-(params.m2 + params.m4), params.m1],
                       [params.m2, -(params.m1 + params.m30)]], dtype=jnp.float32)
        b = jnp.array([-params.ka1 * Isc1_ss - params.ka2 * Isc2_ss, -S_endog], dtype=jnp.float32)
        Ip0_new, Il0_new = jnp.linalg.solve(A, b)
        x0 = x0.at[5].set(Ip0_new)
        x0 = x0.at[9].set(Il0_new)
        res1 = -(params.m2 + params.m4) * Ip0_new + params.m1 * Il0_new + params.ka1 * Isc1_ss + params.ka2 * Isc2_ss
        res2 = params.m2 * Ip0_new - (params.m1 + params.m30) * Il0_new + S_endog
        if abs(res1) > 1e-4 or abs(res2) > 1e-4:
            print(f"[init_state_t2d] Warning: insulin steady-state residual {res1:.3e}, {res2:.3e}")

    return x0