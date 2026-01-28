from typing import Tuple
from jax import jit, lax
import jax.numpy as jnp

from ..core.params import PatientParams, _mgdl_to_mM
from ..sim.realism import NoiseConfig, dynamic_factors_for_step, add_process_noise_structured, disturb_action
from ..physiology.steady_state_solvers import iir_exogenous_pmolkgmin


def _nn(x, dx):
    """Clamp derivatives so states cannot decrease below zero, matching the non-negativity guard in UVA/Padova."""
    return jnp.where((x <= 0.0) & (dx < 0.0), 0.0, dx)

@jit
def hovorka_t1d(
    x: jnp.ndarray,
    action: jnp.ndarray,        # [carb_g/min, insulin_U/min, hr_reserve]
    params: PatientParams,
    last_Qsto: float,           # mg    
    last_foodtaken: float       # g
) -> jnp.ndarray:
    """
    JAX implementation of the Hovorka/UVA T1D model with optional exercise couplings.

    The glucose subsystem matches Hovorka et al. (2004) while the exercise sinks follow
    the Visentin et al. (2014) UVA exercise add-on.  Serves as the core vector field for T1D rollouts.
    States (indices & units):
      0 D1 [mg], 1 D2 [mg], 2 D3 [mg]
      3 Gp [mg/kg], 4 Gt [mg/kg]
      5 Ip [pmol/kg]
      6 x1 [pmol/L offset], 7 x2 [pmol/L], 8 x3 [pmol/L]
      9 Il [pmol/kg]
     10 Isc1 [pmol/kg], 11 Isc2 [pmol/kg]
     12 Gsc [mg/kg]
     13 E1 [-], 14 T_E [min], 15 E2 [-]
     16 Y [mU/min], 17 Gf [mM]

    Parameters (key units):
      BW [kg], Vg [dL/kg], Vi [L/kg]
      kmax,kmin,kabs,k1,k2,ka1,ka2,kd,ksc,ki,m1,m2,m30,m4 [1/min]
      f [-], b,d [-]
      kp1 [mg/kg/min], kp2 [ (mg/kg/min) per (mg/dL) ], kp3 [ (mg/kg/min) per (pmol/L) ]
      Fsnc [mg/kg/min]
      Vm0 [mg/kg/min], Vmx [ (mg/kg/min) per (pmol/L) ], Km0 [mg/kg]
      ke1 [1/min], ke2 [mg/kg]
      Ib [pmol/L]

    Action:
      action[0] = carb_g_per_min [g/min]
      action[1] = insulin_U_per_min [U/min]
      action[2] = hr_reserve [-]
    """

    # Shorthands
    BW   = params.BW
    Vg   = params.Vg  # dL/kg
    Vi   = params.Vi  # L/kg

    kmax = params.kmax; kmin = params.kmin; kabs = params.kabs
    b    = params.b;    d    = params.d;    f    = params.f

    k1 = params.k1; k2 = params.k2
    ka1 = params.ka1; ka2 = params.ka2; kd = params.kd; ksc = params.ksc
    m1 = params.m1; m2 = params.m2; m30 = params.m30; m4 = params.m4
    ki = params.ki

    kp1 = params.kp1; kp2 = params.kp2; kp3 = params.kp3
    Fsnc = params.Fsnc

    Vm0 = params.Vm0; Vmx = params.Vmx; Km0 = params.Km0
    ke1 = params.ke1; ke2 = params.ke2
    Ib  = params.Ib    # pmol/L

    # Unpack action
    CHO_g_per_min, insulin_U_per_min, hr_reserve = action

    # Derived insulin input to SC (pmol/kg/min)
    IIR_pmolkgmin = iir_exogenous_pmolkgmin(params, insulin_U_per_min)

    # Meal (convert g → mg added to D1)
    meal_mg_per_min = CHO_g_per_min * 1000.0

    # ---- Initialize derivative vector ----
    dxdt = jnp.zeros(18)

    # -------------------------------------------------------------------------
    # GUT subsystem (mg)
    # -------------------------------------------------------------------------
    qsto = x[0] + x[1]                 # mg
    Dbar = last_Qsto + last_foodtaken * 1000.0  # mg

    def kgut_when_food():
        return kmin + (kmax - kmin) / 2.0 * (
            jnp.tanh(5.0 / (2.0 * Dbar * (1.0 - b)) * (qsto - b * Dbar))
            - jnp.tanh(5.0 / (2.0 * Dbar * d) * (qsto - d * Dbar)) + 2.0
        )
    kgut = lax.cond(Dbar > 0.0, kgut_when_food, lambda: kmax)

    # D1, D2, D3 [mg]
    dxdt = dxdt.at[0].set(-kmax * x[0] + meal_mg_per_min)
    dxdt = dxdt.at[1].set(kmax * x[0] - kgut * x[1])
    dxdt = dxdt.at[2].set(kgut * x[1] - kabs * x[2])

    # Plasma rate of appearance Ra [mg/kg/min]
    Ra_mgkgmin = f * kabs * x[2] / BW

    # -------------------------------------------------------------------------
    # EXERCISE
    # -------------------------------------------------------------------------
    age = params.age; HR0 = params.HR0
    tau_HR = params.tau_HR; alpha_HR = params.alpha_HR
    n_power = params.n_power; c1 = params.c1; c2 = params.c2
    tau_ex = params.tau_ex; tau_in = params.tau_in
    beta_ex = params.beta_ex; alpha_QE = params.alpha_QE

    HR_max = 220.0 - age
    HR = hr_reserve * (HR_max - HR0) + HR0
    dE1 = (HR - HR0 - x[13]) / tau_HR
    denom = alpha_HR * HR0
    z = (x[13] / jnp.maximum(denom, 1e-6)) ** n_power
    f_E1 = z / (1.0 + z)
    dT_E = (c1 * f_E1 + c2 - x[14]) / tau_ex
    T_E_safe = jnp.maximum(x[14], 1e-3)
    dE2 = -((f_E1 / tau_in) + (1.0 / T_E_safe)) * x[15] + (f_E1 * T_E_safe) / (c1 + c2)

    # Exercise glucose shunts (keep units = mg/kg/min)
    # Use proportional terms on Gp,Gt (mg/kg). The alpha_QE terms act as dimensioned gains.
    Q_E1_uptake   = beta_ex * (x[13] / HR0)                  # mg/kg/min (by design/tuning)
    # Saturating & capped sinks from E2
    E2sq = x[15] * x[15]
    alpha = params.alpha_QE            # 1/min, tune smaller (see below)

    # Michaelis-Menten style saturation
    Km_ex = 120.0                      # mg/kg, tune 80–200
    Vmax_ex = alpha * E2sq             # 1/min effective
    sink_p = Vmax_ex * x[3] / (Km_ex + x[3])   # mg/kg/min (plasma)
    sink_t = Vmax_ex * x[4] / (Km_ex + x[4])   # mg/kg/min (tissue)

    # Fractional per-minute cap (prevents >2–3% of pool per min)
    cap = 0.02                         # 1/min, tune 0.01–0.03
    Q_E21_uptake = jnp.minimum(sink_p, cap * x[3])
    Q_E22_uptake = jnp.minimum(sink_t, cap * x[4])

    # -------------------------------------------------------------------------
    # GLUCOSE subsystem (mg/kg & mg/kg/min)
    # -------------------------------------------------------------------------
    # EGP (mg/kg/min) linear form on Gp mass (mg/kg) and insulin concentration (pmol/L)
    I_conc_pmolL = x[5] / Vi
    # If x[8] (x3) is the insulin effect, you may substitute I_conc_pmolL with x[8] if model uses x3.
    EGP_raw = kp1 - kp2 * x[3] - kp3 * I_conc_pmolL
    EGP_mgkgmin = jnp.maximum(EGP_raw, 0.0)

    # Renal excretion E (mg/kg/min) with ke2 in mg/kg
    E_mgkgmin = jnp.where(x[3] > ke2, ke1 * (x[3] - ke2), 0.0)

    # Insulin-dependent utilization U_id (mg/kg/min) with MM in mg/kg & insulin effect via pmol/L
    Vmt_mgkgmin = Vm0 + Vmx * I_conc_pmolL
    U_id_mgkgmin = Vmt_mgkgmin * x[4] / (Km0 + x[4])  # Km0 & Gt are mg/kg

    # Exchange between compartments (mg/kg/min)
    # dGp/dt: +EGP +Ra - CNS - renal - k1*Gp + k2*Gt - exercise shift
    dGp_dt = EGP_mgkgmin + Ra_mgkgmin - Fsnc - E_mgkgmin - k1 * x[3] + k2 * x[4] - Q_E21_uptake
    dGp_dt = _nn(x[3], dGp_dt)
    dxdt = dxdt.at[3].set(dGp_dt)

    # dGt/dt: +k1*Gp - k2*Gt - U_id - exercise disposal + back shift
    dGt_dt = k1 * x[3] - k2 * x[4] - U_id_mgkgmin - Q_E22_uptake - Q_E1_uptake + Q_E21_uptake
    dGt_dt = _nn(x[4], dGt_dt)
    dxdt = dxdt.at[4].set(dGt_dt)

    # -------------------------------------------------------------------------
    # INSULIN subsystem (pmol/kg & pmol/kg/min)
    # -------------------------------------------------------------------------
    # Plasma insulin Ip dynamics
    dIp_dt = -(m2 + m4) * x[5] + m1 * x[9] + ka1 * x[10] + ka2 * x[11]
    dIp_dt = _nn(x[5], dIp_dt)
    dxdt = dxdt.at[5].set(dIp_dt)

    # Insulin action filters (example first-order)
    # x1 drives remote effect from plasma insulin concentration
    dxdt = dxdt.at[6].set(-params.p2u * x[6] + params.p2u * (I_conc_pmolL - Ib))
    dxdt = dxdt.at[7].set(-ki * (x[7] - I_conc_pmolL))
    dxdt = dxdt.at[8].set(-ki * (x[8] - x[7]))

    # Liver insulin Il
    dIl_dt = -(m1 + m30) * x[9] + m2 * x[5]
    dIl_dt = _nn(x[9], dIl_dt)
    dxdt = dxdt.at[9].set(dIl_dt)

    # SC insulin
    dIsc1_dt = - (ka1 + kd) * x[10] + IIR_pmolkgmin
    dIsc1_dt = _nn(x[10], dIsc1_dt)
    dxdt = dxdt.at[10].set(dIsc1_dt)

    dIsc2_dt = kd * x[10] - ka2 * x[11]
    dIsc2_dt = _nn(x[11], dIsc2_dt)
    dxdt = dxdt.at[11].set(dIsc2_dt)

    # CGM proxy (filtered Gp)
    dGsc_dt = -ksc * x[12] + ksc * x[3]
    dGsc_dt = _nn(x[12], dGsc_dt)
    dxdt = dxdt.at[12].set(dGsc_dt)

    # Exercise states
    dxdt = dxdt.at[13].set(dE1)
    dxdt = dxdt.at[14].set(dT_E)
    dxdt = dxdt.at[15].set(dE2)
    # This is the place holder to match the shape of states
    # extra parameters are used only for Type 2 Diabetes model
    dxdt = dxdt.at[16].set(0.0)
    dxdt = dxdt.at[17].set(0.0)
    return dxdt


@jit
def hybrid_t2d(
    x: jnp.ndarray,
    action: jnp.ndarray,          # [carb_g/min, insulin_U/min, hr_reserve]
    params: PatientParams,
    last_Qsto: float,             # mg
    last_foodtaken: float         # g
) -> jnp.ndarray:
    """
    T2D hybrid dynamics (Hovorka secretion + UVA glucose transport) expressed in CSV units.

    Validity references: dynamic secretion block from the PATCH-A note (beta-cell dynamics) layered on top
    of the Dalla Man et al. (2007) transport equations and Visentin et al. (2014) exercise shunts.
    Includes:
      - Meal appearance Ra (mg/kg/min)
      - Endogenous insulin secretion S (mU/min) → injected into liver in pmol/kg/min
      - Insulin effects x1..x3 driven by I_mU/L via S_Ii and k_ai
      - EGP suppression by x3 (T2D-style)
      - CNS utilization (F_cns0), renal loss (ke1,ke2)
      - Exchange between Gp/Gt
      - SC insulin absorption (2-comp), plus external insulin (U/min)
      - Optional exercise shunts (beta_ex, alpha_QE)
    """
    # State order: [D1,D2,D3,Gp,Gt,Ip,x1,x2,x3,Il,Isc1,Isc2,Gsc,E1,T_E,E2,Y,Gf]

    # ------------------ Unpack core per-kg volumes ------------------
    BW  = params.BW           # kg
    # Vg = params.Vg * params.BW / 10.0   # L
    # Vi = params.Vi * params.BW          # L
    Vg  = params.Vg           # dL/kg (glucose)
    Vi  = params.Vi           # L/kg  (insulin)

    # ------------------ Unpack meal absorption ----------------------
    kmax = params.kmax; kmin = params.kmin; kabs = params.kabs
    b    = params.b;    d    = params.d;    f    = params.f

    # ------------------ Glucose kinetics (per-kg) -------------------
    k1 = params.k1; k2 = params.k2
    kp1 = params.kp1                          # mg/kg/min
    # kp2 multiplies glucose *concentration* (mg/dL)
    kp2 = params.kp2                          # (mg/kg/min) per (mg/dL)
    # kp3 multiplies insulin *concentration* (pmol/L)  (consistent with your CSV writeup)
    kp3 = params.kp3                          # (mg/kg/min) per (pmol/L)

    Fsnc = params.Fsnc                        # mg/kg/min (CNS/erythrocytes)
    ke1  = params.ke1                         # 1/min
    ke2  = params.ke2                         # mg/kg (renal threshold)

    # ------------------ Insulin effects & kinetics ------------------
    Vm0 = params.Vm0                          # mg/kg/min
    Vmx = params.Vmx                          # (mg/kg/min) per (pmol/L)   (T1D CSV style)
    Km0 = params.Km0                          # mg/kg

    ka1 = params.ka1; ka2 = params.ka2; kd = params.kd
    m1  = params.m1;  m2  = params.m2;  m30 = params.m30; m4 = params.m4
    ki  = params.ki
    p2u = params.p2u

    # ------------------ T2D hybrid extras (units noted) -------------
    alpha_s = params.alpha_s                  # 1/min
    beta_s  = params.beta_s                   # mU/(min·mM)
    gamma   = params.gamma                    # 1/min
    K_deriv = params.K_deriv                  # mU/mM   (multiplied by dG/dt when dG/dt>0)
    h_mM    = params.h                        # mM      (glucose secretion setpoint)
    Sb_per_kg = params.Sb_per_kg              # mU/(kg·min) basal secretion per kg

    # Insulin effect filters sensitivities expect I in mU/L
    k_a1 = params.k_a1; k_a2 = params.k_a2; k_a3 = params.k_a3  # 1/min
    S_I1 = params.S_I1                      # L/mU
    S_I2 = params.S_I2                      # L/mU
    S_I3 = params.S_I3                      # L/mU

    # Glucose model T2D base (mmol/min), will convert to mg/kg/min
    EGP_0_mmol_min = params.EGP_0           # mmol/min
    F_cns0_mmol_min = params.F_cns0         # mmol/min

    # Exercise couplings
    beta_ex  = params.beta_ex
    alpha_QE = params.alpha_QE
    age   = params.age;    HR0 = params.HR0
    tau_HR = params.tau_HR; alpha_HR = params.alpha_HR
    n_pow  = params.n_power; c1 = params.c1; c2 = params.c2
    tau_ex = params.tau_ex;  tau_in = params.tau_in
    beta_cell_function = params.beta_cell_function
    tau_dG = getattr(params, "tau_dG", 3.0)

    # ------------------ Unpack state ------------------
    D1, D2, D3   = x[0], x[1], x[2]          # mg
    Gp, Gt       = x[3], x[4]                # mg/kg
    Ip           = x[5]                      # pmol/kg
    x1, x2, x3   = x[6], x[7], x[8]          # -
    Il           = x[9]                      # pmol/kg
    Isc1, Isc2   = x[10], x[11]              # pmol/kg
    Gsc          = x[12]                     # mg/kg
    E1, T_E, E2  = x[13], x[14], x[15]       # -, min, -
    Y_mU_min     = x[16]                     # mU/min
    Gf_mM        = x[17]                     # mM

    # ------------------ Unpack action -----------------
    carb_g_per_min, insulin_U_per_min, hr_reserve = action

    # ------------------ Derived inputs ----------------
    # IIR_ext_pmolkgmin = insulin_U_per_min * 6000.0 / BW   # external insulin to SC
    IIR_ext_pmolkgmin = iir_exogenous_pmolkgmin(params, insulin_U_per_min)
    meal_mg_per_min = carb_g_per_min * 1000.0             # g/min → mg/min

    # Concentrations for effects
    Gb_mgdl      = Gp / Vg                       # mg/dL
    G_mM         = _mgdl_to_mM(Gb_mgdl)          # mM
    I_p_pmol_L   = Ip / Vi                       # pmol/L
    I_p_mU_L     = I_p_pmol_L / 6.0              # mU/L

    # =========================
    # 1) Endogenous secretion (dynamic Y and filtered dG/dt)
    # =========================
    tau_dG = max(tau_dG, 1e-3)
    dGf_mM = (G_mM - Gf_mM) / tau_dG
    dGdt_mM = dGf_mM

    P = beta_s * jnp.maximum(G_mM - h_mM, 0.0)
    D = K_deriv * jnp.maximum(dGdt_mM, 0.0)
    dY = -alpha_s * Y_mU_min + P + D

    S_basal_mU_min = Sb_per_kg * BW
    S_total_mU_min = S_basal_mU_min + beta_cell_function * Y_mU_min
    S_endog_pmolkgmin = (S_total_mU_min * 6.0) / BW

    # =========================
    # 2) Gut / Meal subsystem (mg)
    # =========================
    qsto = D1 + D2
    Dbar = last_Qsto + last_foodtaken * 1000.0  # mg

    def kgut_food():
        return kmin + (kmax - kmin) / 2.0 * (
            jnp.tanh(5.0/(2.0*Dbar*(1.0 - b)) * (qsto - b*Dbar))
            - jnp.tanh(5.0/(2.0*Dbar*d) * (qsto - d*Dbar)) + 2.0
        )
    kgut = lax.cond(Dbar > 0.0, kgut_food, lambda: kmax)

    dD1 = -kmax * D1 + meal_mg_per_min
    dD2 =  kmax * D1 - kgut * D2
    dD3 =  kgut * D2 - kabs * D3

    # Rate of appearance (mg/kg/min)
    Ra_mgkgmin = f * kabs * D3 / BW

    # =========================
    # 3) Exercise (same structure as your T1D)
    # =========================
    HR_max = 220.0 - age
    HR = hr_reserve * (HR_max - HR0) + HR0
    dE1 = (HR - HR0 - E1) / tau_HR
    denom = alpha_HR * HR0
    z = (E1 / jnp.maximum(denom, 1e-6)) ** n_pow
    f_E1 = z / (1.0 + z)
    dTE = (c1 * f_E1 + c2 - T_E) / tau_ex
    T_E_safe = jnp.clip(T_E, 0.5, 60.0)
    dE2 = -((f_E1 / tau_in) + (1.0 / T_E_safe)) * E2 + (f_E1 * T_E_safe) / (c1 + c2)

    # Exercise shunts (mg/kg/min), simple proportional forms
    Q_E1_uptake  = beta_ex * (E1 / jnp.maximum(HR0, 1e-3))      # mild extra uptake

    # Saturate the E2-driven shunts and cap their per-minute fraction
    E2sq   = jnp.maximum(E2, 0.0) ** 2
    alpha  = alpha_QE                     # same units (1/min)
    Km_ex  = 120.0                        # mg/kg (tune 80–200 if desired)
    Vmax_p = alpha * E2sq                 # 1/min, effective "max" rate vs E2
    Vmax_t = Vmax_p

    sink_p = Vmax_p * Gp / (Km_ex + Gp)   # mg/kg/min, saturating with Gp
    sink_t = Vmax_t * Gt / (Km_ex + Gt)   # mg/kg/min, saturating with Gt

    cap = 0.02                            # ≤2% of pool per minute (tune 0.01–0.03)
    Q_E21_uptake = jnp.minimum(sink_p, cap * Gp)   # plasma → tissue shift
    Q_E22_uptake = jnp.minimum(sink_t, cap * Gt)   # tissue disposal

    # =========================
    # 4) Glucose subsystem (mg/kg & mg/kg/min)
    # =========================
    # T2D-style EGP: EGP_0 [mmol/min] suppressed by x3 (dimensionless insulin effect)
    # Convert mmol/min → mg/kg/min by 180/BW
    EGP0_mgkgmin = (EGP_0_mmol_min * 180.0) / BW
    x3_eff = jnp.clip(x3, 0.0, 0.95)
    EGP_mgkgmin  = jnp.maximum(EGP0_mgkgmin * (1.0 - x3_eff), 0.0)

    # CNS/brain usage in mg/kg/min
    F_cns_mgkgmin = (F_cns0_mmol_min * 180.0) / BW

    # Renal excretion with mg/kg threshold
    E_renal_mgkgmin = jnp.where(Gp > ke2, ke1 * (Gp - ke2), 0.0)

    # Insulin-dependent utilization (Michaelis–Menten in mg/kg with insulin modulation)
    # Here we use Vmx per pmol/L (matching your CSV ‘Vmx (mg/kg/min per pmol/l)’):
    Vmt_mgkgmin = Vm0 + Vmx * I_p_pmol_L
    U_id_mgkgmin = Vmt_mgkgmin * Gt / (Km0 + Gt)

    # NOTE: If you prefer the strict “T2D hybrid” form: replace “+kp1 - kp2*Gb - kp3*I”
    # with EGP_mgkgmin above (we are already doing that). If you keep EGP_mgkgmin,
    # drop "+ kp1 - kp2*Gb - kp3*I" to avoid double-counting. Common choice:
    # Comment the next line to REMOVE the linear kp1/kp2/kp3 drive:
    dGp = EGP_mgkgmin + Ra_mgkgmin - F_cns_mgkgmin - E_renal_mgkgmin \
          - k1 * Gp + k2 * Gt - Q_E21_uptake

    dGp = _nn(Gp, dGp)

    dGt =  k1 * Gp - k2 * Gt - U_id_mgkgmin - Q_E22_uptake - Q_E1_uptake + Q_E21_uptake
    dGt = _nn(Gt, dGt)

    # =========================
    # 5) Insulin subsystem (pmol/kg & pmol/kg/min)
    # =========================
    # Plasma insulin
    dIp = -(m2 + m4) * Ip + m1 * Il + ka1 * Isc1 + ka2 * Isc2
    dIp = _nn(Ip, dIp)

    # Effects x1..x3 driven by I in mU/L via S_Ii (L/mU)
    # (A.13–A.15 style)
    d_x1 = -k_a1 * x1 + S_I1 * (I_p_mU_L)
    d_x2 = -k_a2 * x2 + S_I2 * (I_p_mU_L)
    d_x3 = -k_a3 * x3 + S_I3 * (I_p_mU_L)
    # If you prefer your previous p2u/ki structure, you can keep that instead.

    # Liver insulin receives portal endogenous secretion
    dIl = -(m1 + m30) * Il + m2 * Ip + S_endog_pmolkgmin
    dIl = _nn(Il, dIl)

    # SC insulin (external insulin goes to Isc1)
    dIsc1 = -(ka1 + kd) * Isc1 + IIR_ext_pmolkgmin
    dIsc1 = _nn(Isc1, dIsc1)
    dIsc2 =  kd * Isc1 - ka2 * Isc2
    dIsc2 = _nn(Isc2, dIsc2)

    # CGM proxy (filtered Gp)
    dGsc = -params.ksc * Gsc + params.ksc * Gp
    dGsc = _nn(Gsc, dGsc)

    # =========================
    # 6) Exercise states
    # =========================
    dx = jnp.zeros_like(x)
    dx = dx.at[0].set(dD1)
    dx = dx.at[1].set(dD2)
    dx = dx.at[2].set(dD3)
    dx = dx.at[3].set(dGp)
    dx = dx.at[4].set(dGt)
    dx = dx.at[5].set(dIp)
    dx = dx.at[6].set(d_x1)
    dx = dx.at[7].set(d_x2)
    dx = dx.at[8].set(d_x3)
    dx = dx.at[9].set(dIl)
    dx = dx.at[10].set(dIsc1)
    dx = dx.at[11].set(dIsc2)
    dx = dx.at[12].set(dGsc)
    dx = dx.at[13].set(dE1)
    dx = dx.at[14].set(dTE)
    dx = dx.at[15].set(dE2)
    dx = dx.at[16].set(dY)
    dx = dx.at[17].set(dGf_mM)
    return dx


@jit
def t1d_rk4_step(
    x: jnp.ndarray,
    dt: float,
    action: jnp.ndarray,          # [carb_g/min, insulin_U/min, hr_reserve]
    params: PatientParams,
    last_Qsto: float,
    last_foodtaken: float,
    t_min: float,                 # absolute minute (for circadian)
    key: jnp.ndarray,
    cfg: NoiseConfig,
    ou_state_dL: jnp.ndarray      # scalar jnp array for OU state (mg/dL)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Wrapper around hovorka_t1d with action jitter, circadian delta, structured process noise.
    Returns (x_next, key, new_ou_state_dL).
    """

    # 1) action jitter (inject basal wobble as action delta)
    action_jit, key = disturb_action(action, params, key, cfg)

    # 2) dynamic factors (do not mutate params)
    factors = dynamic_factors_for_step(jnp.asarray(t_min), cfg)
    circ = factors["circadian"]  # JAX scalar

    # 3) RK4 on the original static params
    k1 = hovorka_t1d(x, action_jit, params, last_Qsto, last_foodtaken)
    k2 = hovorka_t1d(x + (dt / 2.0) * k1, action_jit, params, last_Qsto, last_foodtaken)
    k3 = hovorka_t1d(x + (dt / 2.0) * k2, action_jit, params, last_Qsto, last_foodtaken)
    k4 = hovorka_t1d(x + dt * k3,         action_jit, params, last_Qsto, last_foodtaken)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # 4) Exact/semi-implicit E2 update
    E1, T_E, E2 = x[13], x[14], x[15]
    HR0, alpha_HR, n_power = params.HR0, params.alpha_HR, params.n_power
    tau_in, c1, c2 = params.tau_in, params.c1, params.c2
    denom = jnp.maximum(alpha_HR * HR0, 1e-3)
    z = (E1 / denom) ** n_power
    f_E1 = z / (1.0 + z)
    T_E_safe = jnp.clip(T_E, 1.0, 60.0)
    sum_c = jnp.maximum(c1 + c2, 1e-3)
    a = (f_E1 / jnp.maximum(tau_in, 1e-6)) + (1.0 / T_E_safe)
    b = (f_E1 * T_E_safe) / sum_c
    exp_term = jnp.exp(-a * dt)
    E2_next = jnp.maximum(E2 * exp_term + jnp.where(a > 0.0, (b / a) * (1.0 - exp_term), b * dt), 0.0)
    x_next = x_next.at[15].set(E2_next)

    # clamps / projections
    POS = jnp.array([0,1,2, 3,4, 5,9, 10,11, 12])
    x_next = x_next.at[POS].set(jnp.maximum(x_next[POS], 0.0))
    x_next = x_next.at[14].set(jnp.clip(x_next[14], 0.5, 60.0))

    # 5) Circadian delta for T1D: modulate kp1 as additive ΔEGP on Gp (exact for constant term over dt)
    delta_egp = (circ - 1.0) * params.kp1  # mg/kg/min
    x_next = x_next.at[3].add(dt * delta_egp)

    # 6) structured process noise
    x_next, key, ou_state_dL = add_process_noise_structured(
        x_next, jnp.asarray(dt), key, cfg, params, ou_state_dL
    )

    return x_next, key, ou_state_dL


@jit
def t2d_rk4_step(
    x: jnp.ndarray,
    dt: float,
    action: jnp.ndarray,          # [carb_g/min, insulin_U/min, hr_reserve]
    params: PatientParams,
    last_Qsto: float,
    last_foodtaken: float,
    t_min: float,                 # absolute minute (for circadian)
    key: jnp.ndarray,
    cfg: NoiseConfig,
    ou_state_dL: jnp.ndarray      # scalar jnp array for OU state (mg/dL)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Wrapper around hybrid_t2d with action jitter, circadian delta on EGP_0*(1-x3),
    structured process noise. Returns (x_next, key, new_ou_state_dL).
    """
    # 1) action jitter
    action_jit, key = disturb_action(action, params, key, cfg)

    # 2) dynamic factors
    factors = dynamic_factors_for_step(jnp.asarray(t_min), cfg)
    circ = factors["circadian"]

    # 3) RK4 on static params
    k1 = hybrid_t2d(x, action_jit, params, last_Qsto, last_foodtaken)
    k2 = hybrid_t2d(x + (dt / 2.0) * k1, action_jit, params, last_Qsto, last_foodtaken)
    k3 = hybrid_t2d(x + (dt / 2.0) * k2, action_jit, params, last_Qsto, last_foodtaken)
    k4 = hybrid_t2d(x + dt * k3,         action_jit, params, last_Qsto, last_foodtaken)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # clamps / projections similar to your t2d_rk4_step
    x_next = x_next.at[14].set(jnp.clip(x_next[14], 0.5, 60.0))
    POS = jnp.array([0,1,2, 3,4, 5,9, 10,11, 12, 16])
    x_next = x_next.at[POS].set(jnp.maximum(x_next[POS], 0.0))

    # 4) Circadian delta for T2D: base EGP term is (EGP_0 * 180 / BW) * (1 - x3_eff)
    #    We approximate its modulation by adding dt * (circ-1) * EGP0_mgkgmin * (1 - x3_eff)
    EGP0_mgkgmin = (params.EGP_0 * 180.0) / params.BW
    x3_eff = jnp.clip(x_next[8], 0.0, 0.95)
    delta_egp = (circ - 1.0) * EGP0_mgkgmin * (1.0 - x3_eff)
    x_next = x_next.at[3].add(dt * delta_egp)

    # 5) structured process noise
    x_next, key, ou_state_dL = add_process_noise_structured(
        x_next, jnp.asarray(dt), key, cfg, params, ou_state_dL
    )

    return x_next, key, ou_state_dL