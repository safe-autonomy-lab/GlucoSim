import jax.numpy as jnp
from ..core.params import PatientParams, _mM_to_mgdl


def tune_Vm0_to_basal(params: PatientParams, x0: jnp.ndarray) -> float:
    """
    Tune Vm0 so that, at the current basal state x0, the whole-body balance
    EGP - Fsnc - E_renal - U_id = 0 holds (no meal, no exercise).

    U_id = (Vm0 + Vmx*I) * Gt / (Km0 + Gt)
    EGP = max(kp1 - kp2*Gp - kp3*I, 0)
    I is plasma insulin concentration (pmol/L).
    """
    # Basal pools and concentration
    Gp = float(x0[3])                  # mg/kg
    Gt = float(x0[4])                  # mg/kg
    Ip = float(x0[5])                  # pmol/kg
    Vi = max(float(params.Vi), 1e-9)   # L/kg
    I_pmolL = Ip / Vi                  # pmol/L

    # EGP from the linear form used in the T1D ODE (floored at 0)
    EGP_raw = params.kp1 - params.kp2 * Gp - params.kp3 * I_pmolL
    EGP = max(float(EGP_raw), 0.0)     # mg/kg/min

    # Renal loss (often zero at fasting, but include it for correctness)
    E_renal = params.ke1 * max(Gp - params.ke2, 0.0)  # mg/kg/min

    # Target peripheral uptake to close the whole-body balance
    target = max(EGP - params.Fsnc - E_renal, 0.0)    # mg/kg/min

    # Solve for Vm0 using the Michaelis–Menten form
    denom = max(Gt, 1e-9)
    Vm0_new = target * (params.Km0 + Gt) / denom - params.Vmx * I_pmolL
    return float(max(Vm0_new, 0.0))

def tune_kp1_to_EGPb(params: PatientParams, x0: jnp.ndarray) -> float:
    """
    Adjust kp1 so the linear EGP evaluated at the basal state equals EGPb.
    Useful when calibrating fasting steady states for the linear T1D model.
    """
    Gp = float(x0[3])
    Ip = float(x0[5])
    Vi = max(float(params.Vi), 1e-9)
    I_pmolL = Ip / Vi
    return float(params.EGPb + params.kp2 * Gp + params.kp3 * I_pmolL)

def egp0_for_target(params: PatientParams, G_target_mM: float, I_target_mU_L: float, iters: int = 10) -> float:
    """
    Iteratively calibrate EGP_0 so a target fasting (G*, I*) becomes an equilibrium.

    Based on the hepatic balance used in Dalla Man et al. (2007); the secant loop enforces
    EGP = F_cns + U_id + renal while respecting the x3 suppression used in the hybrid model.
    """
    BW, Vg = params.BW, params.Vg
    k1, k2 = params.k1, params.k2
    Vm0, Vmx, Km0 = params.Vm0, params.Vmx, params.Km0
    S_I3, k_a3 = params.S_I3, params.k_a3

    Gp_star = _mM_to_mgdl(G_target_mM) * Vg
    I_p_pmol_L = I_target_mU_L * 6.0
    Vmt = Vm0 + Vmx * I_p_pmol_L

    Gt = (k1 / max(k2, 1e-8)) * Gp_star
    for _ in range(int(max(iters, 1))):
        U_id = Vmt * Gt / (Km0 + Gt)
        Gt = max((k1 * Gp_star - U_id) / max(k2, 1e-8), 1e-6)

    F_cns = (params.F_cns0 * 180.0) / BW
    U_id = Vmt * Gt / (Km0 + Gt)

    ke1, ke2 = params.ke1, params.ke2
    E_renal = ke1 * max(Gp_star - ke2, 0.0)
    EGP_required = F_cns + U_id + E_renal

    x3 = min((S_I3 / max(k_a3, 1e-8)) * I_target_mU_L, 0.95)
    scale = max(1.0 - x3, 0.05)
    EGP0_mgkgmin = EGP_required / scale
    EGP0_mmol_min = EGP0_mgkgmin * BW / 180.0
    return float(EGP0_mmol_min)

def insulin_steady_state_from_Sb(params: PatientParams) -> tuple[float, float]:
    """
    Compute fasting plasma/liver insulin pools implied by basal secretion.

    Validity: solves the linear two-compartment balance used in Hovorka et al. (2004, Eq. 5–7).
    Falls back to patient-specified Ipb/Ilb when provided so legacy datasets remain reproducible.
    """
    Ipb = params.Ipb
    Ilb = params.Ilb
    if Ipb > 0.0 and Ilb > 0.0:
        return Ipb, Ilb

    m1 = params.m1
    m2 = params.m2
    m4 = params.m4
    m30 = params.m30
    Sb_per_kg = params.Sb_per_kg

    S_endog = 6.0 * Sb_per_kg                       # basal secretion in pmol/kg/min
    denom_sec = (m1 + m30) - (m2 * m1) / max(m2 + m4, 1e-8)
    denom_sec = max(denom_sec, 1e-8)

    Il0 = S_endog / denom_sec
    Ip0 = (m1 / max(m2 + m4, 1e-8)) * Il0
    return float(Ip0), float(Il0)

def iir_exogenous_pmolkgmin(params: PatientParams, insulin_U_per_min_from_action: float) -> float:
    """Compute exogenous insulin inflow to the first SC compartment, as in Hovorka’s 2-comp SC absorption model."""
    basal_U_per_min = params.basal / 60.0 if params.use_pump else 0.0
    # Remove 'or 0.0' since insulin_U_per_min_from_action is always a numeric value from action array
    total_U_per_min = basal_U_per_min + insulin_U_per_min_from_action
    return total_U_per_min * 6000.0 / params.BW
