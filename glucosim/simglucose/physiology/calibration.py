"""Factory-stage basal calibration, with legacy arithmetic and pass order.

Preset selection and optional stress scaling stay in core.params. Environment
initial-state tuning stays in physiology.initialization and still runs later.
These stages intentionally preserve repeated passes; deduplication or changing
which parameters are calibrated requires a separate behavior change.
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.params import PatientParams


def autobalance_basal_t1d(
    p: PatientParams,
    basal_scale: float = 1.0,
    hepatic_scale: float = 1.0,
) -> PatientParams:
    """
    Make the T1D default a true steady state for the provided (Gpb,Gtb,Ipb) and current p.
    Solves:
      1) peripheral utilization  U_id(Gtb, Ipb) = k1*Gpb - k2*Gtb  (=> Vmx)
      2) hepatic balance         EGP(Gpb,Ipb)   = Fsnc + E_renal + k1*Gpb - k2*Gtb (=> kp1)
    """
    # ---- Targets from basal masses ----
    U_star = p.k1 * p.Gpb - p.k2 * p.Gtb                      # mg/kg/min
    frac   = p.Gtb / (p.Km0 + p.Gtb)                          # dimensionless MM fraction

    # ---- Basal insulin concentration (pmol/L) ----
    I_p_pmol_L = p.Ipb / p.Vi

    # ---- (1) Peripheral: choose Vm0 (keep) and solve Vmx to hit U_star ----
    # Ensure Vmx is positive by capping Vm0 at a fraction of total uptake
    # If Vm0 from CSV is too high, insulin has no room to act.
    # Standard basal insulin-independent uptake is usually 1.0-2.5 mg/kg/min.
    Vm0_target = min(p.Vm0, U_star / frac * 0.75)

    # If Vm0_target is still too high or close to total, force it down.
    # We want Vmx * I_p > 0. So Vm0 + Vmx*I = U_star/frac.
    # Vmx = (U_star/frac - Vm0) / I_p.
    # Let's just use the adjusted Vm0.
    Vm0_new = Vm0_target * basal_scale
    Vmx_new = max((U_star/frac - Vm0_new) / max(I_p_pmol_L, 1e-6), 1e-5)

    # ---- (2) Hepatic: pick kp1 so dGp=0 at basal (no meal, no exercise) ----
    # Renal loss at basal:
    E_renal = p.ke1 * (p.Gpb - p.ke2) if p.Gpb > p.ke2 else 0.0
    # CNS usage (legacy Fsnc, mg/kg/min):
    F_cns = p.Fsnc

    # We want: 0 = EGP + Ra - F_cns - E_renal - k1*Gp + k2*Gt  (Ra=0 at basal)
    # With EGP = kp1 - kp2*Gp - kp3*I_conc (I_conc=Ipb/Vi):
    I_conc = I_p_pmol_L
    kp1_new = (p.kp2 * p.Gpb + p.kp3 * I_conc
               + F_cns + E_renal + p.k1 * p.Gpb - p.k2 * p.Gtb)
    kp1_new = kp1_new * hepatic_scale

    return dataclasses.replace(p, Vm0=Vm0_new, Vmx=Vmx_new, kp1=kp1_new)


def autobalance_basal_t2d(params: PatientParams) -> PatientParams:
    U_star = params.k1 * params.Gpb - params.k2 * params.Gtb
    frac = params.Gtb / max(params.Km0 + params.Gtb, 1e-6)

    I_p_pmol_L = params.Ipb / params.Vi
    I_p_mU_L   = I_p_pmol_L / 6.0

    Vm0_new = 1.0
    Vmx_new = max((U_star / max(frac, 1e-6) - Vm0_new) / max(I_p_pmol_L, 1e-6), 0.0)

    x3_basal = (params.S_I3 / max(params.k_a3, 1e-6)) * I_p_mU_L
    F_cns_mgkgmin = (params.F_cns0 * 180.0) / params.BW
    E_renal = params.ke1 * max(params.Gpb - params.ke2, 0.0)

    one_minus = max(1.0 - x3_basal, 1e-6)
    EGP0_mgkgmin = (U_star + F_cns_mgkgmin + E_renal) / one_minus
    EGP0_mmolmin = EGP0_mgkgmin * params.BW / 180.0

    return dataclasses.replace(params, Vm0=Vm0_new, Vmx=Vmx_new, EGP_0=EGP0_mmolmin)


def _steady_state_insulin_from_Sb(params: PatientParams) -> tuple[float, float]:
    """
    Compute fasting plasma and liver insulin masses implied by the current basal secretion.
    Mirrors the linear two-compartment equilibrium used for Hovorka/UVA calibration.
    """
    S_endog = 6.0 * float(params.Sb_per_kg)
    if S_endog <= 0.0:
        return 0.0, 0.0

    m1 = float(params.m1)
    m2 = float(params.m2)
    m4 = float(params.m4)
    m30 = float(params.m30)

    denom_sec = (m1 + m30) - (m2 * m1) / max(m2 + m4, 1e-8)
    denom_sec = max(denom_sec, 1e-8)

    Il_ss = S_endog / denom_sec
    Ip_ss = (m1 / max(m2 + m4, 1e-8)) * Il_ss
    return float(Ip_ss), float(Il_ss)


def pump_basal_rate_t2d(t2d_params: PatientParams) -> float:
    """Return the legacy pump rate in U/hour for the converted T2D parameters."""
    # Target Ib under the implemented plasma/liver balance, including return flow.
    liver_return = t2d_params.m1 / (t2d_params.m1 + t2d_params.m30)
    clearance = t2d_params.m2 + t2d_params.m4 - liver_return * t2d_params.m2
    required_pmolkgmin = (clearance * t2d_params.Ib * t2d_params.Vi
                         - liver_return * 6.0 * t2d_params.Sb_per_kg)
    # A pump cannot remove insulin: excessive endogenous supply gives zero basal
    # and an achieved insulin concentration above the target.
    return max(0.0, required_pmolkgmin * t2d_params.BW * 60.0 / 6000.0)


def calibrate_t2d_pump(t2d_params: PatientParams) -> PatientParams:
    """Balance glucose, then retain both legacy insulin steady-state passes."""
    # Balance at the pump preset stage; optional stress scaling still runs later.
    t2d_params = autobalance_basal_t2d(t2d_params)

    # Recompute Ipb/Ilb after preset application, before optional stress scaling.
    Ip_ss, Il_ss = _steady_state_insulin_from_Sb(t2d_params)
    t2d_params = dataclasses.replace(t2d_params, Ipb=Ip_ss, Ilb=Il_ss)

    # Preserve the second legacy pass even though it currently recomputes the
    # same result. Removing a calibration pass is outside this extraction.
    Ip_ss, Il_ss = _steady_state_insulin_from_Sb(t2d_params)
    t2d_params = dataclasses.replace(t2d_params, Ipb=Ip_ss, Ilb=Il_ss)

    return t2d_params


def calibrate_t2d_no_pump(params: PatientParams) -> PatientParams:
    """After no-pump overrides, synchronize insulin before balancing glucose."""
    Ip_ss, Il_ss = _steady_state_insulin_from_Sb(params)
    params = dataclasses.replace(params, Ipb=Ip_ss, Ilb=Il_ss)
    params = autobalance_basal_t2d(params)

    return params
