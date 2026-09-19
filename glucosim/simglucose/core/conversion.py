"""Unit conversions and the legacy hybrid T2D parameter conversion.

Parameter types remain in ``params``; this module owns conversion formulas and
checks without depending on the compatibility construction entry points.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from .params import PatientParams


def _mgdl_to_mM(g_mgdl: float) -> float:
    """Convert glucose from mg/dL to mM."""
    return g_mgdl / 18.0  # inverse of _mM_to_mgdl, using 180 mg/mmol


def _mM_to_mgdl(G_mM: float) -> float:
    """Convert glucose from mM to mg/dL."""
    return G_mM * 18.0


def _mmolmin_from_mgkgmin(val_mgkgmin: float, BW: float) -> float:
    """Convert mass rate per kg to molar rate for patient."""
    return (val_mgkgmin * BW) / 180.0


def _as_liters_from_Vg(Vg_raw: float, BW: float) -> float:
    """Convert glucose distribution volume to liters."""
    return float(Vg_raw * BW / 10.0)


def _as_liters_from_Vi(Vi_raw: float, BW: float) -> float:
    """Convert insulin distribution volume to liters."""
    return float(Vi_raw * BW)


def _mu_per_l(Ib_raw: float) -> float:
    """Convert basal insulin concentration."""
    return float(Ib_raw / 6.0)


def _almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    """Helper for floating-point comparisons that respects exact CSV values."""
    return float(np.abs(a - b)) <= tol


def _units_ok(value: float) -> bool:
    """Simple finiteness/positivity guard for derived parameters."""
    return np.isfinite(value) and value >= 0.0


def patient_to_t2d_params(base_params: PatientParams, use_dynamic_HE: bool = False, *,
                         effective_resistance: float | None = None) -> PatientParams:
    """
    Derive the additional fields the hybrid T2D ODE expects without mutating the CSV
    contract (glucose in mg/kg, insulin in pmol/kg, volumes in dL/kg or L/kg).

    Args:
        base_params: PatientParams populated directly from the CSV (plus behavior fields).
        use_dynamic_HE: Whether to model time-varying hepatic extraction.

    Returns:
        A new PatientParams with the original base fields intact and the T2D-only
        derived quantities expressed in the units required by the hybrid ODE.
    """
    BW = float(base_params.BW)
    Vg_dL_per_kg = float(base_params.Vg)
    Vi_L_per_kg = float(base_params.Vi)

    # These conversions only create model-side conveniences; base CSV entries stay mg/kg & pmol/kg.
    V_G = _as_liters_from_Vg(Vg_dL_per_kg, BW)   # L
    V_I = _as_liters_from_Vi(Vi_L_per_kg, BW)    # L

    # Basal glucose concentration in mg/dL is still (mg/kg)/(dL/kg); convert to mM for hybrid secretion.
    Gpb_mg_per_kg = float(base_params.Gpb)
    G_conc_mg_per_dL = Gpb_mg_per_kg / max(Vg_dL_per_kg, 1e-6)
    h_mM = _mgdl_to_mM(G_conc_mg_per_dL)

    # Gut absorption dynamics remain in 1/min; we only surface tau_D for the hybrid model.
    kgut_avg = 0.5 * (float(base_params.kmax) + float(base_params.kmin))
    k_eff = max(1e-6, min(kgut_avg, float(base_params.kabs)))
    tau_D = 0.5 / k_eff

    # SC insulin absorption time constant derived from existing rate constants.
    ka1 = float(base_params.ka1)
    ka2 = float(base_params.ka2)
    kd = float(base_params.kd)
    tau_S = 0.5 * (1.0 / max(ka1 + kd, 1e-6) + 1.0 / max(ka2, 1e-6))

    # Hepatic extraction handling mirrors the original conversion but never rewrites CSV fields.
    m1 = float(base_params.m1)
    m2 = float(base_params.m2)
    m4 = float(base_params.m4)
    HEb = float(jnp.clip(base_params.HEb, 0.0, 0.95))
    m3_b = (HEb * m1) / (1.0 - HEb + 1e-9)
    m5 = float(base_params.m5) if use_dynamic_HE else 0.0
    m6 = HEb

    # Map basal insulin from pmol/L-equivalent (CSV) into mU/L for secretion bookkeeping.
    Ib_mU_L = _mu_per_l(float(base_params.Ib))
    I_p_b = Ib_mU_L * V_I
    K_b = max(0.0, ((m3_b + m1) * (m2 + m4) / m1) - m2)
    S_t_b = max(0.0, K_b * I_p_b)
    # The result cannot be negative.
    S_sys_target_U_per_hr = 0.4

    # 5. Use the derived S_sys_target to calculate the final Sb_per_kg.
    # This is the formula from your discussion, which converts the systemic target
    # back into a per-kilogram portal secretion rate.
    Sb_per_kg = (S_sys_target_U_per_hr * 1000 / 60) / ((1 - HEb) * BW)

    # Insulin sensitivity terms can be scaled by the IR factor without breaking dimensionality.
    # Preserve the legacy floor for direct conversions. Factory-supplied
    # resistance has already been validated as finite and positive, and its
    # requested value (including values below that floor) is authoritative.
    ir_factor = (max(float(base_params.insulin_resistance_factor), 1e-6)
                 if effective_resistance is None else effective_resistance)
    S_I1 = base_params.S_I1 / ir_factor
    S_I2 = base_params.S_I2 / ir_factor
    S_I3 = base_params.S_I3 / ir_factor

    # k12 mirrors k2 but lives in the hybrid state space.
    k12 = float(base_params.k2)

    # Endogenous glucose production and CNS uptake remain mg/kg/min in the CSV.
    # We convert to mmol/min for the hybrid equations here.
    x3_b = (S_I3 / max(float(base_params.k_a3), 1e-8)) * Ib_mU_L
    EGP_b_mmol_per_min = _mmolmin_from_mgkgmin(float(base_params.EGPb), BW)
    EGP_0 = float(EGP_b_mmol_per_min / max(1.0 - x3_b, 1e-6))
    F_cns0 = _mmolmin_from_mgkgmin(float(base_params.Fsnc), BW)

    updated = dataclasses.replace(
        base_params,
        # Hybrid model expects these derived fields.
        A_G=0.9,
        tau_D=tau_D,
        MwG_mg_per_mmol=180.0,
        tau_S=tau_S,
        gamma=base_params.gamma,
        K_deriv=base_params.K_deriv,
        alpha_s=base_params.alpha_s,
        beta_s=base_params.beta_s,
        h=h_mM,
        Sb_per_kg=Sb_per_kg,
        m5=m5,
        m6=m6,
        V_I=V_I,
        k_a1=base_params.k_a1,
        k_a2=base_params.k_a2,
        k_a3=base_params.k_a3,
        S_I1=S_I1,
        S_I2=S_I2,
        S_I3=S_I3,
        V_G=V_G,
        EGP_0=EGP_0,
        F_cns0=F_cns0,
        k12=k12,
        beta_ex=base_params.beta_ex,
        alpha_QE=base_params.alpha_QE,
    )

    _assert_csv_contract_preserved(base_params, updated)
    _assert_t2d_units(updated)

    return updated


def _assert_csv_contract_preserved(original: PatientParams, updated: PatientParams) -> None:
    """
    Make sure the CSV-derived quantities stay identical so unit conversions happen at
    the ODE boundary rather than in parameter packing.
    """
    base_fields = (
        'BW', 'EGPb', 'Gb', 'Ib', 'u2ss',
        'Vg', 'Vi', 'V_G_L', 'V_I_L',
        'Ipb', 'Ilb', 'Gpb', 'Gtb',
        'Fsnc', 'ke1', 'ke2',
        'kp1', 'kp2', 'kp3',
        'k1', 'k2',
        'Vm0', 'Km0', 'Vmx',
    )
    for field in base_fields:
        original_val = getattr(original, field)
        updated_val = getattr(updated, field)
        if isinstance(original_val, (float, int)):
            assert _almost_equal(float(original_val), float(updated_val)), (
                f"CSV base field '{field}' was altered during T2D conversion "
                f"({original_val} -> {updated_val})."
            )
        else:
            assert original_val == updated_val, (
                f"CSV base field '{field}' was altered during T2D conversion."
            )


def _assert_t2d_units(params: PatientParams) -> None:
    """Run quick sanity checks on the hybrid T2D parameters."""
    assert _units_ok(params.V_G), "Expected V_G in liters and non-negative."
    assert _units_ok(params.V_I), "Expected V_I in liters and non-negative."
    assert _units_ok(params.EGP_0), "EGP_0 must be finite mmol/min."
    assert _units_ok(params.F_cns0), "F_cns0 must be finite mmol/min."
    assert np.isfinite(params.h) and params.h >= 0.0, "Setpoint h must be in mM."
    assert np.isfinite(params.Sb_per_kg) and params.Sb_per_kg >= 0.0, "Basal secretion must be >= 0."
