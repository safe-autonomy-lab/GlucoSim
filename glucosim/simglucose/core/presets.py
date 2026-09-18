"""Legacy diabetes preset values, applied before the adapters' calibration passes.

These functions return fresh field dictionaries; they do not calibrate or change
runtime parameter types. Acceptance is supplied by the factory so its existing
global setting remains effective. The no-pump preset also changes physiology;
it is not a delivery-only comparison against the pump preset.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .params import PatientParams

T2D_INSULIN_RESISTANCE = 2.5
T2D_NO_PUMP_INSULIN_RESISTANCE = 2.8


def t2d_scaling_defaults() -> dict:
    """Return fresh legacy scaling settings; callers may supply their own dict."""
    return {
        'Ib_factor': 1.25,      # Residual beta-cell function (25% increase)
        'Ipb_factor': 1.25,     # Plasma insulin at basal state
        'HEb_factor': 0.85,     # Reduced hepatic insulin clearance
        'BW_factor': 1.15,      # Higher body weight for T2D
    }


def t1d_overrides(base_params: PatientParams, acceptance_probability: float) -> dict:
    return {
        # No endogenous insulin production
        'beta_cell_function': 0.0,

        # Normal insulin sensitivity
        'insulin_resistance_factor': 1.0,

        # Typically use pump for better control
        'use_pump': True,

        # Higher adherence to insulin therapy (survival depends on it)
        'bolus_acceptance_prob': acceptance_probability,

        # More careful meal planning
        'meal_acceptance_prob': acceptance_probability,

        # Tighter safety margins due to no endogenous backup
        'meal_safe_window': 60.0,  # minutes
        'bolus_safe_window': 60.0,  # minutes

        # Empirical T1D basal proxy (U/hr); the target-balance formula is T2D-only.
        'basal': base_params.BW * 0.011,  # U/h
    }


def t2d_overrides(basal_rate: float, acceptance_probability: float) -> dict:
    return {
        # Residual beta-cell function (25-30% remaining)
        'beta_cell_function': 0.25,

        # Moderate insulin resistance
        'insulin_resistance_factor': T2D_INSULIN_RESISTANCE,

        # Use pump for better glucose control
        'use_pump': True,

        # For now, full acceptance to simplify learning
        'bolus_acceptance_prob': acceptance_probability,

        # For now, full acceptance to simplify learning
        'meal_acceptance_prob': acceptance_probability,

        # Longer safety windows due to residual insulin production
        'meal_safe_window': 60.0,  # minutes
        'bolus_safe_window': 60.0,  # minutes

        # Higher limits due to insulin resistance
        'max_bolus_U': 10.0,
        'max_meal_g': 100.0,

        # Basal targeting Ib under the two-pool insulin balance
        'basal': basal_rate,  # U/h
    }


def t2d_no_pump_overrides(base_params: PatientParams, acceptance_probability: float) -> dict:
    return {
        # No insulin pump
        'use_pump': False,
        'beta_cell_function': 0.3,

        # Legacy no-pump resistance setting; differs from the pump preset.
        'insulin_resistance_factor': T2D_NO_PUMP_INSULIN_RESISTANCE,
        # Recompute from the original gains, not the already-scaled pump gains.
        'S_I1': base_params.S_I1 / T2D_NO_PUMP_INSULIN_RESISTANCE,
        'S_I2': base_params.S_I2 / T2D_NO_PUMP_INSULIN_RESISTANCE,
        'S_I3': base_params.S_I3 / T2D_NO_PUMP_INSULIN_RESISTANCE,

        # For now, full acceptance to simplify learning
        'bolus_acceptance_prob': acceptance_probability,

        # For now, full acceptance to simplify learning
        'meal_acceptance_prob': acceptance_probability,

        # Longer safety windows due to manual injections
        'meal_safe_window': 60.0,  # minutes
        'bolus_safe_window': 60.0,  # minutes

        # Higher limits to account for less precise delivery
        'max_bolus_U': 10.0,
        'max_meal_g': 100.0,
        'K_deriv': 30.0,

        # No pump, no basal
        'basal': 0.0,  # U/h baseline
    }
