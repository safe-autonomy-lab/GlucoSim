"""Scalar construction options and factory keyword extraction.

These options preserve historical meanings, including ignored T2D autobalance
options and post-calibration stress scales. They are not part of the JAX tree.
Acceptance must be supplied at call time by the compatibility entry point.
T2D factors belong to the adaptation boundary; late patient overrides belong
to the factory. Neither mapping is stored in the options object. Values
in BuildOptions are not coerced or newly validated. The separate effective-input
extractor validates only the explicitly covered physiological factory inputs.
"""
from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class BuildOptions:
    acceptance_probability: float
    autobalance_enabled: bool = True
    autobalance_basal_scale: float = 1.0
    autobalance_hepatic_scale: float = 1.0
    carb_absorption_scale: float = 1.0
    insulin_sensitivity_scale: float = 1.0
    eat_rate_scale: float = 1.0


def extract_construction_options(acceptance_probability: float,
                                 overrides: dict) -> BuildOptions:
    """Consume the six construction keywords from a factory-owned dictionary.

    The caller copies public keyword arguments before calling this function.
    Remaining entries are late patient overrides, including unknown names whose
    existing errors must occur at the factory's final replacement stage.
    Omitted scalar settings use the options class defaults; explicit values
    pass through unchanged. Acceptance is a separate, already resolved input.
    """
    values = {}
    for name in (
        "autobalance_enabled",
        "autobalance_basal_scale",
        "autobalance_hepatic_scale",
        "carb_absorption_scale",
        "insulin_sensitivity_scale",
        "eat_rate_scale",
    ):
        if name in overrides:
            values[name] = overrides.pop(name)
    return BuildOptions(acceptance_probability, **values)


def extract_effective_inputs(diabetes_type: str, overrides: dict) -> dict:
    """Consume covered effective inputs; leave unlisted late overrides alone.

    BW [kg], Vg [dL/kg], Vi [L/kg] and resistance must be positive.
    Gpb [mg/kg], Fsnc and EGPb [mg/kg/min] may be zero. These values
    describe the effective patient, after preset input factors. Output overrides
    that conversion or initialization owns are rejected before CSV loading.
    """
    if diabetes_type == 't1d':
        forbidden = ('kp1', 'Vm0')
    else:
        forbidden = ('h', 'F_cns0', 'Sb_per_kg', 'S_I1', 'S_I2', 'S_I3', 'EGP_0')
    for name in forbidden:
        if name in overrides:
            owner = 'initialization' if name in ('kp1', 'Vm0', 'EGP_0') else 'conversion'
            raise ValueError(f"Cannot override {name} for {diabetes_type}: {owner}-owned output")

    effective = {}
    for name in ('BW', 'Vg', 'Vi', 'Gpb', 'Fsnc', 'EGPb', 'insulin_resistance_factor'):
        if name not in overrides:
            continue
        value = overrides[name]
        positive = name in ('BW', 'Vg', 'Vi', 'insulin_resistance_factor')
        try:
            real_scalar = isinstance(value, Real) and not isinstance(value, bool)
            normalized = float(value) if real_scalar else float('nan')
            valid = (math.isfinite(normalized)
                     and (value > 0 if positive else value >= 0)
                     and (normalized > 0 if positive else normalized >= 0))
        except (OverflowError, TypeError, ValueError):
            valid = False
        if not valid:
            bound = 'positive' if positive else 'nonnegative'
            raise ValueError(f"{name} must be a finite {bound} real scalar (not bool)")
        if name == 'insulin_resistance_factor' and diabetes_type == 't1d' and value != 1:
            raise ValueError("insulin_resistance_factor must be 1 for t1d; its equations do not implement resistance scaling")
        # Downstream NumPy/JAX arithmetic requires a supported scalar, rather
        # than every numbers.Real implementation (for example Fraction).
        effective[name] = normalized
        overrides.pop(name)

    # Stage selection is type-specific. Keep these coefficients' existing
    # numeric handling; conversion/calibration owns their formulas and guards.
    calibration_inputs = ('k1', 'k2', 'Km0', 'ke1', 'ke2')
    if diabetes_type == 't1d':
        calibration_inputs += ('Gtb', 'kp2', 'kp3')
    else:
        calibration_inputs += ('HEb', 'm1', 'm2', 'm30', 'm4', 'k_a3')
        if diabetes_type == 't2d':
            calibration_inputs += ('Ib',)
        # No-pump has no controlled Ib target. T2D Gtb remains a late input
        # pending a contract for its calibration-seed versus initial-pool role.
    for name in calibration_inputs:
        if name in overrides:
            effective[name] = overrides.pop(name)
    return effective
