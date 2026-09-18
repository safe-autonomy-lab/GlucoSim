"""Immutable host-side options for the legacy parameter construction contract.

These options preserve historical meanings, including ignored T2D autobalance
options and post-calibration stress scales. They are not part of the JAX tree.
Acceptance must be supplied at call time by the compatibility entry point.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LegacyBuildOptions:
    acceptance_probability: float
    autobalance_enabled: bool = True
    autobalance_basal_scale: float = 1.0
    autobalance_hepatic_scale: float = 1.0
    carb_absorption_scale: float = 1.0
    insulin_sensitivity_scale: float = 1.0
    eat_rate_scale: float = 1.0
