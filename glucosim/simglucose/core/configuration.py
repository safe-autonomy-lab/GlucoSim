"""Scalar construction options and factory keyword extraction.

These options preserve historical meanings, including ignored T2D autobalance
options and post-calibration stress scales. They are not part of the JAX tree.
Acceptance must be supplied at call time by the compatibility entry point.
T2D factors belong to the adaptation boundary; late patient overrides belong
to the factory. Neither mapping is stored in the options object. Values are
not coerced or newly validated here; legacy consumption rules still apply.
"""
from dataclasses import dataclass


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
