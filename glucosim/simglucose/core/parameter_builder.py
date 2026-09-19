"""Host-side legacy parameter construction, separate from runtime parameter types.

Raw patient construction and direct adapter entry are distinct. This module
retains original source values, calibration order, stress scaling and late
factory overrides. Initialization tuning still belongs to physiology.initialization.
Compatibility functions import this module lazily to avoid a params/builder cycle.
"""
import dataclasses
import hashlib
import os
import logging
from typing import Optional

import numpy as np
import jax.numpy as jnp

from . import patient_loader, conversion
from . import presets
from .configuration import BuildOptions, extract_construction_options, extract_effective_inputs
from .params import PatientParams, EnvParams, NoiseConfig
from .types import PatientType
from ..physiology import calibration, kernels
from ..sim import scenario_gen

logger = logging.getLogger('glucosim.simglucose.core.params')


def _build_patient_from_overrides(patient_name, csv_path, diabetes_type,
                                  acceptance_probability, override_params):
    """Existing factory normalization shared by patient and environment entry points.

    The compatibility facade supplies its live acceptance default at each call.
    Covered effective inputs are separated before loading; other overrides
    retain their final replacement stage.
    """
    override_params = override_params.copy()
    # Resolve the documented default and reject unsupported types before CSV I/O.
    if diabetes_type is None:
        diabetes_type = "t1d"
    if not isinstance(diabetes_type, str) or diabetes_type not in ("t1d", "t2d", "t2d_no_pump"):
        raise ValueError(
            f"Invalid diabetes_type: {diabetes_type!r}. Must be 't1d', 't2d', or 't2d_no_pump'"
        )

    options = extract_construction_options(acceptance_probability, override_params)
    return build_patient_params(patient_name, csv_path, diabetes_type, options, override_params)


def build_patient_params(patient_name: str, csv_path: Optional[str], diabetes_type: str,
                         options: BuildOptions, override_params: dict) -> PatientParams:
    """Build from CSV; caller has resolved type and separated construction options."""

    override_params = override_params.copy()
    effective_overrides = extract_effective_inputs(diabetes_type, override_params)

    # Default CSV path if not provided
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, '..', 'params', 'vpatient_params.csv')

    # Load patient data from CSV
    patient_data = patient_loader.load_patient_parameters_from_csv(csv_path)

    if patient_name not in patient_data:
        available_patients = list(patient_data.keys())
        raise ValueError(f"Patient '{patient_name}' not found in CSV. Available patients: {available_patients}")

    # Get base parameters for the patient
    base_params = patient_data[patient_name].copy()
    # PD tuning: shrink glucose distribution volume and conserved masses
    if 'Vg' in base_params and np.isfinite(base_params['Vg']):
        base_params['Vg'] = float(base_params['Vg'])
    if 'Gpb' in base_params and np.isfinite(base_params['Gpb']):
        base_params['Gpb'] = float(base_params['Gpb'])
    if 'Gtb' in base_params and np.isfinite(base_params['Gtb']):
        base_params['Gtb'] = float(base_params['Gtb'])
    age_ranges = {
        'child': (0, 12),      # 0 to 12 years
        'adolescent': (13, 19), # 13 to 19 years
        'adult': (20, 80)      # 20 to 80 years (reasonable upper limit for adults)
    }
    age = 35
    HR0 = 70.0
    tau = 10.0
    # Customize parameters based on age
    if patient_name.split('#')[0] in age_ranges:
        min_age, max_age = age_ranges[patient_name.split('#')[0]]
        stable_seed = int(hashlib.md5(patient_name.encode("ascii", "ignore")).hexdigest()[:8], 16)
        age = min_age + (stable_seed % (max_age - min_age + 1))
        # HR0: resting heart rate, higher for younger ages
        HR0 = 70 + 50 * max(0, (18 - age) / 18)
        # tau_HR, tau_ex, tau_in: time constants, smaller (faster recovery) for younger ages
        tau = 10 - 0.2 * (20 - age) if age < 20 else 10 + 0.1 * (age - 20)
        tau_HR = tau
        tau_ex = tau
        tau_in = tau

    # Add default values for parameters not in CSV
    defaults = {
        # Eating behavior
        'eat_rate': 10.0,
        'meal_acceptance_prob': options.acceptance_probability,
        'bolus_acceptance_prob': options.acceptance_probability,
        'exercise_acceptance_prob': options.acceptance_probability,
        'V_G_L': (base_params['Vg'] * base_params['BW']) / 10.0,  # dL/kg to L
        'V_I_L': base_params['Vi'] * base_params['BW'],           #

        # Pump settings
        'use_pump': False,

        # Endogenous insulin secretion
        'beta_cell_function': 0.3,
        'insulin_resistance_factor': 2.5,

        # Safety windows
        'meal_safe_window': 60.0,  # minutes
        'bolus_safe_window': 60.0,  # minutes
        'exercise_safe_window': 1440.0,  # minutes (long enough to ignore)

        # Limits
        'max_meal_g': 80.0,
        'max_bolus_U': 80.0 / 10.0, # we maintain 1 U/h insulin covers 10g of carbs
        'max_exercise_min': 90.0,

        # Basal rate
        'basal': 0.0,  # U/h

        # Exercise
        'age': age,
        'HR0': HR0,
        'alpha_HR': 0.01,
        'n_power': 1.0,
        'c1': 0.01,
        'c2': 0.01,
        'tau_HR': tau_HR,
        'tau_ex': tau_ex,
        'tau_in': tau_in,
        'beta_ex': 0.2,
        'alpha_QE': 0.004,
    }

    # Merge defaults with CSV data
    for key, default_value in defaults.items():
        if key not in base_params:
            base_params[key] = default_value

    # Ensure patient identification is present before creating dataclass
    base_params['diabetes_type'] = getattr(PatientType, diabetes_type)
    # Convert base parameters to PatientParams for proper T2D adaptation
    temp_params = PatientParams(**base_params)
    if diabetes_type == "t1d":
        patient_params = build_t1d(temp_params, options, effective_overrides=effective_overrides)
    elif diabetes_type == "t2d":
        patient_params = build_t2d(temp_params, None, options, effective_overrides=effective_overrides)
    elif diabetes_type == "t2d_no_pump":
        patient_params = build_t2d_no_pump(temp_params, None, options, effective_overrides=effective_overrides)
    else:
        raise ValueError(f"Invalid diabetes_type: {diabetes_type}. Must be 't1d', 't2d', or 't2d_no_pump'")

    # Only unlisted/behavioral overrides remain; apply them once at the end.
    if override_params:
        patient_params = dataclasses.replace(patient_params, **override_params)
        logger.info(f"Applied {len(override_params)} parameter overrides: {list(override_params.keys())}")

    # Reject invalid exercise time scales before tracing the ODE with JAX.
    for name in ("tau_HR", "tau_ex", "tau_in", "c2", "HR0", "alpha_HR", "n_power"):
        value = getattr(patient_params, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(patient_params.c1) or patient_params.c1 < 0:
        raise ValueError("c1 must be finite and nonnegative")

    logger.info(f"Applied {diabetes_type} adaptations to patient {patient_name}")

    return patient_params


def build_t1d(base_params: PatientParams, options: BuildOptions, *,
              effective_overrides: Optional[dict] = None) -> PatientParams:
    """Adapt the exact supplied object, without loading or rebuilding its source."""
    autobalance_enabled = options.autobalance_enabled
    autobalance_basal_scale = options.autobalance_basal_scale
    autobalance_hepatic_scale = options.autobalance_hepatic_scale
    carb_absorption_scale = options.carb_absorption_scale
    insulin_sensitivity_scale = options.insulin_sensitivity_scale
    eat_rate_scale = options.eat_rate_scale

    params = base_params
    if effective_overrides:
        params = dataclasses.replace(params, **effective_overrides)

    logger.info("Adapting parameters for Type 1 Diabetes")

    # T1D-specific adjustments
    params = dataclasses.replace(
        params, **presets.t1d_overrides(params, options.acceptance_probability)
    )
    # Autobalance the basal rate, with optional weakening to expose harsher dynamics
    if autobalance_enabled:
        params = calibration.autobalance_basal_t1d(
            params,
            basal_scale=autobalance_basal_scale,
            hepatic_scale=autobalance_hepatic_scale,
        )

    # Gut appearance tweaks for calibration (stronger absorption => sharper spikes)
    if carb_absorption_scale != 1.0:
        params = dataclasses.replace(
            params,
            kmax=params.kmax * carb_absorption_scale,
            kabs=params.kabs * carb_absorption_scale,
        )
    if eat_rate_scale != 1.0:
        params = dataclasses.replace(
            params,
            eat_rate=params.eat_rate * eat_rate_scale,
        )

    # Insulin action scaling for harsher “no bolus” behaviour
    if insulin_sensitivity_scale != 1.0:
        params = dataclasses.replace(
            params,
            Vmx=params.Vmx * insulin_sensitivity_scale,
        )

    return params


def build_t2d(base_params: PatientParams, config: Optional[dict],
              options: BuildOptions, *,
              effective_overrides: Optional[dict] = None) -> PatientParams:
    """Preserve the legacy recipe starting from the exact supplied parameters."""
    carb_absorption_scale = options.carb_absorption_scale
    insulin_sensitivity_scale = options.insulin_sensitivity_scale
    eat_rate_scale = options.eat_rate_scale

    logger.info("Adapting parameters for Type 2 Diabetes (with pump)")

    # Default T2D scaling factors
    default_config = presets.t2d_scaling_defaults()
    config = config or default_config

    # Start with original T1D parameters for proper conversion
    temp_params = base_params

    # Apply physiological scaling before T2D conversion
    temp_params = dataclasses.replace(
        temp_params,
        BW=temp_params.BW * config['BW_factor'],
        Ib=temp_params.Ib * config['Ib_factor'],
        Ipb=temp_params.Ipb * config['Ipb_factor'],
        Ilb=temp_params.Ilb * config['Ipb_factor'],  # Scale liver insulin to maintain equilibrium ratio
        HEb=temp_params.HEb * config['HEb_factor'],
        # Select resistance before converting the unscaled insulin-effect gains.
        insulin_resistance_factor=presets.T2D_INSULIN_RESISTANCE,
    )
    # Factory inputs name the effective patient, not the pre-factor source.
    # The original source gains have not been divided at this point.
    if effective_overrides:
        temp_params = dataclasses.replace(temp_params, **effective_overrides)

    # Convert to T2D format using comprehensive conversion logic
    t2d_params = conversion.patient_to_t2d_params(
        temp_params, use_dynamic_HE=False,
        effective_resistance=(effective_overrides or {}).get('insulin_resistance_factor'),
    )
    basal_rate = calibration.pump_basal_rate_t2d(t2d_params)

    # Apply T2D-specific behavioral and physiological parameters
    t2d_params = dataclasses.replace(
        t2d_params, **presets.t2d_overrides(
            basal_rate, options.acceptance_probability,
            resistance_factor=t2d_params.insulin_resistance_factor,
        )
    )

    t2d_params = calibration.calibrate_t2d_pump(t2d_params)

    # Safety validation for T2D
    assert 0.2 <= t2d_params.beta_cell_function <= 0.3, f"Invalid T2D beta-cell function: {t2d_params.beta_cell_function}"
    if effective_overrides and 'insulin_resistance_factor' in effective_overrides:
        assert np.isfinite(t2d_params.insulin_resistance_factor) and t2d_params.insulin_resistance_factor > 0
    else:
        assert 2.0 <= t2d_params.insulin_resistance_factor <= 3.0, f"Invalid T2D insulin resistance: {t2d_params.insulin_resistance_factor}"
    assert t2d_params.BW > 0, f"Invalid body weight after T2D adjustment: {t2d_params.BW}"

    logger.debug(f"T2D adaptations: beta_cell={t2d_params.beta_cell_function:.2f}, "
                f"IR_factor={t2d_params.insulin_resistance_factor:.1f}, BW={t2d_params.BW:.1f}")

    # Optional scaling knobs for generalization stress tests
    if carb_absorption_scale != 1.0:
        t2d_params = dataclasses.replace(
            t2d_params,
            kmax=t2d_params.kmax * carb_absorption_scale,
            kabs=t2d_params.kabs * carb_absorption_scale,
        )
    if eat_rate_scale != 1.0:
        t2d_params = dataclasses.replace(
            t2d_params,
            eat_rate=t2d_params.eat_rate * eat_rate_scale,
        )
    if insulin_sensitivity_scale != 1.0:
        t2d_params = dataclasses.replace(
            t2d_params,
            Vmx=t2d_params.Vmx * insulin_sensitivity_scale,
        )

    return t2d_params


def build_t2d_no_pump(base_params: PatientParams, config: Optional[dict],
                      options: BuildOptions, *,
                      effective_overrides: Optional[dict] = None) -> PatientParams:
    """Preserve the legacy recipe starting from the exact supplied parameters."""
    carb_absorption_scale = options.carb_absorption_scale
    insulin_sensitivity_scale = options.insulin_sensitivity_scale
    eat_rate_scale = options.eat_rate_scale

    # Apply scaling only after the final no-pump basal calibration below.
    params = build_t2d(
        base_params,
        config,
        dataclasses.replace(options, carb_absorption_scale=1.0,
                            insulin_sensitivity_scale=1.0, eat_rate_scale=1.0),
        effective_overrides=effective_overrides,
    )

    logger.info("Adapting parameters for Type 2 Diabetes (no pump)")

    # No-pump specific adjustments
    resistance = (effective_overrides or {}).get(
        'insulin_resistance_factor', presets.T2D_NO_PUMP_INSULIN_RESISTANCE
    )
    params = dataclasses.replace(
        params, **presets.t2d_no_pump_overrides(
            base_params, options.acceptance_probability, resistance_factor=resistance
        )
    )

    params = calibration.calibrate_t2d_no_pump(params)

    # Safety validation for T2D no-pump
    assert not params.use_pump, "SAFETY: T2D no-pump must not use pump"
    if effective_overrides and 'insulin_resistance_factor' in effective_overrides:
        assert np.isfinite(params.insulin_resistance_factor) and params.insulin_resistance_factor > 0
    else:
        assert params.insulin_resistance_factor >= 2.5, f"T2D no-pump IR factor too low: {params.insulin_resistance_factor}"
    assert params.max_bolus_U <= 25.0, f"Unsafe max bolus for T2D no-pump: {params.max_bolus_U}"

    logger.debug(f"T2D no-pump adaptations: IR_factor={params.insulin_resistance_factor:.1f}, "
                f"max_bolus={params.max_bolus_U:.1f}U, adherence={params.bolus_acceptance_prob:.2f}")

    # Apply scaling once, after the no-pump autobalance step.
    if carb_absorption_scale != 1.0:
        params = dataclasses.replace(
            params,
            kmax=params.kmax * carb_absorption_scale,
            kabs=params.kabs * carb_absorption_scale,
        )
    if eat_rate_scale != 1.0:
        params = dataclasses.replace(
            params,
            eat_rate=params.eat_rate * eat_rate_scale,
        )
    if insulin_sensitivity_scale != 1.0:
        params = dataclasses.replace(
            params,
            Vmx=params.Vmx * insulin_sensitivity_scale,
        )

    return params


def build_env_params(patient_name, csv_path, diabetes_type, simulation_minutes,
                     sample_time, acceptance_probability, patient_overrides) -> EnvParams:
    """Own environment assembly while preserving the public factory's stage order."""
    logger.info(f"Creating environment parameters for patient: {patient_name}")

    cohort_key = patient_name.split('#')[0].lower()
    meal_mu, meal_sigma = scenario_gen.get_meal_profile_for_cohort(cohort_key)

    # Create patient-specific parameters
    patient_params = _build_patient_from_overrides(
        patient_name=patient_name,
        csv_path=csv_path,
        diabetes_type=diabetes_type,
        acceptance_probability=acceptance_probability,
        override_params=patient_overrides
    )

    # Create other parameter structures
    dia_steps = 360 # 6 hours * 60 min/hr
    noise_config = NoiseConfig()

    # dt_mins=1 because the simulation kernel resolution is 1 minute
    # duration_hours=6 matches dia_steps=360
    iob_decay_kernel, insulin_act_kernel = kernels.create_insulin_kernel(dt_mins=1, duration_hours=6)

    # Convert to numpy for storage in EnvParams (JAX arrays are also fine, but casting ensures consistency)
    iob_decay_kernel = np.array(iob_decay_kernel)
    insulin_act_kernel = np.array(insulin_act_kernel)

    insulin_kernel_5 = np.array(insulin_act_kernel).reshape((-1, 5)).sum(axis=1)

    # Create environment parameters
    env_params = EnvParams(
        patient_params=patient_params,
        sample_time=sample_time,
        simulation_minutes=simulation_minutes,
        dia_steps=dia_steps,
        insulin_kernel=tuple(insulin_act_kernel),
        insulin_kernel_5=tuple(insulin_kernel_5),
        iob_kernel=tuple(iob_decay_kernel),
        noise_config=noise_config,
        patient_name=patient_name,
        meal_amount_mu=jnp.asarray(meal_mu, dtype=jnp.float32),
        meal_amount_sigma=jnp.asarray(meal_sigma, dtype=jnp.float32),
    )

    logger.info(f"Environment parameters created successfully for {patient_name}")
    return env_params
