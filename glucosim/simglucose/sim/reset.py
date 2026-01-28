from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
from functools import partial

from ..core.params import EnvParams, PatientParams
from ..core.types import Action
from ..sim.scenario_gen import (
    create_multiday_scenario_jax,
    get_meal_profile_for_cohort,
)
from ..sim.sensor import cgm_measurement
from ..sim.step import mini_step

from ..rl.observation_space import _build_observation
from ..physiology.initialization import initialize_patient_state_history



# Warmup duration in minutes for basal-only burn-in (72h to reach a stable fasting state)
WARMUP_MINUTES = 72 * 60
USE_SCENARIO = True
# To debug blood glucose dynamics, set USE_SCENARIO to False
# With USE_SCENARIO = True, each patient has their own meal schedule, which is generated at reset time
# USE_SCENARIO = False

@partial(jax.jit, static_argnames=("warmup_steps",))
def _warmup_basal_only(state: dict, env_params: EnvParams, warmup_steps: int):
    """
    Run a basal-only burn-in (no meals, bolus, or exercise) to reach fasting steady state.
    """
    zero_action = Action(meal=0.0, bolus=0.0, exercise=0.0)

    def body(_, carry):
        new_carry, _, _, _, _, _ = mini_step(carry, zero_action, env_params)
        # Preserve full pytree structure by merging updates into the existing state
        carry = {**carry, **new_carry}
        return carry

    return jax.lax.fori_loop(0, warmup_steps, body, state)

def _build_base_state(
    env_params: EnvParams,
    tuned_state: jnp.ndarray,
    scenario_meals: jnp.ndarray,
    key: jnp.ndarray,
) -> Dict[str, Any]:
    x0 = tuned_state
    p: PatientParams = env_params.patient_params
    max_steps = env_params.simulation_minutes
    dia_steps = env_params.dia_steps

    ou_state_dL = jnp.array(0.0, dtype=jnp.float32)
    cgm_scale = jnp.array(1.0, dtype=jnp.float32)

    basal_history = initialize_patient_state_history(
        basal_rate_U_hr=p.basal,
        dt_mins=1.0,
        kernel_len=dia_steps
    )

    bg0_mgdl = x0[3] / jnp.maximum(p.Vg, 1e-6)
    bg0_mgdl = jnp.nan_to_num(bg0_mgdl, nan=p.Gb)

    cgm0_mgdl, cgm_scale_new, key2 = cgm_measurement(
        Gp_mgkg=x0[3], Vg_dL_per_kg=p.Vg, key=key, cfg=env_params.noise_config, scale_state=cgm_scale
    )

    return {
        "patient_state": x0,
        "patient_type": p.diabetes_type,
        "scenario_meals": scenario_meals,
        "basal_kernel": basal_history,
        "bolus_kernel": jnp.zeros((dia_steps,), dtype=jnp.float32),
        "planned_meal": jnp.array(0.0, dtype=jnp.float32),
        "is_eating": jnp.array(0, dtype=jnp.int32),
        "last_meal_time": jnp.array(-999, dtype=jnp.int32),
        "last_bolus_time": jnp.array(-999, dtype=jnp.int32),
        "last_hypo_time": jnp.array(-9999, dtype=jnp.int32),
        "last_exercise_time": jnp.array(-9999, dtype=jnp.int32),
        "planned_exercise_min": jnp.array(0.0, dtype=jnp.float32),
        "exercise_intensity": jnp.array(0.0, dtype=jnp.float32),
        "exercise_count": jnp.array(0, dtype=jnp.int32),
        "t": jnp.array(0, dtype=jnp.int32),
        "index": jnp.array(1, dtype=jnp.int32),
        "day_index": jnp.array(0, dtype=jnp.int32),
        "meal_count_daily": jnp.array(0, dtype=jnp.int32),
        "bolus_count_daily": jnp.array(0, dtype=jnp.int32),
        "ex_count_daily": jnp.array(0, dtype=jnp.int32),
        "last_Qsto": x0[0] + x0[1],
        "last_foodtaken": jnp.array(0.0, dtype=jnp.float32),
        "to_eat": jnp.array(0.0, dtype=jnp.float32),
        "BG_hist":  jnp.zeros((max_steps,), dtype=jnp.float32).at[0].set(bg0_mgdl),
        "CGM_hist": jnp.zeros((max_steps,), dtype=jnp.float32).at[0].set(jnp.nan_to_num(cgm0_mgdl, nan=bg0_mgdl)),
        "CHO_hist": jnp.zeros((max_steps,), dtype=jnp.float32),
        "insulin_hist": jnp.zeros((max_steps,), dtype=jnp.float32),
        "IOB_hist": jnp.zeros((max_steps,), dtype=jnp.float32),
        "COB_hist": jnp.zeros((max_steps,), dtype=jnp.float32),
        "key": key2,
        "ou_state_dL": ou_state_dL,
        "cgm_scale": cgm_scale_new,
        "cgm_last": jnp.array(jnp.nan_to_num(cgm0_mgdl, nan=bg0_mgdl), dtype=jnp.float32),
        "cgm_trend": jnp.array(0.0, dtype=jnp.float32),
    }


def _build_state_from_warm(
    env_params: EnvParams,
    warm_state: Dict[str, Any],
    scenario_meals: jnp.ndarray,
    key: jnp.ndarray,
) -> Dict[str, Any]:
    p: PatientParams = env_params.patient_params
    max_steps = env_params.simulation_minutes
    bg_warm = warm_state['patient_state'][3] / jnp.maximum(p.Vg, 1e-6)
    bg_warm = jnp.nan_to_num(bg_warm, nan=p.Gb)

    cgm_warm, cgm_scale_new, key2 = cgm_measurement(
        Gp_mgkg=warm_state['patient_state'][3],
        Vg_dL_per_kg=p.Vg,
        key=key,
        cfg=env_params.noise_config,
        scale_state=jnp.array(1.0, dtype=jnp.float32),
    )

    state = {
        **warm_state,
        "scenario_meals": scenario_meals,
        "t": jnp.array(0, dtype=jnp.int32),
        "index": jnp.array(1, dtype=jnp.int32),
        "day_index": jnp.array(0, dtype=jnp.int32),
        "meal_count_daily": jnp.array(0, dtype=jnp.int32),
        "bolus_count_daily": jnp.array(0, dtype=jnp.int32),
        "ex_count_daily": jnp.array(0, dtype=jnp.int32),
        "last_meal_time": jnp.array(-999, dtype=jnp.int32),
        "last_bolus_time": jnp.array(-999, dtype=jnp.int32),
        "last_exercise_time": jnp.array(-9999, dtype=jnp.int32),
        "last_hypo_time": jnp.array(-9999, dtype=jnp.int32),
        "BG_hist":  jnp.zeros((max_steps,), dtype=jnp.float32).at[0].set(bg_warm),
        "CGM_hist": jnp.zeros((max_steps,), dtype=jnp.float32).at[0].set(jnp.nan_to_num(cgm_warm, nan=bg_warm)),
        "CHO_hist": jnp.zeros((max_steps,), dtype=jnp.float32),
        "insulin_hist": jnp.zeros((max_steps,), dtype=jnp.float32),
        "IOB_hist": jnp.zeros((max_steps,), dtype=jnp.float32),
        "COB_hist": jnp.zeros((max_steps,), dtype=jnp.float32),
        "planned_meal": jnp.array(0.0, dtype=jnp.float32),
        "planned_exercise_min": jnp.array(0.0, dtype=jnp.float32),
        "exercise_intensity": jnp.array(0.0, dtype=jnp.float32),
        "exercise_count": jnp.array(0, dtype=jnp.int32),
        "is_eating": jnp.array(0, dtype=jnp.int32),
        "to_eat": jnp.array(0.0, dtype=jnp.float32),
        "last_foodtaken": jnp.array(0.0, dtype=jnp.float32),
        "key": key2,
        "ou_state_dL": jnp.array(0.0, dtype=jnp.float32),
        "cgm_scale": cgm_scale_new,
        "cgm_last": jnp.array(jnp.nan_to_num(cgm_warm, nan=bg_warm), dtype=jnp.float32),
        "cgm_trend": jnp.array(0.0, dtype=jnp.float32),
    }
    return state


@jax.jit
def _build_scenario(env_params: EnvParams, key: jnp.ndarray) -> jnp.ndarray:
    cohort_key = env_params.patient_name.split('#')[0]
    meal_mu, meal_sigma = env_params.meal_amount_mu, env_params.meal_amount_sigma
    meal_mu = jnp.asarray(meal_mu) if meal_mu is not None else get_meal_profile_for_cohort(cohort_key)[0]
    meal_sigma = jnp.asarray(meal_sigma) if meal_sigma is not None else get_meal_profile_for_cohort(cohort_key)[1]

    if USE_SCENARIO:
        return create_multiday_scenario_jax(
            key=key,
            amount_mu=meal_mu,
            amount_sigma=meal_sigma,
            num_days=max(1, env_params.simulation_minutes // 1440),
        )
    return jnp.zeros((12, 6, 2)).at[0].set(-1)


def reset(
    env_params: EnvParams,
    tuned_state: jnp.ndarray,
    key: jnp.ndarray,
    warm_state: Dict[str, Any] = None,
    return_warm_state: bool = False,
) -> Tuple[Dict[str, Any], jnp.ndarray]:
    """
    Jittable core that assumes 'tuned_state' is already computed on Python side.
    Returns (state, obs) and **writes cgm_last/cgm_trend** into state.
    """
    key, k_scn, k_meas = jax.random.split(key, 3)
    scenario_meals = _build_scenario(env_params, k_scn)

    if warm_state is None:
        base_state = _build_base_state(env_params, tuned_state, scenario_meals, k_meas)
        warm_state = _warmup_basal_only(
            {**base_state, "scenario_meals": jnp.zeros_like(scenario_meals)},
            env_params,
            warmup_steps=WARMUP_MINUTES,
        )
        warm_state = {**warm_state, "scenario_meals": scenario_meals}

    state = _build_state_from_warm(env_params, warm_state, scenario_meals, k_meas)
    obs = _build_observation(state, env_params)
    if return_warm_state:
        return warm_state, state, obs
    return state, obs
