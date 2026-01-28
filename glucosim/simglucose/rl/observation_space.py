from typing import Dict, Any

import jax
import jax.numpy as jnp

from ..core.params import EnvParams, PatientParams

MAX_MEAL_PER_DAY = 7    # Support 7 small meals per day strategy
MAX_BOLUS_PER_DAY = 8   # Support pre-bolus for each meal + corrections


@jax.jit
def _build_observation(state: Dict[str, Any], env_params: 'EnvParams') -> jnp.ndarray:
    """
    15-D observation (still RNG-free):
      [ cgm, iob, cob, cgm_trend, time_sin, time_cos,
        time_since_meal, time_since_bolus, planned_meal_left (pending/ongoing buffer), last_exercise,
        meal_count_norm, bolus_count_norm,
        time_until_meal_norm, next_meal_size_norm, is_pre_bolus_window ]
    """
    p: PatientParams = env_params.patient_params

    cgm = state['cgm_last']
    # Use IOB decay kernel for IOB calculation, NOT the activity kernel
    # total iob = jnp.sum((state['basal_kernel'] + state['bolus_kernel']) * iob_kernel)    
    iob_kernel = jnp.asarray(env_params.iob_kernel, dtype=jnp.float32)
    # Only track Bolus IOB for the agent (Basal is constant/background)
    iob_bolus = jnp.sum(state['bolus_kernel'] * iob_kernel)
    cob = (state['patient_state'][0] + state['patient_state'][1] + state['patient_state'][2]) / 1000.0  # g

    cgm_trend = state['cgm_trend']  # mg/dL/min (maintain in your step code)

    t_day = state['t'] % 1440
    ang = (2.0 * jnp.pi * t_day) / 1440.0
    time_sin = jnp.sin(ang)
    time_cos = jnp.cos(ang)

    def _norm_since(last_t):
        return jnp.where(last_t < 0, 1.0, jnp.minimum((state['t'] - last_t) / 180.0, 1.0))
    time_since_meal  = _norm_since(state['last_meal_time'])
    time_since_bolus = _norm_since(state['last_bolus_time'])

    planned_meal_left_normalized = jnp.clip(state['planned_meal'] / p.max_meal_g, 0.0, 1.0)
    # TODO: this is optional, if we want to use exercise model, we should uncomment this
    # planned_ex_min_normalized = jnp.clip(state['planned_exercise_min'] / p.max_exercise_min, 0.0, 1.0)
    meal_count_norm = jnp.minimum(state['meal_count_daily'] / MAX_MEAL_PER_DAY, 1.0)
    bolus_count_norm = jnp.minimum(state['bolus_count_daily'] / MAX_BOLUS_PER_DAY, 1.0)

    # Look ahead to the next scheduled meal in the current day
    # Times are in minutes from midnight; invalid entries are negative.
    todays_schedule = state['scenario_meals'][state['t'] // 1440]
    meal_times = todays_schedule[:, 0]
    meal_amounts = todays_schedule[:, 1]
    minutes_into_day = state['t'] % 1440

    valid = meal_times >= 0.0
    upcoming = jnp.logical_and(valid, meal_times >= minutes_into_day)
    future_times = jnp.where(upcoming, meal_times, jnp.inf)
    next_idx = jnp.argmin(future_times)
    has_future = jnp.any(upcoming)

    next_time_min = jnp.where(has_future, future_times[next_idx], jnp.inf)
    next_amount_g = jnp.where(has_future, meal_amounts[next_idx], 0.0)
    time_until = jnp.maximum(0.0, next_time_min - minutes_into_day)

    time_until_meal_norm = jnp.where(
        has_future,
        jnp.minimum(time_until / 180.0, 1.0),  # 0 when imminent, 1 when >=3h away
        1.0,
    )
    next_meal_size_norm = jnp.clip(next_amount_g / p.max_meal_g, 0.0, 1.0)
    is_pre_bolus_window = jnp.where(
        has_future & (time_until >= 15.0) & (time_until <= 30.0),
        1.0,
        0.0,
    )

    return jnp.array([
        cgm, iob_bolus, cob, cgm_trend, time_sin, time_cos,
        time_since_meal, time_since_bolus, planned_meal_left_normalized, 
        # planned_ex_min_normalized,
        meal_count_norm, bolus_count_norm,
        time_until_meal_norm, next_meal_size_norm, is_pre_bolus_window
    ], dtype=jnp.float32)
