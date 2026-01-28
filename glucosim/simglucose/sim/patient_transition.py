from typing import Dict
import jax
import jax.numpy as jnp

from ..core.types import Action, PatientType
from ..core.params import PatientParams
from ..physiology.glucose_dynamics import t1d_rk4_step, t2d_rk4_step
from ..sim.sensor import cgm_measurement
from ..sim.realism import NoiseConfig


# ---------------------------------------------------------------------
# IOB / COB (unchanged, deterministic and realistic share these)
# ---------------------------------------------------------------------
@jax.jit
def patient_iob(state: Dict, insulin_kernel: jnp.ndarray) -> jnp.ndarray:
    """Calculates IOB from the insulin history kernels."""
    active_basal_iob = jnp.sum(state["basal_kernel"] * insulin_kernel)
    active_bolus_iob = jnp.sum(state["bolus_kernel"] * insulin_kernel)
    total_iob = active_basal_iob + active_bolus_iob
    return jnp.maximum(0.0, total_iob)


@jax.jit
def patient_cob(state: Dict) -> jnp.ndarray:
    """Calculates COB from the gut compartments of the ODE state."""
    total_cob_mg = state["patient_state"][0] + state["patient_state"][1] + state["patient_state"][2]
    total_cob_g = total_cob_mg / 1000.0
    return jnp.maximum(0.0, total_cob_g)


# ---------------------------------------------------------------------
# Realistic step (circadian + OU process + structured noise)
# ---------------------------------------------------------------------
@jax.jit
def patient_step(
    state: Dict, 
    patient_action: Action,
    params: PatientParams,
    key: jnp.ndarray, 
    cfg: NoiseConfig,
    eat_rate_per_min: float,
) -> Dict:

    patient_state = state['patient_state']
    meal_g = patient_action.meal
    bolus_U = patient_action.bolus
    # ex_frac is passed directly from step.py (intensity if active, 0.0 otherwise)
    ex_intensity = patient_action.exercise

    is_new_meal = (meal_g > 0) & (jnp.abs(state['planned_meal']) < 1e-3)
    planned_meal = state['planned_meal'] + meal_g
    # Consume at the per-minute rate derived from the controller interval to avoid overeating.
    to_eat = jnp.minimum(planned_meal, eat_rate_per_min)
    planned_meal -= to_eat
    # TODO: this is optional, if we want to use exercise model, we should uncomment this
    # planned_exercise_min = jnp.maximum(state['planned_exercise_min'] - 1.0, 0)
    
    is_eating = jnp.where(is_new_meal | (planned_meal > 0), 1, 0)
    last_Qsto = jax.lax.cond(is_new_meal,
                             lambda: patient_state[0] + patient_state[1],
                             lambda: state['last_Qsto'])
    last_foodtaken = jax.lax.cond(is_eating == 1,
                                  lambda: state['last_foodtaken'] + to_eat,
                                  lambda: 0.0)

    # Add meal mass explicitly to D1 (index 0) to ensure conservation (Impulse method)
    # This avoids numerical integration errors from treating meal as a rate over dt.
    # D1 is in mg, to_eat is in g.
    patient_state_ode = patient_state.at[0].add(to_eat * 1000.0)

    # The ODE (iir_exogenous_pmolkgmin) adds params.basal automatically.
    # Pass 0.0 for meal rate because we handled it as an impulse above.
    ode_action = jnp.array([0.0, bolus_U, ex_intensity], dtype=patient_state.dtype)
    BLOOD_GLUCOSE_DYNAMICS = t1d_rk4_step if params.diabetes_type == PatientType.t1d else t2d_rk4_step
    x_next, key, ou_state = BLOOD_GLUCOSE_DYNAMICS(
        patient_state_ode,
        1.0,
        ode_action,
        params,
        last_Qsto,
        last_foodtaken,
        state['t'],
        key,
        cfg,
        state['ou_state_dL']
    )

    cgm_prev = state['cgm_last']
    cgm_now, cgm_scale, key = cgm_measurement(
        Gp_mgkg=x_next[3], Vg_dL_per_kg=params.Vg,
        key=key, cfg=cfg, scale_state=state['cgm_scale']
    )
    cgm_now = jnp.where(jnp.isnan(cgm_now), cgm_prev, cgm_now)
    cgm_trend = cgm_now - cgm_prev

    return {
        'patient_state': x_next,
        'patient_type': params.diabetes_type,
        'scenario_meals': state['scenario_meals'],
        'basal_kernel': jnp.roll(state['basal_kernel'], 1).at[0].set(params.basal / 60.0),
        'bolus_kernel': jnp.roll(state['bolus_kernel'], 1).at[0].set(bolus_U),
        'planned_meal': planned_meal,
        'is_eating': is_eating,
        'to_eat': to_eat,
        'last_Qsto': last_Qsto,
        'last_foodtaken': jax.lax.cond(planned_meal > 0, lambda: last_foodtaken, lambda: 0.0),
        'last_meal_time': jax.lax.cond(is_new_meal, lambda: state['t'], lambda: state['last_meal_time']),
        'last_bolus_time': jax.lax.cond(bolus_U > 0, lambda: state['t'], lambda: state['last_bolus_time']),
        't': state['t'] + 1,
        'key': key,
        'ou_state_dL': ou_state,
        'cgm_last': cgm_now,
        'cgm_trend': cgm_trend,
        'cgm_scale': cgm_scale,
        # TODO: this is optional, if we want to use exercise model, we should uncomment this
        'planned_exercise_min': 0.0,
        'exercise_intensity': state.get('exercise_intensity', 0.0),
        'exercise_count': state.get('exercise_count', 0),
        'last_exercise_time': state.get('last_exercise_time', -999),
    }
