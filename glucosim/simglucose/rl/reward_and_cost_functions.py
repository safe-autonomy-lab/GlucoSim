from functools import partial
import jax
import jax.numpy as jnp

from ..physiology.kernels import create_carb_gamma_kernel
from ..rl.observation_space import MAX_MEAL_PER_DAY, MAX_BOLUS_PER_DAY

# =============================================================================
# CONSTANTS
# =============================================================================
REWARD_SCALE = 0.5
DELTA_RISK_SCALE = 1.0
RAW_COST_SCALE_FOR_CONSTRAINT = 0.35

STRUCTURAL_WEIGHT = 0.1
TARGET_BOLUS_PER_DAY = 5.0
TARGET_MEAL_PER_DAY = 3.0
PACE_SLACK = 1.0
PROGRESSIVE_WEIGHT = 0.001
MIN_BOLUS_SPACING = 30.0
SPACING_WEIGHT = 0.01

ACTION_COST_BOLUS_PER_U = 0.005
ACTION_COST_MEAL_PER_G = 0.001
EVENT_COST_BOLUS = 0.005
EVENT_COST_MEAL = 0.005

RISK_WAIVER_THRESHOLD = 1.5
WEIGHT_HYPO_MILD = 3.0
WEIGHT_HYPO_SEVERE = 10.0
WEIGHT_HYPER_MILD = 1.0  # Restored
WEIGHT_HYPER_SEVERE = 4.0
MOMENTUM_WEIGHT = 0.015


def _resample_to_dt(arr: jnp.ndarray, dt_minutes: float) -> jnp.ndarray:
    factor = int(round(dt_minutes))
    if factor <= 1:
        return arr
    usable = (arr.shape[0] // factor) * factor
    trimmed = arr[:usable]
    return trimmed.reshape((usable // factor, factor)).sum(axis=1)


@jax.jit
def calculate_graded_danger_cost_raw(trace: jnp.ndarray, dt: float = 1.0) -> float:
    """
    Returns RAW cost magnitude.
    SAFE MOMENTUM VERSION with Mild Hyper restored.
    """
    safe_dt = jnp.maximum(dt, 1.0)

    if trace.shape[0] > 1:
        deltas = trace[1:] - trace[:-1]
        raw_velocity = deltas / safe_dt
        velocity = jnp.clip(raw_velocity, -10.0, 10.0)
        velocity = jnp.append(velocity, velocity[-1])
    else:
        velocity = jnp.zeros_like(trace)

    # Momentum (High + Rising)
    momentum_threshold = 160.0
    high_excess = jnp.maximum(0.0, trace - momentum_threshold)
    rising_rate = jnp.maximum(0.0, velocity)
    momentum_cost = high_excess * rising_rate * MOMENTUM_WEIGHT

    # Static penalties
    mild_low_excess = jnp.maximum(0.0, 70.0 - trace)
    severe_low_excess = jnp.maximum(0.0, 54.0 - trace)
    
    # Restored: Mild Hyper (Area above 180) to prevent hovering
    mild_high_excess = jnp.maximum(0.0, trace - 180.0)
    
    severe_high_excess = jnp.maximum(0.0, trace - 250.0)

    cost_elements = (
        WEIGHT_HYPO_MILD * (mild_low_excess / 20.0)
        + WEIGHT_HYPO_SEVERE * ((severe_low_excess / 20.0) ** 2)
        + 1.0 * (mild_high_excess / 50.0) # Scaled down mild hyper cost
        + momentum_cost
        + WEIGHT_HYPER_SEVERE * ((severe_high_excess / 50.0) ** 2)
        + 0.1 * jnp.maximum(0.0, -(velocity + 2.0))
    )

    total_cost = jnp.mean(cost_elements)
    # Clamp max cost to 50.0 to prevent NaN/Infinity
    return jnp.minimum(total_cost, 50.0)


@partial(jax.jit, static_argnames=("dt", "forecast_steps", "cob_kernel_len"))
def jax_forecast_reward_function(
    current_bg: float,
    iob_kernel: jnp.ndarray,
    cob_states: jnp.ndarray,
    insulin_kernel: jnp.ndarray,
    action_taken: jnp.ndarray,
    isf: float,
    beta_cell_function: float,
    Gb: float,
    BW: float,
    Vg: float,
    dt: float,
    forecast_steps: int,
    cob_kernel_len: int,
    meal_count: int,
    bolus_count: int,
    time_since_bolus: float = 1440.0,
    time_since_meal: float = 1440.0,
    minutes_into_day: float = 720.0,
    **kwargs,
):
    basal_u_per_min, bolus_u, meal_g, exercise_frac = action_taken

    # 1. FORECASTS
    insulin_kernel_dt = insulin_kernel
    iob_kernel_dt = iob_kernel
    phi = 0.35
    csf_mgdl_per_g = phi * (1000.0 / (BW * Vg + 1e-6))
    k_endo_per_min = 0.02
    k_ex_mgdl_per_min = 0.3
    current_iob = jnp.sum(iob_kernel_dt)

    new_insulin_u = bolus_u + basal_u_per_min * dt
    dose_series = iob_kernel_dt + jnp.zeros_like(iob_kernel_dt).at[0].set(new_insulin_u)
    ins_act = jnp.convolve(dose_series, insulin_kernel_dt)[:forecast_steps] * isf

    # Assume cob_states is mg, meal_g is g.
    cob_states_g = cob_states / 1000.0
    
    # If cob_kernel_len is steps, don't multiply by dt in kernel gen
    # Assuming cob_kernel_len is in STEPS based on previous context.
    new_carb = _resample_to_dt(create_carb_gamma_kernel(int(cob_kernel_len * dt), meal_g), dt)
    carb_series_g = cob_states_g + new_carb
    carb_eff = carb_series_g[:forecast_steps] * csf_mgdl_per_g

    def step_fn(bg, i):
        endo = beta_cell_function * k_endo_per_min * jnp.maximum(bg - Gb, 0.0) * dt
        ex = k_ex_mgdl_per_min * exercise_frac * dt
        new_bg = bg + carb_eff[i] - (ins_act[i] + endo + ex)
        # Safety clamp
        return jnp.clip(new_bg, 40.0, 600.0), new_bg
        
    _, trace_action = jax.lax.scan(step_fn, current_bg, jnp.arange(forecast_steps))

    # Baseline forecast (No action)
    new_insulin_no = basal_u_per_min * dt
    dose_series_no = iob_kernel_dt + jnp.zeros_like(iob_kernel_dt).at[0].set(new_insulin_no)
    ins_act_no = jnp.convolve(dose_series_no, insulin_kernel_dt)[:forecast_steps] * isf
    carb_eff_no = cob_states_g[:forecast_steps] * csf_mgdl_per_g

    def step_fn_no(bg, i):
        endo = beta_cell_function * k_endo_per_min * jnp.maximum(bg - Gb, 0.0) * dt
        ex = 0.0
        new_bg = bg + carb_eff_no[i] - (ins_act_no[i] + endo + ex)
        return jnp.clip(new_bg, 40.0, 600.0), new_bg
        
    _, trace_baseline = jax.lax.scan(step_fn_no, current_bg, jnp.arange(forecast_steps))

    # 2. COSTS & DELTA RISK
    cost_action_raw = calculate_graded_danger_cost_raw(trace_action, dt)
    cost_baseline_raw = calculate_graded_danger_cost_raw(trace_baseline, dt)
    delta_risk_raw = cost_baseline_raw - cost_action_raw

    # 3. FLAGS
    is_bolus = bolus_u > 0
    is_meal = meal_g > 0
    is_ex = exercise_frac > 0
    is_inaction = (~is_bolus) & (~is_meal) & (~is_ex)

    is_urgent = delta_risk_raw > RISK_WAIVER_THRESHOLD
    is_healthy_state = (current_bg >= 70.0) & (current_bg <= 160.0)

    # 4. DELTA RISK GATING (Dual Gate)
    baseline_min = jnp.min(trace_baseline)
    treating_low = (current_bg < 90.0) | (baseline_min < 80.0)

    delta_risk_effective = delta_risk_raw
    # Gate 1: Meals only earn positive risk if treating low
    delta_risk_effective = jnp.where(
        is_meal & (~treating_low),
        jnp.minimum(delta_risk_effective, 0.0),
        delta_risk_effective,
    )
    # Gate 2: Don't punish safe meals
    delta_risk_effective = jnp.where(
        is_meal & is_healthy_state,
        jnp.maximum(delta_risk_effective, 0.0),
        delta_risk_effective
    )
    # Gate 3: Exercise
    delta_risk_effective = jnp.where(
        is_ex & (current_bg < 140.0),
        jnp.minimum(delta_risk_effective, 0.0),
        delta_risk_effective
    )

    # 5. PENALTIES
    frac_day = jnp.clip(minutes_into_day / 1440.0, 0.0, 1.0)
    pace_bolus = TARGET_BOLUS_PER_DAY * frac_day + PACE_SLACK
    pace_meal = TARGET_MEAL_PER_DAY * frac_day + PACE_SLACK

    next_b = bolus_count + jnp.where(is_bolus, 1, 0)
    next_m = meal_count + jnp.where(is_meal, 1, 0)
    excess_b = jnp.maximum(0.0, next_b - pace_bolus)
    excess_m = jnp.maximum(0.0, next_m - pace_meal)

    progressive_penalty_raw = PROGRESSIVE_WEIGHT * (excess_b ** 2 + excess_m ** 2)

    friction_raw = (
        ACTION_COST_BOLUS_PER_U * bolus_u +
        ACTION_COST_MEAL_PER_G * meal_g +
        EVENT_COST_BOLUS * jnp.where(is_bolus, 1.0, 0.0) +
        EVENT_COST_MEAL * jnp.where(is_meal, 1.0, 0.0)
    )
    friction_raw += jnp.where(
        is_bolus & (current_iob > 3.5),
        0.03 * (current_iob - 3.5),
        0.0
    )

    bolus_too_soon = time_since_bolus < MIN_BOLUS_SPACING
    spacing_penalty_raw = jnp.where(is_bolus & bolus_too_soon, SPACING_WEIGHT, 0.0)

    bolus_over = next_b > MAX_BOLUS_PER_DAY
    meal_over = next_m > MAX_MEAL_PER_DAY
    structural_penalty_raw = (
        STRUCTURAL_WEIGHT * jnp.where(bolus_over, (next_b - MAX_BOLUS_PER_DAY)**2, 0.0) +
        STRUCTURAL_WEIGHT * jnp.where(meal_over, (next_m - MAX_MEAL_PER_DAY)**2, 0.0)
    )

    # 6. WAIVERS (Fixed Logic)
    # Urgency
    progressive_penalty = jnp.where(is_urgent, progressive_penalty_raw * 0.2, progressive_penalty_raw)
    spacing_penalty = jnp.where(is_urgent, spacing_penalty_raw * 0.3, spacing_penalty_raw)

    # Meal: Zero out friction for meals to encourage "taking the action" if needed
    friction = jnp.where(is_meal & is_healthy_state, 0.0, friction_raw)
    structural_penalty = jnp.where(is_meal & is_healthy_state, 0.0, structural_penalty_raw)

    # Healthy
    friction = jnp.where(is_healthy_state & (~is_meal), friction * 0.1, friction)
    structural_penalty = jnp.where(is_healthy_state & (~is_meal), structural_penalty * 0.1, structural_penalty)
    progressive_penalty = jnp.where(is_healthy_state, progressive_penalty * 0.1, progressive_penalty)
    spacing_penalty = jnp.where(is_healthy_state, spacing_penalty * 0.1, spacing_penalty)

    # 7. SURVIVAL (Inaction Gated)
    bg_safe = (current_bg >= 70.0) & (current_bg <= 180.0)
    bg_perfect = (current_bg >= 90.0) & (current_bg <= 140.0)
    
    reward_safe = jnp.where(bg_safe, 0.1, 0.0)
    reward_perfect = jnp.where(bg_perfect, 0.2, 0.0)
    
    # Only pay survival drip if agent is NOT acting (true passive reward)
    survival_drip = jnp.where(is_inaction, reward_safe + reward_perfect, 0.0)

    high_inaction_pen = jnp.where(
        is_inaction & (current_bg > 180.0),
        0.005 * (current_bg - 180.0),
        0.0
    )

    # 8. FINAL ASSEMBLY
    raw_reward = (
        (DELTA_RISK_SCALE * delta_risk_effective)
        + survival_drip
        - friction
        - progressive_penalty
        - spacing_penalty
        - high_inaction_pen
        - structural_penalty
    )

    reward = raw_reward * REWARD_SCALE

    # =========================================================================
    # 9. COST CALCULATION
    # =========================================================================
    # If we are EATING and HEALTHY, use the BASELINE cost (what would happen without meal)
    # instead of the Action cost (which includes meal spike).
    # This prevents the constraint from panicking about the food.
    cost_to_use = jnp.where(
        is_meal & is_healthy_state,
        cost_baseline_raw,  # Ignore the meal spike in the safety cost
        cost_action_raw     # Otherwise use real risk
    )
    
    cost = cost_to_use * RAW_COST_SCALE_FOR_CONSTRAINT

    reward_info = {
        "delta_risk": delta_risk_effective,
        "cost_action_raw": cost_action_raw,
        "friction": friction,
        "survival_drip": survival_drip,
    }

    cost_info = {
        "cost_raw": cost_action_raw,
        "cost_scaled": cost,
    }

    return reward, cost, reward_info, cost_info

# TODO: Implement FDA approved sim glucose reward and cost function, cost function based on MAGE or CV!. This will help us to jusfiy
# def reward_function_simple(
#     current_bg: float,
#     iob_kernel: jnp.ndarray,
#     cob_states: jnp.ndarray,
#     insulin_kernel: jnp.ndarray,
#     action_taken: jnp.ndarray,
#     isf: float,
#     beta_cell_function: float,
#     Gb: float,
#     BW: float,
#     Vg: float,
#     dt: float,
# ):
#     return jax_forecast_reward_function(