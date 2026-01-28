from typing import Dict

import jax as jax
import jax.numpy as jnp

from ..sim.patient_transition import patient_step, patient_iob, patient_cob
from ..core.params import EnvParams, PatientParams
from ..rl.observation_space import MAX_MEAL_PER_DAY, MAX_BOLUS_PER_DAY, _build_observation
from ..rl.reward_and_cost_functions import jax_forecast_reward_function
from ..sim.realism import NoiseConfig
from ..core.types import Action, ControllerAction

BASE_ISF_MGDL_PER_U = 50.0  # Nominal mg/dL drop per unit, scaled by insulin resistance
# Bolus control guards to reduce rapid repeat dosing when BG is already OK
BOLUS_BG_MIN = 70.0  # mg/dL; block bolus below this
# the simulator purpose is to learn based on the blood glucose... reward should handle frequent recommendations
BOLUS_WINDOW_MULT = 1.0  # widen safe window to slow successive boluses
MEAL_WINDOW_MULT = 1.0  # widen safe window to slow successive meals
_DOWNSAMPLE_FACTOR = 5  # match the 5-minute control interval for reward proxy
REWARD_FUNCTIONN = jax_forecast_reward_function

# Aggregate minute-resolution kernels into 5-minute bins for the reward proxy
def _downsample_sum(arr: jnp.ndarray, factor: int = _DOWNSAMPLE_FACTOR) -> jnp.ndarray:
    new_len = arr.shape[0] // factor
    return arr.reshape((new_len, factor)).sum(axis=1)

# Mini-step function
@jax.jit
def mini_step(state: dict, patient_action: Action, env_params: EnvParams, skip_scenario_meal: bool = False) -> tuple:
    """
    A clean, JIT-compiled function that calls the dedicated patient_step
    function to update the patient's state.
    """
    patient_params: PatientParams = env_params.patient_params
    noise_config: NoiseConfig = env_params.noise_config
    # Scale the per-step eat_rate down to the per-minute simulator resolution.
    eat_rate_per_min = jnp.asarray(patient_params.eat_rate / float(env_params.sample_time), dtype=jnp.float32)
    key, subkey_step = jax.random.split(state['key'], 2)
    # Scenarios here
    current_total_minutes = state['t']
    current_day_index = current_total_minutes // 1440 # 1440 minutes in a day
    minutes_into_day = current_total_minutes % 1440
    # Select the meal schedule for the CURRENT day from the pre-generated multi-day plan
    todays_schedule = state['scenario_meals'][current_day_index]
    meal_times = todays_schedule[:, 0]
    meal_amounts = todays_schedule[:, 1]

    # Check if a meal is scheduled for this minute of this day
    mask = jnp.round(meal_times) == minutes_into_day
    scenario_meal = jnp.sum(jnp.where(mask, meal_amounts, 0.0))
    scenario_meal = jnp.where(skip_scenario_meal, 0.0, scenario_meal)

    # If you want to use the pump, you can wrap the basal with the pump
    # But you should be extra careful for the unit conversion
    # Current parameters are already designed to convert to fit the simulator
    bolus_U = patient_action.bolus
    ex_intensity = patient_action.exercise
    # originally just max.
    # If both scenario_meal and patient_action.meal are > 0, sum them, else take the maximum.
    both_meals_positive = jnp.logical_and(scenario_meal > 0, patient_action.meal > 0)
    meal_g = jnp.where(
        both_meals_positive,
        patient_action.meal,
        jnp.maximum(scenario_meal, patient_action.meal)
    )

    patient_action_for_ode = Action(meal=meal_g, bolus=bolus_U, exercise=ex_intensity)
    new_patient_state = patient_step(
        state=state,
        patient_action=patient_action_for_ode,
        params=patient_params,
        key=subkey_step,
        cfg=noise_config,
        eat_rate_per_min=eat_rate_per_min
    )
    # Extract results from the new patient state
    BG = new_patient_state['patient_state'][3] / patient_params.Vg
    CGM = new_patient_state['cgm_last']
    # Update RNG chain in the returned state
    new_patient_state = {**new_patient_state, 'key': key} # master RNG for next call
    basal_delivered = patient_params.basal / 60.0 # Convert U/h to U/min
    bolus_delivered = bolus_U
    return new_patient_state, basal_delivered, bolus_delivered, BG, CGM, scenario_meal

# Full step function
@jax.jit
def step(state: Dict, action: ControllerAction, env_params: EnvParams, key: jnp.ndarray) -> tuple:
    keys = jax.random.split(key, 7)
    key = keys[0]
    meal_accept_key, bolus_accept_key, exercise_accept_key, meal_noise_key, bolus_noise_key, ex_noise_key = keys[1:]

    sample_time = env_params.sample_time
    simulation_minutes = env_params.simulation_minutes
    patient_params = env_params.patient_params
    # Widen bolus safe window to discourage rapid repeat dosing
    bolus_safe_window = patient_params.bolus_safe_window * BOLUS_WINDOW_MULT
    meal_safe_window = patient_params.meal_safe_window * MEAL_WINDOW_MULT
    meal_safe_window = jnp.where(meal_safe_window <= 0.0, patient_params.meal_safe_window, meal_safe_window)

    # Meal Acceptance Logic
    meal_prob_check = jax.random.uniform(meal_accept_key) < patient_params.meal_acceptance_prob
    meal_time_check = (state['t'] - state['last_meal_time']) > meal_safe_window
    # Cancel meal if CGM is already high to avoid compounding spikes
    # This is natural behavior of the patient, they will not eat if they are already high
    cgm_est = state['CGM_hist'][jnp.maximum(state['index'] - 1, 0)]
    meal_cancel_check = cgm_est < 200.0
    # Hypo override: if BG is low, allow the meal regardless of other gates
    hypo_override = (cgm_est < 70.0) & (action.meal > 0.0)
    meal_accepted = jnp.logical_and(meal_prob_check, jnp.logical_and(meal_time_check, meal_cancel_check))
    positive_meal_check = action.meal > 0.
    meal_accepted = jax.lax.cond(positive_meal_check, lambda: meal_accepted, lambda: False)

    # --- NEW LOGIC: Eating vs Exercise Exclusion ---
    # Cannot start eating if already exercising
    is_exercising = state.get('planned_exercise_min', 0.0) > 0
    meal_accepted = jnp.logical_and(meal_accepted, jnp.logical_not(is_exercising))
    # -----------------------------------------------
    
    # Bolus Acceptance Logic
    bolus_prob_check = jax.random.uniform(bolus_accept_key) < patient_params.bolus_acceptance_prob
    bolus_time_check = (state['t'] - state['last_bolus_time']) > bolus_safe_window
    bolus_accepted = jnp.logical_and(bolus_prob_check, bolus_time_check)
    positive_bolus_check = action.bolus > 0.
    bolus_accepted = jax.lax.cond(positive_bolus_check, lambda: bolus_accepted, lambda: False)
    # Block bolus when CGM is already near-normal to prevent spam dosing
    cgm_hist = state['CGM_hist']
    idx = state['index']
    cgm_est = cgm_hist[jnp.maximum(idx - 1, 0)]
    bolus_bg_check = cgm_est >= BOLUS_BG_MIN
    bolus_accepted = jnp.logical_and(bolus_accepted, bolus_bg_check)

    # Hard daily limits for meal/bolus recommendations
    current_meal_count = state['meal_count_daily']
    current_bolus_count = state['bolus_count_daily']
    meal_limit_reached = current_meal_count >= MAX_MEAL_PER_DAY
    bolus_limit_reached = current_bolus_count >= MAX_BOLUS_PER_DAY
    meal_limit_block = jnp.logical_and(positive_meal_check, meal_limit_reached)
    bolus_limit_block = jnp.logical_and(positive_bolus_check, bolus_limit_reached)
    meal_accepted = jnp.logical_and(meal_accepted, jnp.logical_not(meal_limit_reached))
    bolus_accepted = jnp.logical_and(bolus_accepted, jnp.logical_not(bolus_limit_reached))
    # Low BG eats anyway: override all blocks when hypoglycemic and a meal is requested
    meal_accepted = jax.lax.cond(hypo_override, lambda: True, lambda: meal_accepted)

    # Preemption Override: If a scenario meal is imminent, allow the agent to intervene
    # to trigger the overlap/cancel logic, even if recently ate or meal count high.
    todays_schedule = state['scenario_meals'][state['t'] // 1440]
    meal_times = todays_schedule[:, 0]
    meal_amounts = todays_schedule[:, 1]
    mins_into_day = state['t'] % 1440
    
    # Check for upcoming meal within 15 mins (Shield's window)
    diff = meal_times - mins_into_day
    # > 0 means future. <= 15 means within window.
    upcoming_mask = (diff > 0.0) & (diff <= 15.0) & (meal_amounts > 0.0)
    has_upcoming_meal = jnp.any(upcoming_mask)
    
    preempt_override = has_upcoming_meal & positive_meal_check
    
    # Debug print to investigate why preemption might fail
    # jax.debug.print("T={t} | Mask={m} | HasUpcoming={h} | ActionMeal={a} | Override={o} | AcceptedBefore={ab}",
    #                 t=state['t'], m=upcoming_mask, h=has_upcoming_meal, a=action.meal, o=preempt_override, ab=meal_accepted)

    meal_accepted = jax.lax.cond(preempt_override, lambda: True, lambda: meal_accepted)
    
    # Exercise Acceptance Logic
    exercise_prob_check = jax.random.uniform(exercise_accept_key) < patient_params.exercise_acceptance_prob
    exercise_time_check = (state['t'] - state['last_exercise_time']) > patient_params.exercise_safe_window
    
    exercise_accepted = jnp.logical_and(exercise_prob_check, exercise_time_check)
    
    positive_exercise_check = action.exercise > 0.
    exercise_accepted = jax.lax.cond(positive_exercise_check, lambda: exercise_accepted, lambda: False)

    # --- Eating vs Exercise Exclusion ---
    # Cannot start exercising if already eating
    is_eating = state.get('planned_meal', 0.0) > 0
    exercise_accepted = jnp.logical_and(exercise_accepted, jnp.logical_not(is_eating))
    
    # Cannot start exercising if starting to eat in this same step (Meal Priority)
    exercise_accepted = jnp.logical_and(exercise_accepted, jnp.logical_not(meal_accepted))
    # -----------------------------------------------
    
    # Apply acceptance decisions
    actual_meal_g = jax.lax.cond(meal_accepted, lambda: action.meal * patient_params.max_meal_g, lambda: 0.0)
    actual_bolus_U = jax.lax.cond(bolus_accepted, lambda: action.bolus * patient_params.max_bolus_U, lambda: 0.0)
    
    # Exercise logic:
    # Action is duration fraction (0..1) -> duration in minutes
    new_exercise_min = action.exercise * patient_params.max_exercise_min
    
    # Apply noise to the accepted amounts
    meal_noise = jax.random.normal(meal_noise_key) * 0.1 * actual_meal_g
    bolus_noise = jax.random.normal(bolus_noise_key) * 0.01 * actual_bolus_U
    
    # Random intensity for this specific exercise session (e.g. 0.3 to 0.8)
    random_intensity = 0.3 + jax.random.uniform(ex_noise_key) * 0.5
    
    final_meal_g = jnp.maximum(0, actual_meal_g + meal_noise)
    final_bolus_U = jnp.maximum(0, actual_bolus_U + bolus_noise)

    current_planned_min = jnp.asarray(state.get('planned_exercise_min', 0.0), dtype=jnp.float32)
    current_intensity = jnp.asarray(state.get('exercise_intensity', 0.0), dtype=jnp.float32)
    current_count = state.get('exercise_count', 0)

    # Update state variables if new exercise is accepted
    updated_planned_min = jax.lax.cond(
        exercise_accepted,
        lambda: new_exercise_min,
        lambda: current_planned_min
    )
    updated_intensity = jax.lax.cond(
        exercise_accepted,
        lambda: random_intensity,
        lambda: current_intensity
    )
    # Exercise count updated only if accepted (and > 0 check implicit in acceptance)
    updated_ex_count = jax.lax.cond(
        exercise_accepted,
        lambda: current_count + 1,
        lambda: current_count
    )

    # Meal and Bolus Counts (Daily Reset Logic)
    # Reset counts if new day. Sim starts at t=0. 
    # We can track 'last_day_index' in state or just use t % 1440 == 0 logic?
    # Simpler: Just accumulate total count for now, or reset every 1440 steps?
    # The prompt asks for "times in a day". Let's track accumulated count and reset at midnight.
    # BUT: 'state' is immutable dict in JAX. We need fields for counts.
    
    current_ex_count_daily = state.get('ex_count_daily', 0)
    
    # Check for new day (midnight reset)
    # If t % 1440 == 0 and t > 0, reset.
    # Better: store 'day_index'. If current day > stored day, reset.
    current_day = state['t'] // 1440
    stored_day = state.get('day_index', 0)
    is_new_day = current_day > stored_day
    
    meal_cnt = jnp.where(is_new_day, 0, current_meal_count)
    bolus_cnt = jnp.where(is_new_day, 0, current_bolus_count)
    ex_cnt = jnp.where(is_new_day, 0, current_ex_count_daily)

    # Preserve counts BEFORE applying this action for reward shaping
    meal_cnt_before = meal_cnt
    bolus_cnt_before = bolus_cnt

    # Increment if action accepted and meaningful (>0)
    # Note: exercise_accepted is already calculated. 
    # We need meal_accepted and bolus_accepted check combined with > 0.
    # 'actual_meal_g' > 0 implies accepted.
    
    meal_cnt = jnp.where(actual_meal_g > 0, meal_cnt + 1, meal_cnt)
    bolus_cnt = jnp.where(actual_bolus_U > 0, bolus_cnt + 1, bolus_cnt)
    ex_cnt = jnp.where(exercise_accepted, ex_cnt + 1, ex_cnt)
    
    # Pass accepted duration to patient_step only if newly accepted
    # If continuing existing exercise, patient_step handles countdown
    # Actually, patient_step logic for duration needs to be simple decrement
    # So we pass the NEW state values to the carry
    
    patient_action = Action(
        meal=final_meal_g,
        bolus=final_bolus_U,
        exercise=0.0 # exercise handled via state updates
    )
    
    cho_total = 0.0
    insulin_total = 0.0
    bg_sum = 0.0
    cgm_sum = 0.0
    scenario_meal_sum = 0.0
    
    # Update carry with new counts and day index
    carry = {
        **state, 
        'planned_exercise_min': updated_planned_min, 
        'exercise_intensity': updated_intensity,
        'exercise_count': updated_ex_count, # This was the old single-limit counter, keep for compatibility or replace?
        'meal_count_daily': meal_cnt,
        'bolus_count_daily': bolus_cnt,
        'ex_count_daily': ex_cnt,
        'day_index': current_day
    }
    # If newly accepted, update last_exercise_time
    carry['last_exercise_time'] = jax.lax.cond(exercise_accepted, lambda: state['t'], lambda: state['last_exercise_time'])

    # Skip scenario meal during the current safety window after an accepted controller meal
    scenario_block_until = jnp.where(
        meal_accepted,
        state['t'] + meal_safe_window,
        state.get('scenario_block_until', jnp.array(-1, dtype=jnp.int32))
    )

    for i in range(sample_time):
        step_action = patient_action
        if i > 0:
            step_action = step_action._replace(meal=0.0, bolus=0.0)
            
        # Determine effective intensity for this mini-step based on updated carry state
        is_exercising = carry['planned_exercise_min'] > 0
        intensity_now = jax.lax.cond(is_exercising, lambda: carry['exercise_intensity'], lambda: 0.0)
        
        # Update action with current effective intensity
        # Note: patient_step now only reads ex_frac for ODE intensity
        step_action = step_action._replace(exercise=intensity_now)

        # Execute mini-step
        # mini_step calls patient_step, which decrements planned_exercise_min
        skip_scenario = (state['t'] + i) < scenario_block_until
        carry, basal, bolus, bg, cgm, scenario_meal = mini_step(
            carry,
            step_action,
            env_params,
            skip_scenario_meal=skip_scenario
        )

        # IMPORTANT: 'carry' is updated by mini_step and ALREADY has the updated kernels from patient_step
        # We do NOT need to manually update/shift them again here. doing so caused a double-update bug.
        carry = carry.copy()
        # Ensure counts persist through mini-steps (though they don't change inside mini-loop)
        carry['meal_count_daily'] = meal_cnt
        carry['bolus_count_daily'] = bolus_cnt
        carry['ex_count_daily'] = ex_cnt
        carry['day_index'] = current_day
        carry['scenario_block_until'] = scenario_block_until

        insulin_total += basal + bolus
        cho_total += jnp.asarray(carry.get('to_eat', 0.0), dtype=jnp.float32)
        bg_sum += bg
        cgm_sum += cgm
        scenario_meal_sum += scenario_meal

    new_state = carry
    index = state['index']
    bg_avg = bg_sum / sample_time
    cgm_avg = cgm_sum / sample_time
    scenario_meal_avg = scenario_meal_sum / sample_time
    # Compute CGM trend as the raw delta over the controller sample interval (e.g., CGM(t) - CGM(t-5min))
    prev_cgm_sample = state['CGM_hist'][jnp.maximum(index - 1, 0)]
    cgm_trend_sample = cgm_avg - prev_cgm_sample
    
    # Use IOB decay kernel for IOB calculation (fraction remaining), NOT the activity kernel
    iob_decay_kernel = jnp.asarray(env_params.iob_kernel, dtype=jnp.float32)
    iob = patient_iob(new_state, iob_decay_kernel)
    cob = patient_cob(new_state)
    new_state['BG_hist'] = state['BG_hist'].at[index].set(bg_avg)
    new_state['CGM_hist'] = state['CGM_hist'].at[index].set(cgm_avg)
    new_state['IOB_hist'] = state['IOB_hist'].at[index].set(iob)
    new_state['COB_hist'] = state['COB_hist'].at[index].set(cob)
    new_state['CHO_hist'] = state['CHO_hist'].at[index].set(cho_total)
    new_state['insulin_hist'] = state['insulin_hist'].at[index].set(insulin_total)
    new_state['cgm_trend'] = cgm_trend_sample
    new_state['index'] = index + 1
    action_time = state['t']
    new_state['last_meal_time'] = jnp.where(meal_accepted, action_time, new_state['last_meal_time'])
    new_state['last_bolus_time'] = jnp.where(bolus_accepted, action_time, new_state['last_bolus_time'])

    isf = jnp.asarray(
        getattr(patient_params, "ISF_mgdl_per_U", getattr(patient_params, "ISF", BASE_ISF_MGDL_PER_U)),
        dtype=jnp.float32,
    )
    dt_minutes = float(sample_time)
    
    # Determine if exercise is active for reward function
    exercise_active_now = new_state['planned_exercise_min'] > 0
    exercise_intensity_now = jax.lax.cond(exercise_active_now, lambda: new_state['exercise_intensity'], lambda: 0.0)

    # Calculate time since last meal and bolus for low-frequency reward shaping
    # Use state BEFORE action (state) to get time since last action
    current_time = state['t']
    time_since_meal = jnp.maximum(0.0, current_time - state['last_meal_time'])
    time_since_bolus = jnp.maximum(0.0, current_time - state['last_bolus_time'])
    
    # Update last_hypo_time if current BG is hypoglycemic
    is_hypo_now = bg_avg < 75.0
    new_state['last_hypo_time'] = jnp.where(
        is_hypo_now,
        current_time,
        state.get('last_hypo_time', jnp.array(-9999, dtype=jnp.int32))
    )

    # Downsample minute-resolution kernels into 5-minute bins so the proxy forecast
    # matches the 5-minute control interval without increasing compute.
    insulin_kernel_5 = jnp.asarray(env_params.insulin_kernel_5, dtype=jnp.float32)
    iob_kernel_minute = new_state['basal_kernel'] + new_state['bolus_kernel']
    iob_kernel_5 = _downsample_sum(iob_kernel_minute, _DOWNSAMPLE_FACTOR)
    
    reward, cost, reward_info, cost_info = REWARD_FUNCTIONN(
        current_bg=new_state['BG_hist'][index],
        iob_kernel=iob_kernel_5,
        cob_states=new_state['patient_state'][0:3],
        insulin_kernel=insulin_kernel_5,
        action_taken=jnp.asarray([
            patient_params.basal / 60.0,
            patient_action.bolus,
            patient_action.meal,
            exercise_intensity_now
        ], dtype=jnp.float32),
        isf=isf,
        beta_cell_function=patient_params.beta_cell_function,
        Gb=patient_params.Gb,
        BW=patient_params.BW,
        Vg=patient_params.Vg,
        dt=dt_minutes,
        forecast_steps=24,
        cob_kernel_len=3,
        meal_count=meal_cnt_before,
        bolus_count=bolus_cnt_before,
        time_since_meal=time_since_meal,
        time_since_bolus=time_since_bolus,
        minutes_into_day=state['t'] % 1440,
    )

    # The frequency penalties are now gated inside jax_forecast_reward_function, so we use the reward as is.
    
    # Additional Rejection Penalties
    # If the action was blocked by safety rules (time window, daily limit, etc.), apply a small penalty.
    # FIX: Only penalize if the agent ACTUALLY ATTEMPTED an action (positive check) that was rejected.
    meal_time_rejection = jnp.logical_and(positive_meal_check, jnp.logical_not(meal_time_check))
    bolus_time_rejection = jnp.logical_and(positive_bolus_check, jnp.logical_not(bolus_time_check))
    
    was_rejected = jnp.logical_or(meal_time_rejection, bolus_time_rejection)
    was_rejected = jnp.logical_or(was_rejected, meal_limit_block)
    was_rejected = jnp.logical_or(was_rejected, bolus_limit_block)
    
    rejection_penalty = jnp.where(was_rejected, 0.5, 0.0)

    terminated = jnp.logical_or(bg_avg < 10, bg_avg > 600)
    truncated = (new_state['t'] >= simulation_minutes)
    # Encoded termination cause to remain JAX-compatible (no strings inside jit)
    # 0: none, 1: hypo, 2: hyper, 3: timeout
    termination_cause = jnp.zeros((), dtype=jnp.int32)
    termination_cause = jnp.where(terminated & (bg_avg < 10), jnp.int32(1), termination_cause)
    termination_cause = jnp.where(terminated & (bg_avg > 600), jnp.int32(2), termination_cause)
    termination_cause = jnp.where(~terminated & truncated, jnp.int32(3), termination_cause)
    time_in_range = jnp.logical_and(bg_avg >= 70, bg_avg <= 180)
    hypo = bg_avg < 70
    hyper = bg_avg > 180

    # Final Reward Assembly
    reward = reward - rejection_penalty

    # Failure-only terminal penalty to prevent early-termination incentives
    # Scale by remaining steps so dying early is strictly worse than enduring negatives
    max_steps = simulation_minutes // sample_time
    remaining_steps = jnp.maximum(0, max_steps - state['index'])
    PER_STEP_PENALTY_SCALE = 2.0  # tune relative to typical per-step magnitude
    terminal_penalty = PER_STEP_PENALTY_SCALE * remaining_steps
    reward = jnp.where(terminated, reward - terminal_penalty, reward)

    bolus_and_zero_meal_check = jnp.logical_and(positive_bolus_check, jnp.logical_not(positive_meal_check))
    meal_and_zero_bolus_check = jnp.logical_and(positive_meal_check, jnp.logical_not(positive_bolus_check))

    special_case = jnp.logical_or(jnp.logical_and(bolus_and_zero_meal_check, hyper),
                                  jnp.logical_and(meal_and_zero_bolus_check, hypo))
    cost = jnp.where(special_case, cost * 0.5, cost)

    obs = _build_observation(new_state, env_params)
    info = {'reward_info': reward_info,
            'cost_info': cost_info,
            'time_in_range': time_in_range,
            'hypo': hypo,
            'hyper': hyper,
            'bolus_accepted': bolus_accepted,
            'meal_accepted': meal_accepted,
            'meal_limit_reached': meal_limit_reached,
            'bolus_limit_reached': bolus_limit_reached,
            'bolus_block_time_window': jnp.logical_and(positive_bolus_check, jnp.logical_not(bolus_time_check)),
            'bolus_block_bg_low': jnp.logical_and(positive_bolus_check, jnp.logical_not(bolus_bg_check)),
            'meal_block_time_window': jnp.logical_and(positive_meal_check, jnp.logical_not(meal_time_check)),
            'scenario_meal_avg': scenario_meal_avg,
            'insulin_total_U': insulin_total,
            'meal_total_g': final_meal_g,
            'exercise_intesity': exercise_intensity_now,
            'termination_cause': termination_cause}
    return new_state, obs, reward, cost, terminated, truncated, info
