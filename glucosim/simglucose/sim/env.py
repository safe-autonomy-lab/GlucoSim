import torch
import csv
import os
import logging
from collections import OrderedDict
import jax
import jax.numpy as jnp
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from typing import Dict, Optional
from ..sim.step import step
from ..sim.reset import reset
from ..physiology.initialization import tune_initial_state
from ..core.params import create_env_params
from ..core.types import Action

# Set up logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)


class JaxSimEnv(gym.Env):
    """A Gymnasium wrapper for the JAX-based diabetes simulator."""
    metadata = {"render_modes": ["human"]}
    _warmup_cache = OrderedDict()
    _warmup_cache_max_size = 128
    cause_map = {
        0: 'none',
        1: 'hypo',
        2: 'hyper',
        3: 'timeout'
    }

    def __init__(
            self, 
            env_id: str, 
            key: jnp.ndarray, 
            action_dim: int = 5, 
            simulation_minutes: int = 24 * 60, 
            sample_time: int = 5,
            patient_name: str = "adolescent#001",
            patient_overrides: Dict = None,
            log_file: Optional[str] = None
        ):
        super().__init__()
        assert env_id in ['t1d-v0', 't2d-v0', 't2d_no_pump-v0'], f"Invalid environment ID: {env_id}"
        
        # Store configuration
        self.env_id = env_id
        self.simulation_minutes = simulation_minutes
        self.sample_time = sample_time
        self.diabetes_type = env_id.split('-')[0]

        overrides = dict(patient_overrides or {})
        # Override probabilities to 1.0 for deterministic action execution in Gym unless user overrides
        overrides.setdefault('meal_acceptance_prob', 1.0)
        overrides.setdefault('bolus_acceptance_prob', 1.0)
        overrides.setdefault('exercise_acceptance_prob', 1.0)

        self.env_params = create_env_params(
            patient_name=patient_name,
            diabetes_type=self.diabetes_type, 
            simulation_minutes=simulation_minutes, 
            sample_time=sample_time,
            **overrides
        )        

        
        # Define discrete action space: [bolus_level, meal_level, exercise_level]
        # Each dimension has `action_dim` discrete levels (0, 1, 2, 3, ...)
        # TODO: temporally, action dim is 2
        self.action_space = spaces.MultiDiscrete([action_dim] * 2)
        # Define discrete action mappings for safety and interpretability
        # Safety-validated discrete levels to prevent dangerous dosing
        max_bolus = self.env_params.patient_params.max_bolus_U  # 10.0 U
        max_meal = self.env_params.patient_params.max_meal_g    # 80.0 g
        self.patient_params = self.env_params.patient_params

        logger.info("--------------------------------")
        logger.info(f"Creating environment parameters for patient: {patient_name}")
        logger.info(f"Diabetes type: {self.diabetes_type}")
        logger.info(f"Meal acceptance probability: {self.patient_params.meal_acceptance_prob}")
        logger.info(f"Bolus acceptance probability: {self.patient_params.bolus_acceptance_prob}")
        logger.info(f"Exercise acceptance probability: {self.patient_params.exercise_acceptance_prob}")
        logger.info(f"Max bolus: {self.patient_params.max_bolus_U}")
        logger.info(f"Max meal: {self.patient_params.max_meal_g}")
        logger.info(f"Max exercise duration: {self.patient_params.max_exercise_min}")
        logger.info("--------------------------------")


        # Bolus levels: 0=none, 1=small, 2=medium, 3=large - all within safe limits
        # Fixed: Use action_dim instead of action_dim - 1 to match the space size
        assert max_bolus >= 5.33, f"Max bolus {max_bolus}U too low for large discrete action"
        self.bolus_levels = np.linspace(0.0, 1.0, action_dim)
        
        # Meal levels: 0=none, 1=small, 2=medium, 3=large - physiologically reasonable
        assert max_meal >= 50.0, f"Max meal {max_meal}g too low for large discrete action"
        self.meal_levels = np.linspace(0.0, 1.0, action_dim)

        # Exercise levels: fraction of maximum exercise duration (0.0 to 1.0)
        self.exercise_levels = np.linspace(0.0, 1.0, action_dim)
        
        logger.info("Initialized discrete action space:")
        logger.info(f"Bolus levels (normalized): {self.bolus_levels}")
        logger.info(f"Meal levels (normalized): {self.meal_levels}")
        logger.info(f"Exercise levels (duration fraction): {self.exercise_levels}")
        logger.info(f"Max bolus: {self.env_params.patient_params.max_bolus_U}U")
        logger.info(f"Max meal: {self.env_params.patient_params.max_meal_g}g")
        logger.info(f"Max exercise duration: {self.env_params.patient_params.max_exercise_min}min")

        # Correctly define the low and high bounds for each of the 14 features
        # [ cgm, iob, cob, cgm_trend, time_sin, time_cos,
        #   time_since_meal, time_since_bolus, planned_meal_left,
        #   meal_count_norm, bolus_count_norm,
        #   time_until_meal_norm, next_meal_size_norm, is_pre_bolus_window ]
        obs_space_low = np.array([
            10.0,  # CGM min (mg/dL)
            -0.1,  # IOB min (U)
            -0.1,  # COB min (g)
            -20.0, # cgm_trend min (mg/dL/min) - estimated safe bound
            -1.0,  # time_sin min
            -1.0,  # time_cos min
            0.0,   # time_since_meal min (normalized)
            0.0,   # time_since_bolus min (normalized)
            0.0,   # planned_meal_left min (normalized)
            # 0.0,   # planned_ex_min min (normalized)
            0.0,   # meal_count_norm min
            0.0,   # bolus_count_norm min
            0.0,   # time_until_meal_norm min
            0.0,   # next_meal_size_norm min
            0.0    # is_pre_bolus_window min
        ], dtype=np.float32)

        obs_space_high = np.array([
            600.0, # CGM max (mg/dL)
            50.0,  # IOB max (U)
            200.0, # COB max (g)
            20.0,  # cgm_trend max (mg/dL/min) - estimated safe bound
            1.0,   # time_sin max
            1.0,   # time_cos max
            1.0,   # time_since_meal max (normalized)
            1.0,   # time_since_bolus max (normalized)
            1.0,   # planned_meal_left max (normalized)
            # 1.0,   # planned_ex_min max (normalized)
            1.0,   # meal_count_norm max
            1.0,   # bolus_count_norm max
            1.0,   # time_until_meal_norm max
            1.0,   # next_meal_size_norm max
            1.0    # is_pre_bolus_window max
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=obs_space_low,
            high=obs_space_high,
            dtype=np.float32
        )
        
        # This will hold the current state of the JAX simulation
        self._jax_state = None
        self.key = key # Master random key
        
        # Logging setup
        self.episode_meal_count = 0
        self.episode_bolus_count = 0
        self.episode_exercise_count = 0
        self.episode_return = 0.0
        self.episode_cost = 0.0
        self.episode_idx = 0
        self.episode_steps = 0
        # Use a filename based on env_id
        self.log_file = log_file
        # self.log_file = f"simulation_log_{env_id}.csv"
        self._log_buffer = []
        self._log_flush_interval = 10
        self._log_file_initialized = os.path.isfile(self.log_file) if self.log_file is not None else False

    def get_patient_params(self):
        return self.patient_params

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset episode statistics
        self.episode_meal_count = 0
        self.episode_bolus_count = 0
        self.episode_exercise_count = 0
        self.episode_return = 0.0
        self.episode_cost = 0.0
        self.episode_steps = 0
        # Create a new master key for the episode
        if self.key is None:
            self.key = jax.random.PRNGKey(seed if seed is not None else 0)
        
        # Split the key for the reset function
        self.key, reset_key = jax.random.split(self.key)

        # Tune the initial state (calculates steady state based on patient params)
        self.env_params, tuned_state = tune_initial_state(self.env_params)

        # Call the JAX-native reset function (with warmup caching)
        # Since tuning the initial state is time-consuming, we cache the warmup state
        # We assume that patients' parameters are fixed for the entire episode and noise configuration is fixed
        cache_key = (
            self.env_params.patient_params,
            self.env_params.sample_time,
            self.env_params.simulation_minutes,
            self.env_params.dia_steps,
            self.env_params.noise_config,
        )
        warm_state = self._get_warm_state(cache_key)
        initial_state, initial_obs = reset(
                self.env_params, tuned_state, reset_key, warm_state=warm_state
            )
        # We need to call this at least once
        if warm_state is None:
            warm_state, initial_state, initial_obs = reset(self.env_params, tuned_state, reset_key, return_warm_state=True)
            self._store_warm_state(cache_key, warm_state)
        else:
            initial_state, initial_obs = reset(self.env_params, tuned_state, reset_key, warm_state=warm_state)
        
        self.scenario_meals = initial_state.get('scenario_meals', [])
        self.patient_params = self.get_patient_params()
        # If we want to print the scenario meals, uncomment the following line
        # self._print_scenario_meals(self.scenario_meals)
        
        # Store the initial JAX state internally
        self._jax_state = initial_state
        
        # Log patient info for debugging
        logger.debug(f"Reset with patient type {self.env_params.patient_params.diabetes_type}: "
                    f"BW={self.env_params.patient_params.BW:.1f}kg, "
                    f"beta_cell={self.env_params.patient_params.beta_cell_function:.2f}, "
                    f"IR_factor={self.env_params.patient_params.insulin_resistance_factor:.1f}")
        
        # Return the standard Gymnasium reset signature
        obs = np.asarray(initial_obs)
        info = {'bolus_accepted': 0.0, 'meal_accepted': 0.0, 'exercise_accepted': 0.0}
        info = self._sanitize_info(info)
        return obs, info

    def _get_warm_state(self, cache_key):
        warm_state = self._warmup_cache.get(cache_key)
        if warm_state is not None:
            self._warmup_cache.move_to_end(cache_key)
        return warm_state

    def _store_warm_state(self, cache_key, warm_state):
        self._warmup_cache[cache_key] = warm_state
        self._warmup_cache.move_to_end(cache_key)
        if len(self._warmup_cache) > self._warmup_cache_max_size:
            self._warmup_cache.popitem(last=False)

    def step(self, action):
        if self._jax_state is None:
            raise RuntimeError("Cannot call step before reset. Call env.reset() first.")
            
        # torch.Tensor is used when we work with omnisafe RL framework
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # Track requested actions
        if int(action[0]) > 0: self.episode_bolus_count += 1
        if int(action[1]) > 0: self.episode_meal_count += 1
        # if int(action[2]) > 0: self.episode_exercise_count += 1
        self.episode_steps += 1
        # Convert to expected dictionary format for the JAX step function
        action_tuple = Action(
            bolus=self.bolus_levels[int(action[0])],
            meal=self.meal_levels[int(action[1])],
            exercise=0.0,
        )
        
        # Split the key for the step function
        self.key, step_key = jax.random.split(self.key)
        
        # Call the JAX-native step function with the current state and action
        new_state, obs, reward, cost, terminated, truncated, info = step(
            self._jax_state, action_tuple, self.env_params, step_key
        )
        # JAX returns DeviceArrays; convert for bookkeeping/adjustments
        reward = float(reward)
        cost = float(cost)

        reward, cost, info = self._apply_terminal_penalty(
            reward, cost, info, terminated, truncated
        )

        # Update cumulative return and cost
        self.episode_return += reward
        self.episode_cost += cost

        if terminated | truncated:
            cause_code = int(info.get('termination_cause', 0))
            self._last_termination_cause = self.cause_map.get(cause_code, 'unknown')
            logger.info('--------------------------------')
            logger.info('obs: %s', obs)
            logger.info(f'discrete action: bolus_level={action[0]}, meal_level={action[1]}')
            logger.info('continuous action taken: %s', action_tuple)
            logger.info('truncated or terminated')
            logger.info('reward: %s', reward)
            logger.info('cost: %s', cost)
            logger.info('terminated: %s', terminated)
            logger.info('truncated: %s', truncated)
            logger.info('termination_cause: %s', self._last_termination_cause)
            logger.info('info: %s', info)
            logger.info('--------------------------------')
            info['_last_termination_cause'] = self._last_termination_cause            
            # Log episode results to CSV
            self.episode_idx += 1
            self._log_episode()

        # Update the internal state with the new state returned by JAX
        self._jax_state = new_state        
        # Return the standard Gymnasium step signature
        obs = np.asarray(obs)
        terminated = bool(terminated)
        truncated = bool(truncated)
        info = self._sanitize_info(info)
        return obs, reward, cost, terminated, truncated, info

    def _sanitize_info(self, info):
        """Convert JAX/NumPy values to plain Python/NumPy to avoid device-array leaks."""
        def _to_host(val):
            if isinstance(val, dict):
                return {k: _to_host(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return type(val)(_to_host(v) for v in val)
            try:
                arr = np.asarray(val)
            except Exception:
                return val
            if arr.shape == ():
                return arr.item()
            return arr

        return _to_host(info)

    def _apply_terminal_penalty(self, reward, cost, info, terminated, truncated):
        """Apply end-of-episode penalties when the patient terminates early."""
        if not bool(terminated) or bool(truncated):
            return reward, cost, info

        max_steps = self.env_params.simulation_minutes // self.env_params.sample_time
        remaining_steps = max(0, max_steps - self.episode_steps)
        cause_code = int(info.get('termination_cause', 0))

        if cause_code == 2:  # hyper
            reward += -3.0 * (1.0 + (remaining_steps / max_steps))
            ghost_cost = float(remaining_steps)
        elif cause_code == 1:  # hypo
            reward += -5.0 * (1.0 + (remaining_steps / max_steps))
            ghost_cost = float(2 * remaining_steps)
        else:
            return reward, cost, info

        cost += ghost_cost
        info = {**info, 'ghost_cost': ghost_cost}
        return reward, cost, info
        
    def _log_episode(self):
        if self.log_file is not None:
            self._log_buffer.append([
                self.episode_idx,
                self.episode_bolus_count,
                self.episode_meal_count,
                self.episode_exercise_count,
                self.episode_return,
                self.episode_cost,
                self.episode_steps,
                getattr(self, '_last_termination_cause', 'unknown')
            ])
            if len(self._log_buffer) >= self._log_flush_interval:
                self._flush_log_buffer()

    def _flush_log_buffer(self):
        if not self._log_buffer:
            return
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not self._log_file_initialized:
                writer.writerow(['Episode', 'Bolus_Count', 'Meal_Count', 'Exercise_Count', 'Return', 'Cost', 'Steps', 'Termination_Cause'])
                self._log_file_initialized = True
            writer.writerows(self._log_buffer)
        self._log_buffer = []

    def close(self):
        self._flush_log_buffer()

    def render(self):
        if self._jax_state:
            bg = self._jax_state.get('cgm_last', 0.0) # Safely get CGM if available
            t = self._jax_state.get('t', 0)
            print(f"Time Step: {t}, CGM: {bg:.2f} mg/dL")
        else:
            print("Environment has not been reset yet.")

    def _format_minutes(self, minutes: float) -> str:
        """Convert minutes since midnight into HH:MM, showing rollover if it exceeds a day."""
        minutes_rounded = max(0, int(round(float(minutes))))
        day_offset, minute_in_day = divmod(minutes_rounded, 1440)
        hour = minute_in_day // 60
        minute = minute_in_day % 60
        suffix = f" (+{day_offset}d)" if day_offset else ""
        return f"{hour:02d}:{minute:02d}{suffix}"

    def _print_scenario_meals(self, scenario_meals):
        """
        Print the generated scenario meals in a human-friendly format:
        Day N: HH:MM: Xg | HH:MM: Yg ...
        """
        if scenario_meals is None or len(np.shape(scenario_meals)) == 0:
            print("No scenario meals were generated.")
            return

        arr = np.asarray(scenario_meals)
        if arr.ndim != 3 or arr.shape[2] < 2:
            print(f"Scenario meals have unexpected shape: {arr.shape}")
            return

        # Show every generated day so users can inspect the whole schedule, not just the sim horizon
        days_to_show = arr.shape[0]

        print("Generated meal scenario:")
        for day in range(days_to_show):
            day_schedule = arr[day]
            meals = [(t, a) for t, a in day_schedule if t >= 0 and a > 0]
            if not meals:
                print(f"  Day {day + 1}: no meals scheduled")
                continue

            meals.sort(key=lambda x: x[0])
            meals_str = " | ".join(
                f"{self._format_minutes(t)}: {int(round(a))}g" for t, a in meals
            )
            print(f"  Day {day + 1}: {meals_str}")
