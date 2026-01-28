from typing import Any, ClassVar, Union, Tuple, List, Dict, Optional
import torch
from gymnasium import spaces
import numpy as np
from omnisafe.envs.core import CMDP, env_register
import glucosim
from glucosim.safety_gymnasium.vector import SafetySyncVectorEnv
from glucosim.simglucose.sim.env import JaxSimEnv
from stable_baselines3.common.utils import set_random_seed
import jax


@env_register
class DiabetesEnvs(CMDP):
    _support_envs: ClassVar[List[str]] = ['t1d-v0', 't2d-v0', 't2d_no_pump-v0']
    need_auto_reset_wrapper = False  # Whether `AutoReset` Wrapper is needed
    need_time_limit_wrapper = False  # Whether `TimeLimit` Wrapper is needed

    def __init__(self, env_id: str, device: str, num_envs: int = 1, render_mode: Optional[str] = None, simulation_minutes: int = 24 * 60, sample_time: int = 5, patient_name: str = "adolescent#001", seed: int = 0, **kwargs) -> None:
        self.render_mode = render_mode
        self.simulation_minutes = simulation_minutes # this will be max step for the simulator
        self.sample_time = sample_time
        master_key = jax.random.PRNGKey(0)
        env_keys = jax.random.split(master_key, num_envs)
        env_fns = [lambda k=k: JaxSimEnv(env_id, key=k, simulation_minutes=simulation_minutes, sample_time=sample_time, patient_name=patient_name) for k in env_keys]

        self.env = SafetySyncVectorEnv(env_fns)
        self._num_envs = num_envs
        self._device = torch.device(device)

        self._org_observation_space = self.env.single_observation_space

        self._observation_space = spaces.Box(low=self._org_observation_space.low, high=self._org_observation_space.high)
        self._action_space = self.env.single_action_space
        
    def reset(self, seed: Union[int, None] = None, options: Union[Dict[str, Any], None] = None) -> Tuple[torch.Tensor, dict]:
        obs, info = self.env.reset(seed=seed)
        obs = torch.from_numpy(np.array(obs)).float().to(self._device)
        return obs, info
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # Handle both continuous and discrete actions    
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        obs = torch.from_numpy(np.array(obs)).float().to(self._device)
        reward = torch.from_numpy(np.array(reward)).float().to(self._device)
        cost = torch.from_numpy(np.array(cost)).float().to(self._device)
        terminated = torch.from_numpy(np.array(terminated)).float().to(self._device)
        truncated = torch.from_numpy(np.array(truncated)).float().to(self._device)
        if torch.any(terminated) or torch.any(truncated):
            # Convert to boolean before OR operation
            info['_final_observation'] = (terminated > 0) | (truncated > 0)
            info['final_observation'] = obs
            # exit()
        return obs, reward, cost, terminated, truncated, info
    
    def set_seed(self, seed: int) -> None:
        set_random_seed(seed)

    def render(self, history: Optional[Dict[str, Any]] = None, save_dir: Optional[str] = None, episode: Optional[int] = None) -> Any:
        """Render controller behaviour by saving glucose/action plots if history is provided or can be loaded from CSV."""
        try:
            import os
            import glob
            import matplotlib.pyplot as plt
            import pandas as pd

            if history is None and save_dir:
                csv_files = sorted(glob.glob(os.path.join(save_dir, "*.csv")))
                if csv_files:
                    latest_csv = csv_files[-1]
                    df_csv = pd.read_csv(latest_csv)
                    history = {
                        'CGM': df_csv.get('cgm', []),
                        'BG': df_csv.get('cgm', []),
                        'IOB': df_csv.get('iob', []),
                        'COB': df_csv.get('cob', []),
                        'bolus_units': df_csv.get('bolus_units', []),
                        'meal_grams': df_csv.get('meal_grams', []),
                        'exercise_minutes': df_csv.get('exercise_minutes', []),
                        'bolus_index': df_csv.get('bolus_index', []),
                        'meal_index': df_csv.get('meal_index', []),
                        'exercise_index': df_csv.get('exercise_index', []),
                        'time_hours': df_csv.get('time_hours', []),
                    }

            if history is None:
                return self.env.render()

            out_dir = save_dir or "./diabetes_evaluation/renders"
            os.makedirs(out_dir, exist_ok=True)
            episode_label = episode if episode is not None else "latest"

            cgm_values = np.asarray(history.get('CGM', history.get('BG', [])))
            iob_values = np.asarray(history.get('IOB', []))
            cob_values = np.asarray(history.get('COB', []))
            actions = history.get('action', [])

            if cgm_values.size == 0 or len(actions) == 0:
                return self.env.render()

            # Trim all time series to the shortest length to stay aligned
            series_lengths = [len(cgm_values), len(iob_values), len(cob_values), len(actions)]
            valid_lengths = [l for l in series_lengths if l > 0]
            min_len = min(valid_lengths) if valid_lengths else 0
            if min_len == 0:
                return self.env.render()

            cgm_values = cgm_values[:min_len]
            iob_values = iob_values[:min_len]
            cob_values = cob_values[:min_len]
            actions = actions[:min_len]

            jax_env = self.env.envs[0] if hasattr(self.env, 'envs') and self.env.envs else None
            bolus_levels = getattr(jax_env, 'bolus_levels', None) if jax_env is not None else None
            meal_levels = getattr(jax_env, 'meal_levels', None) if jax_env is not None else None
            exercise_levels = getattr(jax_env, 'exercise_levels', None) if jax_env is not None else None
            patient_params = getattr(jax_env, 'env_params', None)
            max_bolus_units = getattr(patient_params.patient_params, 'max_bolus_U', None) if patient_params is not None else None
            max_meal_grams = getattr(patient_params.patient_params, 'max_meal_g', None) if patient_params is not None else None
            max_exercise_min = getattr(patient_params.patient_params, 'max_exercise_min', None) if patient_params is not None else None

            bolus_units_logged = list(history.get('bolus_units', []))
            meal_grams_logged = list(history.get('meal_grams', []))
            exercise_minutes_logged = list(history.get('exercise_minutes', []))

            def _action_to_units(action_val):
                arr = np.array(action_val, copy=False).flatten()
                bolus_idx = int(arr[0]) if arr.size > 0 else 0
                meal_idx = int(arr[1]) if arr.size > 1 else 0
                exercise_idx = int(arr[2]) if arr.size > 2 else 0

                bolus_units = float(arr[0]) if arr.size > 0 else 0.0
                meal_grams = float(arr[1]) if arr.size > 1 else 0.0
                exercise_minutes = float(arr[2]) if arr.size > 2 else 0.0

                if bolus_levels is not None and max_bolus_units is not None and bolus_idx < len(bolus_levels):
                    bolus_units = float(bolus_levels[bolus_idx] * max_bolus_units)
                if meal_levels is not None and max_meal_grams is not None and meal_idx < len(meal_levels):
                    meal_grams = float(meal_levels[meal_idx] * max_meal_grams)
                if exercise_levels is not None and max_exercise_min is not None and exercise_idx < len(exercise_levels):
                    exercise_minutes = float(exercise_levels[exercise_idx] * max_exercise_min)

                return bolus_units, meal_grams, exercise_minutes

            # Backfill units if they were not recorded
            if not bolus_units_logged or not meal_grams_logged:
                bolus_units_logged, meal_grams_logged, exercise_minutes_logged = [], [], []
                for action in actions:
                    bu, mg, ex_min = _action_to_units(action)
                    bolus_units_logged.append(bu)
                    meal_grams_logged.append(mg)
                    exercise_minutes_logged.append(ex_min)
            else:
                bolus_units_logged = bolus_units_logged[:min_len]
                meal_grams_logged = meal_grams_logged[:min_len]
                exercise_minutes_logged = exercise_minutes_logged[:min_len]

            time_axis_hours = np.arange(min_len) * (self.sample_time / 60.0)

            fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

            axs[0].plot(time_axis_hours, cgm_values, label='CGM (mg/dL)', color='blue')
            axs[0].axhline(70, color='red', linestyle='--', label='Hypo (70)')
            axs[0].axhline(180, color='orange', linestyle='--', label='Hyper (180)')
            axs[0].set_ylabel('Glucose (mg/dL)')
            
            # Extract metrics from history if available
            metrics = history.get('metrics', {})
            title_str = 'Blood Glucose Trend'
            if metrics:
                tir = metrics.get('tir', 0.0)
                sd = metrics.get('sd', 0.0)
                cv = metrics.get('cv', 0.0)
                mag = metrics.get('mag', 0.0)
                mage = metrics.get('mage', 0.0)
                title_str += f"\nTIR: {tir:.1f}% | SD: {sd:.1f} | CV: {cv:.1f}% | MAG: {mag:.2f} | MAGE: {mage:.1f}"
            
            axs[0].set_title(title_str)
            axs[0].legend(loc='upper right')
            axs[0].grid(True, alpha=0.3)

            axs[1].plot(time_axis_hours, iob_values, label='IOB (U)', color='purple')
            axs[1].set_ylabel('IOB (U)')
            axs[1].legend(loc='upper right')
            axs[1].grid(True, alpha=0.3)

            axs[2].plot(time_axis_hours, cob_values, label='COB (g)', color='brown')
            axs[2].set_ylabel('COB (g)')
            axs[2].legend(loc='upper right')
            axs[2].grid(True, alpha=0.3)

            axs[3].step(time_axis_hours, bolus_units_logged, where='post', label='Bolus (U)', color='crimson')
            axs[3].step(time_axis_hours, meal_grams_logged, where='post', label='Meal (g)', color='darkgreen', alpha=0.7)
            if any(exercise_minutes_logged):
                axs[3].step(time_axis_hours, exercise_minutes_logged, where='post', label='Exercise (min)', color='navy', alpha=0.6)
            axs[3].set_ylabel('Action Magnitude')
            axs[3].set_xlabel('Time (hours)')
            axs[3].legend(loc='upper right')
            axs[3].grid(True, alpha=0.3)

            plt.tight_layout()
            trend_path = os.path.join(out_dir, f"episode_{episode_label}_controller.png")
            plt.savefig(trend_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Histogram view of action magnitudes
            hist_fig, hist_axs = plt.subplots(1, 3, figsize=(14, 4))
            hist_axs[0].hist(bolus_units_logged, bins=10, color='crimson', alpha=0.8)
            hist_axs[0].set_title('Bolus Distribution (U)')
            hist_axs[0].set_xlabel('Units')
            hist_axs[0].set_ylabel('Count')

            hist_axs[1].hist(meal_grams_logged, bins=10, color='darkgreen', alpha=0.8)
            hist_axs[1].set_title('Meal Distribution (g)')
            hist_axs[1].set_xlabel('Grams')

            hist_axs[2].hist(exercise_minutes_logged, bins=10, color='navy', alpha=0.8)
            hist_axs[2].set_title('Exercise Distribution (min)')
            hist_axs[2].set_xlabel('Minutes')

            plt.tight_layout()
            hist_path = os.path.join(out_dir, f"episode_{episode_label}_controller_hist.png")
            plt.savefig(hist_path, dpi=150, bbox_inches='tight')
            plt.close(hist_fig)

            return {"trend_path": trend_path, "hist_path": hist_path}

        except Exception as exc:
            print(f"Render failed: {exc}")
            return self.env.render()
    
    def render_rgb_array(self) -> np.ndarray:
        return self.env.render()
    
    def close(self) -> None:
        self.env.close()

    @property
    def max_episode_steps(self) -> None:
        return self.simulation_minutes / self.sample_time
    
    
