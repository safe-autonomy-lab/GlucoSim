import sys
import os
# Add project root to sys.path to allow imports from envs.*
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import argparse
import dataclasses
import logging
from typing import Optional, Tuple, Callable

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

from glucobench.simglucose.core.params import PatientParams, create_env_params, NoiseConfig
from glucobench.simglucose.physiology.initialization import tune_initial_state
from glucobench.simglucose.physiology.glucose_dynamics import (
    t1d_rk4_step,
    t2d_rk4_step,
    hovorka_t1d,
    hybrid_t2d,
)
from glucobench.simglucose.core.types import PatientType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Safety bounds for glucose monitoring (mg/dL)
GLUCOSE_HYPOGLYCEMIA_THRESHOLD = 70.0
GLUCOSE_HYPERGLYCEMIA_THRESHOLD = 250.0
GLUCOSE_SEVERE_HYPO_THRESHOLD = 50.0
GLUCOSE_SEVERE_HYPER_THRESHOLD = 400.0


def simulate(
    t_span_min: float,
    dt_min: float,
    x0: jnp.ndarray,
    action_fn: Callable[[float], jnp.ndarray],
    params: PatientParams,
    cfg: NoiseConfig,
    key: jnp.ndarray,
    t0_min: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified simulation loop for T1D and T2D models.
    """
    num_steps = int(t_span_min / dt_min)
    times = np.linspace(0, t_span_min, num_steps + 1)
    states = np.zeros((num_steps + 1, x0.shape[0]), dtype=float)
    states[0] = np.array(x0)

    x = x0
    prev_carb = 0.0
    last_Qsto = 0.0
    last_foodtaken = 0.0

    # per-day key (site variability)
    key, day_key = jax.random.split(key)
    day_key = jax.random.fold_in(day_key, 0)
    ou_state_dL = jnp.array(0.0)

    # Dispatch based on patient type
    is_t1d = (params.diabetes_type == PatientType.t1d)
    step_fn = t1d_rk4_step if is_t1d else t2d_rk4_step

    logger.info(f"Starting simulation for {params.diabetes_type} over {t_span_min} minutes...")

    for i in range(num_steps):
        t = times[i]
        action = action_fn(t)  # [carb, insulin, hr_reserve]
        carb = float(action[0])

        # Meal tracking
        if carb > 0 and prev_carb == 0:
            last_Qsto = float(states[i, 0] + states[i, 1])  # D1 + D2 (mg)
            last_foodtaken = 0.0
            logger.debug(f"New meal at t={t:.1f} min: last_Qsto={last_Qsto:.1f} mg")

        last_foodtaken += carb

        # Step integration
        key, sk = jax.random.split(key)
        
        if is_t1d:
             x, key, ou_state_dL = step_fn(
                x=x, dt=dt_min, action=action, params=params,
                last_Qsto=last_Qsto, last_foodtaken=last_foodtaken,
                t_min=t0_min + t, key=sk, cfg=cfg, ou_state_dL=ou_state_dL
            )
        else:
            # T2D step signature includes day_key
            x, key, ou_state_dL = step_fn(
                x=x, dt=dt_min, action=action, params=params,
                last_Qsto=last_Qsto, last_foodtaken=last_foodtaken,
                t_min=t0_min + t, key=sk, cfg=cfg, ou_state_dL=ou_state_dL
            )
            
        states[i + 1] = np.array(x)

        if carb == 0 and prev_carb > 0:
            last_foodtaken = 0.0
            logger.debug(f"Meal ended at t={t:.1f} min")

        prev_carb = carb

        # Safety Check (mg/dL)
        if is_t1d:
             # T1D: Gp is index 3
             G_mgdL = float(x[3] / params.Vg)
        else:
             # T2D: Gp is index 3 (same)
             G_mgdL = float(x[3] / params.Vg)

        if G_mgdL < GLUCOSE_SEVERE_HYPO_THRESHOLD or G_mgdL > GLUCOSE_SEVERE_HYPER_THRESHOLD:
            logger.warning(f"Simulation stopped at t={t:.1f} min due to extreme glucose {G_mgdL:.1f} mg/dL")
            return times[:i + 2], states[:i + 2]

    return times, states

def _physiology_only_config(cfg: NoiseConfig) -> NoiseConfig:
    """Disable all realism noise to isolate core ODE dynamics."""
    return dataclasses.replace(
        cfg,
        meal_logn_sigma=0.0,
        bolus_logn_sigma=0.0,
        missed_bolus_prob=0.0,
        pump_basal_rel_sigma=0.0,
        circadian_egp_amp_rel=0.0,
        sigma_gpgt_dL=0.0,
        sigma_ip=0.0,
        ou_sigma_dL=0.0,
        cgm_bias_mgdl=0.0,
        cgm_scale_bias=0.0,
        cgm_rw_sigma_bias=0.0,
        cgm_obs_sigma_mgdl=0.0,
        cgm_dropout_prob=0.0,
        enable=False,
    )

def _apply_t1d_x1_steady_state(x0: jnp.ndarray, params: PatientParams) -> jnp.ndarray:
    """Initialize T1D x1 to its basal steady state (I_conc - Ib)."""
    I_conc_pmol_L = x0[5] / params.Vi
    x1_ss = I_conc_pmol_L - params.Ib
    return x0.at[6].set(x1_ss)

def _log_equilibrium_residual(x0: jnp.ndarray, action: jnp.ndarray, params: PatientParams) -> None:
    """Log ||dx/dt|| at t=0 for a basal-only fixed point sanity check."""
    last_Qsto = 0.0
    last_foodtaken = 0.0
    if params.diabetes_type == PatientType.t1d:
        dxdt = hovorka_t1d(x0, action, params, last_Qsto, last_foodtaken)
    else:
        dxdt = hybrid_t2d(x0, action, params, last_Qsto, last_foodtaken)
    residual_l2 = float(jnp.linalg.norm(dxdt))
    residual_max = float(jnp.max(jnp.abs(dxdt)))
    logger.info(f"Equilibrium residual at t=0: ||dx/dt||={residual_l2:.4e}, max|dx/dt|={residual_max:.4e}")

def plot_states(times: np.ndarray, states: np.ndarray, params: PatientParams, fig_path: Optional[str] = None, csv_path: Optional[str] = None, log_step: int = 10):
    """
    Comprehensive plotting of T1D states and key derived variables.

    Args:
        times: Time array in minutes
        states: States array (16 x time)
        params: T1D model parameters
        fig_path: Optional path to save figure
        csv_path: Optional path to save CSV with values every 60 indexes
    """
    times_hours = times / 60.0
    fig = plt.figure(figsize=(15, 20), constrained_layout=True)
    gs = GridSpec(8, 2, figure=fig, hspace=1.0)

    # Collect data for CSV export (every 60 indexes)
    csv_data = {}
    csv_data['time_hours'] = times_hours[::log_step]

    # Plasma Glucose (mg/dL)
    ax_G = fig.add_subplot(gs[0, :])
    G_mgdL = states[:, 3] / params.Vg        # mg/dL
    # Collect for CSV
    csv_data['G_mgdL'] = G_mgdL[::log_step]
    ax_G.plot(times_hours, G_mgdL, label='G (mg/dL)')
    ax_G.set_ylim(40, 400)
    ax_G.set_ylabel('Glucose (mg/dL)')
    ax_G.set_title('Plasma Glucose')
    ax_G.axhspan(GLUCOSE_HYPOGLYCEMIA_THRESHOLD, GLUCOSE_HYPERGLYCEMIA_THRESHOLD, color='green', alpha=0.1)
    ax_G.axhspan(GLUCOSE_SEVERE_HYPO_THRESHOLD, GLUCOSE_HYPOGLYCEMIA_THRESHOLD, color='yellow', alpha=0.1)
    ax_G.axhspan(GLUCOSE_HYPERGLYCEMIA_THRESHOLD, GLUCOSE_SEVERE_HYPER_THRESHOLD, color='yellow', alpha=0.1)
    ax_G.legend()

    # Meal absorption states (g): gut compartments D1/D2/D3, not plasma glucose.
    ax_meal = fig.add_subplot(gs[1, 0])
    D1_g = states[:, 0] / 1000.0
    D2_g = states[:, 1] / 1000.0
    D3_g = states[:, 2] / 1000.0
    csv_data['D1_g'] = D1_g[::log_step]
    csv_data['D2_g'] = D2_g[::log_step]
    csv_data['D3_g'] = D3_g[::log_step]
    ax_meal.plot(times_hours, D1_g, label='D1 (g)', color='orange')
    ax_meal.plot(times_hours, D2_g, label='D2 (g)', color='red')
    ax_meal.plot(times_hours, D3_g, label='D3 (g)', color='brown')
    ax_meal.set_ylabel('Meal States (g)')
    ax_meal.set_title('Meal Absorption')
    ax_meal.legend()

    # Plasma Insulin (mU/L)
    ax_Ip = fig.add_subplot(gs[1, 1])
    # Ip total pmol -> concentration pmol/L = Ip / (Vi L/kg * BW kg), then mU/L = pmol/L / 6
    I_p_pmol_L = states[:, 5] / params.Vi    # pmol/L
    I_p_mU_L   = I_p_pmol_L / 6.0
    # Collect for CSV
    csv_data['I_p_mU_L'] = I_p_mU_L[::log_step]
    ax_Ip.plot(times_hours, I_p_mU_L, label='Ip (mU/L)')
    ax_Ip.set_ylabel('Plasma Insulin (mU/L)')
    ax_Ip.set_title('Plasma Insulin')
    ax_Ip.legend()

    # Insulin effects
    ax_eff1 = fig.add_subplot(gs[2, 0])
    # Collect for CSV
    csv_data['x1'] = states[:, 6][::log_step]
    csv_data['x2'] = states[:, 7][::log_step]
    ax_eff1.plot(times_hours, states[:, 6], label='x1 (remote)')
    ax_eff1.plot(times_hours, states[:, 7], label='x2 (interstitial)')
    ax_eff1.set_ylabel('Effects')
    ax_eff1.set_title('Insulin Effects 1')
    ax_eff1.legend()

    ax_eff2 = fig.add_subplot(gs[2, 1])
    # Collect for CSV
    csv_data['x3'] = states[:, 8][::log_step]
    ax_eff2.plot(times_hours, states[:, 8], label='x3 (disposal)')
    ax_eff2.set_ylabel('Effects')
    ax_eff2.set_title('Insulin Effects 2')
    ax_eff2.legend()

    # Insulin kinetics (pmol)
    ax_ins_kin = fig.add_subplot(gs[3, 0])
    # Collect for CSV
    csv_data['Il'] = states[:, 9][::log_step]
    csv_data['Isc1'] = states[:, 10][::log_step]
    csv_data['Isc2'] = states[:, 11][::log_step]
    ax_ins_kin.plot(times_hours, states[:, 9], label='Il (pmol)')
    ax_ins_kin.plot(times_hours, states[:, 10], label='Isc1 (pmol)')
    ax_ins_kin.plot(times_hours, states[:, 11], label='Isc2 (pmol)')
    ax_ins_kin.set_ylabel('Insulin (pmol)')
    ax_ins_kin.set_title('Insulin Kinetics')
    ax_ins_kin.legend()

    # Glucose compartments (mmol/kg): plasma/tissue glucose, not gut carbs.
    ax_Q = fig.add_subplot(gs[3, 1])
    Gp_mmol_per_kg = states[:, 3] / 180.0
    Gt_mmol_per_kg = states[:, 4] / 180.0
    # Collect for CSV
    csv_data['Gp_mmol_per_kg'] = Gp_mmol_per_kg[::log_step]
    csv_data['Gt_mmol_per_kg'] = Gt_mmol_per_kg[::log_step]
    ax_Q.plot(times_hours, Gp_mmol_per_kg, label='Gp (mmol/kg)')
    ax_Q.plot(times_hours, Gt_mmol_per_kg, label='Gt (mmol/kg)')
    ax_Q.set_ylabel('Glucose (mmol/kg)')
    ax_Q.set_title('Glucose Compartments')
    ax_Q.legend()

    # Gut glucose (Gsc mmol)
    Gsc_mmol_per_kg = states[:, 12] / 180.0

    ax_Gsc = fig.add_subplot(gs[4, 0])
    # Collect for CSV
    csv_data['Gsc'] = Gsc_mmol_per_kg[::log_step]
    ax_Gsc.plot(times_hours, Gsc_mmol_per_kg, label='Gsc (mmol)')
    ax_Gsc.set_ylabel('Gut Glucose (mmol)')
    ax_Gsc.set_title('Delayed Gut Glucose')
    ax_Gsc.legend()

    # Exercise states (zero in tests)
    ax_ex = fig.add_subplot(gs[4, 1])
    ax_ex.plot(times_hours, states[:, 13], label='E1 (HR)')
    ax_ex.plot(times_hours, states[:, 14], label='T_E (min)')
    ax_ex.plot(times_hours, states[:, 15], label='E2 (effect)')
    ax_ex.set_ylabel('Exercise States')
    ax_ex.set_title('Exercise Model')
    ax_ex.legend()

    # Set common x-label and grid
    for ax in fig.get_axes():
        ax.set_xlabel('Time (hours)')
        ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {fig_path}")
    else:
        plt.show()

    # Save CSV with values every 60 indexes
    if csv_path:
        df_csv = pd.DataFrame(csv_data)
        df_csv.to_csv(csv_path, index=False)
        logger.info(f"CSV data saved to {csv_path}")
        logger.info(f"CSV contains {len(df_csv)} rows (every 60 indexes) with columns: {list(df_csv.columns)}")


# --- Scenarios ---

def get_zero_action():
    return lambda t: jnp.array([0.0, 0.0, 0.0])

def get_meal_scenario(start_time=5.0, duration=15.0, amount_g=40.0):
    rate = amount_g / duration
    end_time = start_time + duration
    def action(t):
        if start_time <= t < end_time:
            return jnp.array([rate, 0.0, 0.0])
        return jnp.array([0.0, 0.0, 0.0])
    return action

def get_bolus_scenario(start_time=5.0, duration=1.0, amount_u=5.0):
    rate = amount_u / duration
    end_time = start_time + duration
    def action(t):
        if start_time <= t < end_time:
            return jnp.array([0.0, rate, 0.0])
        return jnp.array([0.0, 0.0, 0.0])
    return action

def get_exercise_scenario(params: PatientParams):
    basal_u_min = params.basal / 60.0
    def action(t):
        # 50 min ramp up, 30 min steady, 50 min ramp down
        hr_reserve = 0.0
        if 60 <= t < 110:
            hr_reserve = 0.5 * (t - 60) / 50.0
        elif 110 <= t < 140:
            hr_reserve = 0.5
        elif 140 <= t < 190:
            hr_reserve = 0.5 * (1.0 - (t - 140) / 50.0)
        
        # Keep basal on during exercise
        return jnp.array([0.0, basal_u_min, hr_reserve])
    return action


def main():
    parser = argparse.ArgumentParser(description="Unified Diabetes Simulator")
    parser.add_argument("--type", type=str, choices=["t1d", "t2d", "t2d_no_pump"], default="t1d", help="Patient Type")
    parser.add_argument("--name", type=str, default=None, help="Patient Name (e.g. adolescent#001)")
    parser.add_argument("--scenario", type=str, choices=["basal", "meal", "bolus", "exercise"], default="basal", help="Simulation Scenario")
    parser.add_argument("--hours", type=float, default=24.0, help="Simulation duration in hours")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--physiology_only", action="store_true", help="Disable all realism noise/jitter for sanity checks")
    
    args = parser.parse_args()

    # Defaults
    default_names = {
        "t1d": "adolescent#001",
        "t2d": "adolescent#001",
        "t2d_no_pump": "adolescent#001"
    }
    p_name = args.name if args.name else default_names[args.type]

    # Initialize Params
    logger.info(f"Initializing {args.type} patient: {p_name}")
    env_params = create_env_params(patient_name=p_name, diabetes_type=args.type)
    env_params, x0 = tune_initial_state(env_params)
    params = env_params.patient_params

    # 1) Physiology-only baseline (no realism noise/jitter).
    cfg = _physiology_only_config(env_params.noise_config) if args.physiology_only else env_params.noise_config
    if args.physiology_only:
        logger.info("Physiology-only mode enabled: realism noise and action jitter disabled.")

    # 2) Fix T1D x1 steady-state to remove artificial transients at t=0.
    if params.diabetes_type == PatientType.t1d:
        x0 = _apply_t1d_x1_steady_state(x0, params)

    # Select Scenario
    if args.scenario == "basal":
        # For basal test, ensure basal insulin is delivered if pump is used
        if params.use_pump:
             def basal_action(t):
                 return jnp.array([0.0, params.basal/60.0, 0.0])
             action_fn = basal_action
        else:
             action_fn = get_zero_action()
    elif args.scenario == "meal":
        action_fn = get_meal_scenario(amount_g=75.0)
    elif args.scenario == "bolus":
        action_fn = get_bolus_scenario(amount_u=5.0)
    elif args.scenario == "exercise":
        action_fn = get_exercise_scenario(params)

    # 3) One-line equilibrium residual check at t=0.
    _log_equilibrium_residual(x0, action_fn(0.0), params)
    
    # Run Simulation
    os.makedirs(args.output_dir, exist_ok=True)
    t_span = args.hours * 60.0
    key = jax.random.PRNGKey(42)
    
    times, states = simulate(
        t_span_min=t_span,
        dt_min=1.0,
        x0=x0,
        action_fn=action_fn,
        params=params,
        cfg=cfg,
        key=key
    )

    # Plot & Save
    base_name = f"{args.type}_{args.scenario}"
    plot_states(
        times, states, params, 
        fig_path=os.path.join(args.output_dir, f"{base_name}.png"),
        csv_path=os.path.join(args.output_dir, f"{base_name}.csv"),
        log_step=10
    )

if __name__ == "__main__":
    main()
