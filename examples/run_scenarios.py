import sys
import os
# Prefer this checkout when running the script directly from source.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import argparse
import dataclasses
import logging
import json
import tempfile
from typing import Optional, Tuple, Callable

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

from glucosim.simglucose.core.params import PatientParams, create_env_params, NoiseConfig
from glucosim.simglucose.physiology.initialization import tune_initial_state
from glucosim.simglucose.physiology.glucose_dynamics import (
    t1d_rk4_step,
    t2d_rk4_step,
    hovorka_t1d,
    hybrid_t2d,
)
from glucosim.simglucose.core.types import PatientType

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
    if not np.isfinite(dt_min) or dt_min <= 0:
        raise ValueError("dt_min must be finite and positive")
    if not np.isfinite(t_span_min) or t_span_min < 0:
        raise ValueError("t_span_min must be finite and nonnegative")
    # Include a final remainder step; timestamps are actual elapsed minutes.
    times = np.append(np.arange(0.0, t_span_min, dt_min), t_span_min)
    num_steps = len(times) - 1
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
        step_dt = times[i + 1] - t
        action = action_fn(t)  # [carb, insulin, hr_reserve]
        carb = float(action[0])

        # Meal tracking
        if carb > 0 and prev_carb == 0:
            last_Qsto = float(states[i, 0] + states[i, 1])  # D1 + D2 (mg)
            last_foodtaken = 0.0
            logger.debug(f"New meal at t={t:.1f} min: last_Qsto={last_Qsto:.1f} mg")

        last_foodtaken += carb * step_dt  # g/min -> grams consumed this step

        # Step integration
        key, sk = jax.random.split(key)
        
        if is_t1d:
             x, key, ou_state_dL = step_fn(
                x=x, dt=step_dt, action=action, params=params,
                last_Qsto=last_Qsto, last_foodtaken=last_foodtaken,
                t_min=t0_min + t, key=sk, cfg=cfg, ou_state_dL=ou_state_dL
            )
        else:
            # T2D step signature includes day_key
            x, key, ou_state_dL = step_fn(
                x=x, dt=step_dt, action=action, params=params,
                last_Qsto=last_Qsto, last_foodtaken=last_foodtaken,
                t_min=t0_min + t, key=sk, cfg=cfg, ou_state_dL=ou_state_dL
            )
            
        states[i + 1] = np.array(x)

        if carb == 0 and prev_carb > 0:
            # Retain the meal size for gastric emptying until the next meal.
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

    # Insulin kinetics (pmol/kg)
    ax_ins_kin = fig.add_subplot(gs[3, 0])
    # Collect for CSV
    csv_data['Il'] = states[:, 9][::log_step]
    csv_data['Isc1'] = states[:, 10][::log_step]
    csv_data['Isc2'] = states[:, 11][::log_step]
    ax_ins_kin.plot(times_hours, states[:, 9], label='Il (pmol/kg)')
    ax_ins_kin.plot(times_hours, states[:, 10], label='Isc1 (pmol/kg)')
    ax_ins_kin.plot(times_hours, states[:, 11], label='Isc2 (pmol/kg)')
    ax_ins_kin.set_ylabel('Insulin (pmol/kg)')
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

    # Filtered subcutaneous glucose (mmol/kg)
    Gsc_mmol_per_kg = states[:, 12] / 180.0

    ax_Gsc = fig.add_subplot(gs[4, 0])
    # Collect for CSV
    csv_data['Gsc'] = Gsc_mmol_per_kg[::log_step]
    ax_Gsc.plot(times_hours, Gsc_mmol_per_kg, label='Gsc (mmol/kg)')
    ax_Gsc.set_ylabel('Glucose (mmol/kg)')
    ax_Gsc.set_title('Filtered Subcutaneous Glucose')
    ax_Gsc.legend()

    # Exercise states (zero in tests)
    ax_ex = fig.add_subplot(gs[4, 1])
    ax_ex.plot(times_hours, states[:, 13], label='E1 (bpm)')
    ax_ex.plot(times_hours, states[:, 14], label='T_E (min)')
    ax_ex.plot(times_hours, states[:, 15], label='E2 (min)')
    ax_ex.set_ylabel('Exercise States')
    ax_ex.set_title('Exercise Model')
    ax_ex.legend()

    # Set common x-label and grid
    for ax in fig.get_axes():
        ax.set_xlabel('Time (hours)')
        ax.grid(True, linestyle='--', alpha=0.7)

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


def save_glucose_gif(times, states, params, path):
    """Animate plasma glucose using at most 120 frames; no ffmpeg required."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    hours = times / 60.0
    glucose = states[:, 3] / params.Vg
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.set(xlabel="Time (hours)", ylabel="Plasma glucose (mg/dL)",
           xlim=(0, max(float(hours[-1]), 1 / 60)),
           ylim=(min(40, float(glucose.min()) - 10),
                 max(400, float(glucose.max()) + 10)))
    ax.axhspan(GLUCOSE_HYPOGLYCEMIA_THRESHOLD,
               GLUCOSE_HYPERGLYCEMIA_THRESHOLD, color="green", alpha=0.1)
    line, = ax.plot([], [], label="Plasma glucose")
    ax.legend()

    def update(index):
        line.set_data(hours[:index + 1], glucose[:index + 1])
        return (line,)

    frames = np.linspace(0, len(times) - 1, min(120, len(times)), dtype=int)
    animation = FuncAnimation(fig, update, frames=frames, interval=100)
    try:
        animation.save(path, writer=PillowWriter(fps=10), dpi=100)
    finally:
        plt.close(fig)
    logger.info(f"GIF saved to {path}")


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
    def action(t):
        # 50 min ramp up, 30 min steady, 50 min ramp down
        hr_reserve = 0.0
        if 60 <= t < 110:
            hr_reserve = 0.5 * (t - 60) / 50.0
        elif 110 <= t < 140:
            hr_reserve = 0.5
        elif 140 <= t < 190:
            hr_reserve = 0.5 * (1.0 - (t - 140) / 50.0)
        
        # The ODE already supplies basal insulin; the action is additional insulin.
        return jnp.array([0.0, 0.0, hr_reserve])
    return action


def get_meal_bolus_scenario(meal_g, bolus_u):
    """A 15-minute meal at minute 30, with a one-minute SC bolus at minute 30."""
    meal = get_meal_scenario(start_time=30., duration=15., amount_g=meal_g)
    bolus = get_bolus_scenario(start_time=30., duration=1., amount_u=bolus_u)
    return lambda t: meal(t) + bolus(t)


def response_metrics(times, states, params, horizon_min):
    """Time-weighted plasma-glucose diagnostics; incomplete runs are ineligible."""
    glucose = np.asarray(states[:, 3] / params.Vg)
    finite = bool(np.isfinite(states).all() and np.isfinite(glucose).all())
    complete = bool(finite and np.isclose(times[-1], horizon_min, rtol=0, atol=1e-6))
    widths = np.diff(times)
    duration = float(times[-1] - times[0])
    # Trapezoidal area; interval endpoint average for time-in-range indicators.
    def area(values):
        return float(np.sum(.5 * (values[:-1] + values[1:]) * widths))
    if not finite or duration <= 0:
        return dict(complete=False, eligible=False, min_mgdl=np.nan, peak_mgdl=np.nan,
                    final_mgdl=np.nan, tir_70_180_pct=np.nan, below_70_min=np.nan,
                    hyper_auc_mgdl_min=np.nan, simulated_min=duration)
    return dict(
        complete=complete, eligible=complete and bool(np.min(glucose) >= 70.),
        min_mgdl=float(np.min(glucose)), peak_mgdl=float(np.max(glucose)),
        final_mgdl=float(glucose[-1]),
        tir_70_180_pct=100. * area(((glucose >= 70.) & (glucose <= 180.)).astype(float)) / duration,
        below_70_min=area((glucose < 70.).astype(float)),
        hyper_auc_mgdl_min=area(np.maximum(glucose - 180., 0.)), simulated_min=duration,
    )


def select_bolus(rows):
    """Lowest hyperglycemic area among complete, non-hypoglycemic candidates.

    Ties favor the smaller bolus. This is retrospective simulator selection,
    not a clinical recommendation or an online controller.
    """
    eligible = [row for row in rows if row['eligible']]
    return min(eligible, key=lambda row: (row['hyper_auc_mgdl_min'], row['bolus_u'])) if eligible else None


def run_suite(args):
    """Cross diabetes type x meal size x bolus grid, using deterministic physiology."""
    types = ('t1d', 't2d', 't2d_no_pump')
    meals = sorted(set(args.meals))
    doses = sorted(set([0.] + args.boluses))
    horizon = args.hours * 60.
    if not np.isfinite(horizon) or horizon < 60.:
        raise ValueError('Suite --hours must be finite and at least 1 (default: 6)')
    if not meals or any(not np.isfinite(g) or g <= 0 for g in meals):
        raise ValueError('--meals must contain finite positive grams')
    if any(not np.isfinite(u) or u < 0 for u in doses):
        raise ValueError('--boluses must contain finite nonnegative units')
    os.makedirs(args.output_dir, exist_ok=True)
    # Preserve earlier experiment outputs on repeated invocations.
    output = tempfile.mkdtemp(prefix='meal_bolus_', dir=args.output_dir)
    rows, recommendations, traces = [], [], []
    fig, axes = plt.subplots(len(types), len(meals), figsize=(5 * len(meals), 10),
                             squeeze=False, constrained_layout=True)
    config = dict(seed=args.seed, hours=args.hours, patient=args.name or 'adolescent#001',
                  types=types, meal_grams=meals, candidate_bolus_units=doses,
                  meal_start_min=30, meal_duration_min=15, bolus_duration_min=1,
                  physiology_only=True, glucose_source='plasma',
                  selection='Complete horizon, min glucose >=70; minimize AUC above 180; tie: smaller dose')
    with open(os.path.join(output, 'config.json'), 'w') as file:
        json.dump(config, file, indent=2)
    try:
        for type_index, kind in enumerate(types):
            env, x0 = tune_initial_state(create_env_params(patient_name=config['patient'], diabetes_type=kind))
            p = env.patient_params
            cfg = _physiology_only_config(env.noise_config)
            allowed = [u for u in doses if u <= p.max_bolus_U]
            def evaluate(meal, dose):
                times, states = simulate(horizon, 1., x0, get_meal_bolus_scenario(meal, dose),
                                         p, cfg, jax.random.PRNGKey(args.seed))
                metrics = response_metrics(times, states, p, horizon)
                row = dict(diabetes_type=kind, patient=config['patient'], meal_g=meal,
                           bolus_u=dose, seed=args.seed, BW_kg=float(p.BW),
                           basal_u_hr=float(p.basal), **metrics)
                rows.append(row)
                traces.append(pd.DataFrame(dict(diabetes_type=kind, meal_g=meal, bolus_u=dose,
                                               time_min=times, plasma_mgdl=states[:, 3] / p.Vg)))
                return row, times, states
            _, basal_times, basal_states = evaluate(0., 0.)
            for meal_index, meal in enumerate(meals):
                candidates = [evaluate(meal, dose) for dose in allowed]
                chosen = select_bolus([item[0] for item in candidates])
                recommendations.append(dict(diabetes_type=kind, meal_g=meal,
                    recommended_bolus_u=chosen['bolus_u'] if chosen else np.nan,
                    status='selected_on_simulated_grid' if chosen else 'no_eligible_dose',
                    tested_doses_u=';'.join(map(str, allowed)),
                    hyper_auc_mgdl_min=chosen['hyper_auc_mgdl_min'] if chosen else np.nan))
                ax = axes[type_index, meal_index]
                ax.axhspan(70, 180, color='green', alpha=.08)
                ax.axvline(.5, color='gray', linewidth=.8)
                ax.plot(basal_times / 60., basal_states[:, 3] / p.Vg, ':', label='No meal / no bolus')
                _, times, states = candidates[0]  # zero dose is always included
                ax.plot(times / 60., states[:, 3] / p.Vg, label='Meal / no bolus')
                if chosen is not None:
                    _, times, states = next(item for item in candidates if item[0] is chosen)
                    ax.plot(times / 60., states[:, 3] / p.Vg, '--', label=f"Selected {chosen['bolus_u']:g} U")
                    if args.gif:
                        save_glucose_gif(times, states, p, os.path.join(output, f'{kind}_meal_{meal:g}g.gif'))
                else:
                    ax.text(.02, .96, 'No eligible dose', transform=ax.transAxes, va='top')
                ax.set(title=f'{kind}: {meal:g} g', xlabel='Time (hours)', ylabel='Plasma glucose (mg/dL)')
                ax.legend(fontsize=8)
        fig.suptitle('Meal and bolus response — deterministic simulator grid search')
        fig.savefig(os.path.join(output, 'comparison.png'), dpi=150)
        pd.DataFrame(rows).to_csv(os.path.join(output, 'candidate_metrics.csv'), index=False)
        pd.DataFrame(recommendations).to_csv(os.path.join(output, 'recommendations.csv'), index=False)
        pd.concat(traces, ignore_index=True).to_csv(os.path.join(output, 'glucose_traces.csv'), index=False)
    finally:
        plt.close(fig)
    print(f'Saved {len(rows)} simulations and {len(recommendations)} meal comparisons to {output}')
    print('Bolus selections are retrospective simulator results, not patient dosing advice.')
    return output


def main():
    parser = argparse.ArgumentParser(description="Unified Diabetes Simulator")
    parser.add_argument("--type", type=str, choices=["t1d", "t2d", "t2d_no_pump"], default="t1d", help="Patient Type")
    parser.add_argument("--name", type=str, default=None, help="Patient Name (e.g. adolescent#001)")
    parser.add_argument("--scenario", type=str, choices=["basal", "meal", "bolus", "exercise"], default="basal", help="Simulation Scenario")
    parser.add_argument("--hours", type=float, default=None, help="Simulation hours (default: 24 single / 6 suite)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--gif", action="store_true", help="Also save an animated plasma-glucose GIF in --output_dir")
    parser.add_argument("--physiology_only", action="store_true", help="Disable all realism noise/jitter for sanity checks")
    
    parser.add_argument('--suite', action='store_true', help='Compare all three diabetes types across meals and bolus doses; deterministic physiology')
    parser.add_argument('--meals', type=float, nargs='+', default=[30., 60., 90.], help='Suite meal sizes in grams')
    parser.add_argument('--boluses', type=float, nargs='+', default=[0., 2., 4., 6., 8., 10.], help='Suite bolus grid in U; zero always included, values above patient max excluded')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.hours = args.hours if args.hours is not None else (6. if args.suite else 24.)
    if args.suite:
        run_suite(args)
        return

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
        # Basal insulin is added by the ODE, not by the action.
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
    key = jax.random.PRNGKey(args.seed)
    
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
    if args.gif:
        save_glucose_gif(times, states, params,
                         os.path.join(args.output_dir, f"{base_name}.gif"))

if __name__ == "__main__":
    main()
