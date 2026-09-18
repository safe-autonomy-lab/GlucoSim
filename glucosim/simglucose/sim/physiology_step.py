"""Order action disturbances, deterministic integration, circadian flow and noise.

Keep these existing JIT boundaries and PRNG calls in their original order.
The legacy physiology.glucose_dynamics import paths re-export these functions.
"""
from typing import Tuple

from jax import jit
import jax.numpy as jnp

from ..core.params import PatientParams, NoiseConfig
from ..physiology.integration import integrate_t1d, integrate_t2d
from .realism import dynamic_factors_for_step, add_process_noise_structured, disturb_action


@jit
def t1d_rk4_step(
    x: jnp.ndarray,
    dt: float,
    action: jnp.ndarray,          # [carb_g/min, insulin_U/min, hr_reserve]
    params: PatientParams,
    last_Qsto: float,
    last_foodtaken: float,
    t_min: float,                 # absolute minute (for circadian)
    key: jnp.ndarray,
    cfg: NoiseConfig,
    ou_state_dL: jnp.ndarray      # scalar jnp array for OU state (mg/dL)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Wrapper around hovorka_t1d with action jitter, circadian delta, structured process noise.
    Returns (x_next, key, new_ou_state_dL).
    """

    # 1) action jitter (inject basal wobble as action delta)
    action_jit, key = disturb_action(action, params, key, cfg)

    # 2) dynamic factors (do not mutate params)
    factors = dynamic_factors_for_step(jnp.asarray(t_min), cfg)
    circ = factors["circadian"]  # JAX scalar

    # 3) Deterministic integration and physical-pool projection
    x_next = integrate_t1d(x, dt, action_jit, params, last_Qsto, last_foodtaken)

    # 5) Circadian delta for T1D: modulate kp1 as additive ΔEGP on Gp (exact for constant term over dt)
    delta_egp = (circ - 1.0) * params.kp1  # mg/kg/min
    x_next = x_next.at[3].add(dt * delta_egp)

    # 6) structured process noise
    x_next, key, ou_state_dL = add_process_noise_structured(
        x_next, jnp.asarray(dt), key, cfg, params, ou_state_dL
    )

    return x_next, key, ou_state_dL


@jit
def t2d_rk4_step(
    x: jnp.ndarray,
    dt: float,
    action: jnp.ndarray,          # [carb_g/min, insulin_U/min, hr_reserve]
    params: PatientParams,
    last_Qsto: float,
    last_foodtaken: float,
    t_min: float,                 # absolute minute (for circadian)
    key: jnp.ndarray,
    cfg: NoiseConfig,
    ou_state_dL: jnp.ndarray      # scalar jnp array for OU state (mg/dL)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Wrapper around hybrid_t2d with action jitter, circadian delta on EGP_0*(1-x3),
    structured process noise. Returns (x_next, key, new_ou_state_dL).
    """
    # 1) action jitter
    action_jit, key = disturb_action(action, params, key, cfg)

    # 2) dynamic factors
    factors = dynamic_factors_for_step(jnp.asarray(t_min), cfg)
    circ = factors["circadian"]

    # 3) Deterministic integration and physical-pool projection
    x_next = integrate_t2d(x, dt, action_jit, params, last_Qsto, last_foodtaken)

    # 4) Circadian delta for T2D: base EGP term is (EGP_0 * 180 / BW) * (1 - x3_eff)
    #    We approximate its modulation by adding dt * (circ-1) * EGP0_mgkgmin * (1 - x3_eff)
    EGP0_mgkgmin = (params.EGP_0 * 180.0) / params.BW
    x3_eff = jnp.clip(x_next[8], 0.0, 0.95)
    delta_egp = (circ - 1.0) * EGP0_mgkgmin * (1.0 - x3_eff)
    x_next = x_next.at[3].add(dt * delta_egp)

    # 5) structured process noise
    x_next, key, ou_state_dL = add_process_noise_structured(
        x_next, jnp.asarray(dt), key, cfg, params, ou_state_dL
    )

    return x_next, key, ou_state_dL
