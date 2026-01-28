from typing import Tuple, Dict

import jax.numpy as jnp
from jax import random, lax

from ..core.params import PatientParams, NoiseConfig

# ============================
# Small utilities
# ============================
def _splitn(key: jnp.ndarray, n: int):
    ks = random.split(key, n + 1)
    return ks[0], list(ks[1:])

def _ln_jitter(key: jnp.ndarray, sigma: float, value: jnp.ndarray) -> jnp.ndarray:
    """log-normal multiplicative jitter on a (possibly 0-D) array."""
    return lax.cond(
        sigma <= 0.0,
        lambda: value,
        lambda: value * jnp.exp(sigma * random.normal(key, shape=()))
    )

def _circadian_scale(t_min: jnp.ndarray,
                     amp_rel: float,
                     phase_min: float,
                     period_min: float) -> jnp.ndarray:
    """JAX scalar circadian factor."""
    def _apply_circadian():
        w = 2.0 * jnp.pi / period_min
        return 1.0 + amp_rel * jnp.sin(w * (t_min - phase_min))
        
    return lax.cond(
        amp_rel == 0.0,
        lambda: jnp.array(1.0),
        _apply_circadian
    )


# ============================
# Action realism
# ============================
def disturb_action(action: jnp.ndarray,
                   params: PatientParams,
                   key: jnp.ndarray,
                   cfg: NoiseConfig) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Return (noisy_action, new_key).
    Inject basal wobble as action delta (U/min) instead of mutating params.
    """
    if not cfg.enable:
        return action, key

    key, (k_meal, k_bolus, k_miss, k_basal, k_hr) = _splitn(key, 5)

    meal = _ln_jitter(k_meal, cfg.meal_logn_sigma, action[0])

    miss = random.bernoulli(k_miss, cfg.missed_bolus_prob)
    insulin = jnp.where(miss, 0.0, _ln_jitter(k_bolus, cfg.bolus_logn_sigma, action[1]))

    # pump basal wobble: add a small delta to insulin action stream
    wobble = jnp.where(
        params.use_pump,
        (params.basal / 60.0) * (_ln_jitter(k_basal, cfg.pump_basal_rel_sigma, jnp.array(1.0)) - 1.0),
        0.0,
    )
    insulin = insulin + wobble

    hr = jnp.clip(action[2] + 0.02 * random.normal(k_hr, shape=()), 0.0, 1.0)

    noisy = jnp.stack([meal, insulin, hr]).astype(action.dtype)
    return noisy, key


# ============================
# Dynamic factors per step
# ============================
def dynamic_factors_for_step(t_min: jnp.ndarray,
                             cfg: NoiseConfig) -> Dict[str, jnp.ndarray]:
    """
    Return JAX scalars for dynamic effects; do not mutate params inside jit.
    """
    if not cfg.enable:
        return {"circadian": jnp.array(1.0)}
    return {
        "circadian": _circadian_scale(t_min, cfg.circadian_egp_amp_rel,
                                      cfg.circadian_phase_min, cfg.circadian_period_min)
    }


# ============================
# Process noise
# ============================
def _white_exchange_eps(key: jnp.ndarray, dt: jnp.ndarray,
                        sigma_dL: float, Vg: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    White noise epsilon in mg/kg generated from mg/dL sigma and Vg.
    """
    key, (k,) = _splitn(key, 1)
    sdt = jnp.sqrt(jnp.maximum(dt, 1e-6))
    eps_dL = sigma_dL * sdt * random.normal(k, shape=())
    eps_mgkg = eps_dL * Vg
    return eps_mgkg, key

def _ou_exchange_eps(key: jnp.ndarray, dt: jnp.ndarray,
                     theta: float, sigma_dL: float,
                     Vg: float, ou_state_dL: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    OU on mg/dL; return (eps_mgkg, new_ou_state_dL, new_key).
    """
    key, (k,) = _splitn(key, 1)
    e = jnp.exp(-theta * dt)
    sd = jnp.sqrt((sigma_dL ** 2) * (1.0 - e * e))
    new_state_dL = e * ou_state_dL + sd * random.normal(k, shape=())
    eps_mgkg = new_state_dL * Vg
    return eps_mgkg, new_state_dL, key

def add_process_noise_structured(x_next: jnp.ndarray,
                                 dt: jnp.ndarray,
                                 key: jnp.ndarray,
                                 cfg: NoiseConfig,
                                 params: PatientParams,
                                 ou_state_dL: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Structured push/pull on (Gp,Gt): (-ε, +ε), tiny Ip diffusion.
    Returns (x_noisy, new_key, new_ou_state_dL).
    """
    if not cfg.enable:
        return x_next, key, ou_state_dL

    if cfg.use_ou:
        eps_mgkg, ou_state_dL, key = _ou_exchange_eps(key, dt, cfg.ou_theta, cfg.ou_sigma_dL, params.Vg, ou_state_dL)
    else:
        eps_mgkg, key = _white_exchange_eps(key, dt, cfg.sigma_gpgt_dL, params.Vg)

    key, (k_ip,) = _splitn(key, 1)
    sdt = jnp.sqrt(jnp.maximum(dt, 1e-6))
    dip = cfg.sigma_ip * sdt * random.normal(k_ip, shape=())

    x = x_next
    x = x.at[3].add(-eps_mgkg)  # Gp
    x = x.at[4].add(+eps_mgkg)  # Gt
    x = x.at[5].add(dip)        # Ip

    # Re-project physical pools >= 0
    POS = jnp.array([0,1,2, 3,4, 5,9, 10,11, 12])
    x = x.at[POS].set(jnp.maximum(x[POS], 0.0))
    return x, key, ou_state_dL