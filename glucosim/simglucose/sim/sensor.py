from typing import Tuple
import jax
import jax.numpy as jnp

from ..core.params import NoiseConfig
from ..sim.realism import _splitn

# ============================
# CGM observation
# ============================
def cgm_measurement(
    Gp_mgkg: jnp.ndarray,
    Vg_dL_per_kg: float,
    key: jnp.ndarray,
    cfg: NoiseConfig,
    scale_state: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Return (reading_mgdl_or_nan, new_scale_state, new_key) as JAX scalars.
    """
    if not cfg.enable:
        gb = Gp_mgkg / jnp.maximum(Vg_dL_per_kg, 1e-6)
        return gb, scale_state, key

    key, (k_rw, k_eps, k_drop) = _splitn(key, 3)
    scale = jnp.clip(scale_state + cfg.cgm_rw_sigma_bias * jax.random.normal(k_rw, shape=()), 0.9, 1.1)

    gb_mgdl = Gp_mgkg / jnp.maximum(Vg_dL_per_kg, 1e-6)
    mean = (gb_mgdl + cfg.cgm_bias_mgdl) * (1.0 + cfg.cgm_scale_bias) * scale
    obs = mean + cfg.cgm_obs_sigma_mgdl * jax.random.normal(k_eps, shape=())
    dropout = jax.random.bernoulli(k_drop, cfg.cgm_dropout_prob)
    reading = jnp.where(dropout, jnp.nan, obs)
    reading = jnp.clip(reading, 40.0, 400.0)
    return reading, scale, key
