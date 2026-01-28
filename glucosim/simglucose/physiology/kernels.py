import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=('steps'))
def create_carb_gamma_kernel(steps, units, shape=3.0, scale=20.0/5.0):
    """Build a normalized gamma kernel to distribute carb grams over future timesteps."""
    t = jnp.arange(0, steps, dtype=jnp.float32)
    kernel = t**(shape - 1) * jnp.exp(-t / scale)
    kernel_sum = jnp.sum(kernel)
    return jax.lax.cond(kernel_sum == 0,
                        lambda k: k,
                        lambda k: k / kernel_sum * units,
                        kernel)

@partial(jax.jit, static_argnames=('dt_mins', 'duration_hours', 'peak_time_mins'))
def create_insulin_kernel(dt_mins=1, duration_hours=6, peak_time_mins=75):
    """Return IOB decay and activity kernels that approximate real insulin pharmacodynamics."""
    length = int((duration_hours * 60) / dt_mins)
    t = jnp.arange(0, length, dtype=jnp.float32) * dt_mins
    
    tau = peak_time_mins * (1 - peak_time_mins / (duration_hours * 60))
    activity_curve = (t / tau) * jnp.exp(1 - (t / tau))
    activity_curve = activity_curve / jnp.sum(activity_curve)
    
    spent_insulin = jnp.cumsum(activity_curve)
    iob_kernel = 1.0 - spent_insulin
    iob_kernel = jnp.where(iob_kernel < 0, 0.0, iob_kernel)
    
    return iob_kernel, activity_curve
