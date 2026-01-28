import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def risk_index(bg_trace: jnp.ndarray) -> jnp.ndarray:
    """A simplified, JAX-native version of the blood glucose risk index."""
    scaled_bg = 1.509 * (jnp.log(bg_trace)**1.084 - 5.381)
    risk = 10 * (scaled_bg**2)
    return jnp.mean(risk)

@partial(jax.jit, static_argnames=("dt",))
def glucose_variability_metrics(trace: jnp.ndarray, dt: float):
    """
    trace: (T,) glucose forecast in mg/dL
    dt: minutes per step (e.g., 5.0)

    Returns:
      sd_mgdl, cv_pct, mag_mgdl_per_min, mage_mgdl
    """
    trace = jnp.asarray(trace)
    T = trace.shape[0]

    # ----- SD & CV -----
    mean_g = jnp.mean(trace)
    sd_g = jnp.std(trace)
    cv_pct = 100.0 * sd_g / (mean_g + 1e-6)

    # ----- MAG: mean absolute glucose change per minute -----
    dg = trace[1:] - trace[:-1]
    mag = jnp.mean(jnp.abs(dg)) / (dt + 1e-6)  # mg/dL per min

    # ----- MAGE (approx): average excursion size of peak<->trough swings > 1*SD -----
    # Identify turning points: sign change in first difference.
    # d1[t] = trace[t+1]-trace[t] for t=0..T-2
    d1 = dg
    s = jnp.sign(d1)
    s = jnp.where(s == 0.0, 1.0, s)  # treat flats as + to avoid missing changes

    # turn at index i (1..T-2) if sign flips between s[i-1] and s[i]
    # (d1 has length T-1; turning points live in trace indices 1..T-2)
    flip = (s[:-1] * s[1:]) < 0.0  # length T-2, corresponds to trace index 1..T-2

    # We'll scan through trace indices 1..T-2 and build excursions between alternating extrema.
    # Keep last extrema value and whether it's a peak/trough.
    idxs = jnp.arange(1, T-1)  # candidate indices

    def scan_fn(carry, x):
        last_ext_val, last_ext_is_peak, sum_exc, count_exc = carry
        i = x

        is_turn = flip[i - 1]  # flip aligned to trace index i
        # classify the turning point using slope before/after:
        # peak if slope goes + to -, trough if - to +
        prev_sign = s[i - 1]
        next_sign = s[i]
        is_peak = (prev_sign > 0) & (next_sign < 0)
        # trough if prev<0 and next>0; if neither (rare due to flats), keep last flag
        is_peak = jnp.where(is_turn, is_peak, last_ext_is_peak)

        val = trace[i]

        def handle_turn(_):
            exc = jnp.abs(val - last_ext_val)
            keep = exc > sd_g  # classic MAGE threshold = 1 SD
            sum_exc2 = sum_exc + jnp.where(keep, exc, 0.0)
            count_exc2 = count_exc + jnp.where(keep, 1.0, 0.0)
            return (val, is_peak, sum_exc2, count_exc2)

        def handle_no(_):
            return (last_ext_val, last_ext_is_peak, sum_exc, count_exc)

        return jax.lax.cond(is_turn, handle_turn, handle_no, operand=None), None

    # Initialize last_ext_val as trace[0] to seed the first excursion.
    init = (trace[0], False, 0.0, 0.0)
    (last_ext_val, last_ext_is_peak, sum_exc, count_exc), _ = jax.lax.scan(scan_fn, init, idxs)

    mage = sum_exc / (count_exc + 1e-6)

    return sd_g, cv_pct, mag, mage


@partial(jax.jit, static_argnames=("low", "high"))
def time_in_range(trace: jnp.ndarray, low: float = 70.0, high: float = 180.0) -> jnp.ndarray:
    """Return the fraction of glucose readings within the target range."""
    trace = jnp.asarray(trace)
    in_range = (trace >= low) & (trace <= high)
    return jnp.mean(in_range)
