"""Deterministic RK stages, stable E2 substeps and physical-pool projections.

Inputs are rates in g/min and U/min plus dimensionless heart-rate reserve;
state units are defined by vector_fields. No PRNG, circadian modulation or
input disturbances are applied here. These helpers intentionally have no JIT
decorators: they trace inside the existing stochastic step JIT boundaries.
The E2 splitting makes the complete scheme lower order than classical RK4.
"""
import jax.numpy as jnp

from ..core.params import PatientParams
from .vector_fields import hovorka_t1d, hybrid_t2d


def _exercise_e2_step(x, dt, params):
    """Exponential E2 substep with E1/T_E frozen at the start of the step.

    E2 has units min; a is 1/min and b is min/min. This stabilizes the fast
    decay at resting T_E=c2 without changing the vector field. Coupling to
    E1/T_E is first order, so the complete method is not fourth-order RK4.
    """
    z = (x[13] / jnp.maximum(params.alpha_HR * params.HR0, 1e-6)) ** params.n_power
    f = z / (1.0 + z)
    time_scale = jnp.maximum(x[14], 1e-3)
    a = f / params.tau_in + 1.0 / time_scale
    b = f * time_scale / (params.c1 + params.c2)
    return x[15] * jnp.exp(-a * dt) + (b / a) * (-jnp.expm1(-a * dt))


def integrate_t1d(
    x: jnp.ndarray,
    dt: float,
    action: jnp.ndarray,
    params: PatientParams,
    last_Qsto: float,
    last_foodtaken: float,
) -> jnp.ndarray:
    """Advance deterministic T1D dynamics and project physical pools."""
    # 3) RK4 on the original static params
    # E2 can decay on a 0.01-minute scale. Use its stable affine solution
    # at every RK stage, not just at the endpoint (which leaves stiff stages).
    e2_half = _exercise_e2_step(x, dt / 2.0, params)
    e2_end = _exercise_e2_step(x, dt, params)
    k1 = hovorka_t1d(x, action, params, last_Qsto, last_foodtaken)
    stage2 = (x + (dt / 2.0) * k1).at[15].set(e2_half)
    k2 = hovorka_t1d(stage2, action, params, last_Qsto, last_foodtaken)
    stage3 = (x + (dt / 2.0) * k2).at[15].set(e2_half)
    k3 = hovorka_t1d(stage3, action, params, last_Qsto, last_foodtaken)
    stage4 = (x + dt * k3).at[15].set(e2_end)
    k4 = hovorka_t1d(stage4, action, params, last_Qsto, last_foodtaken)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next = x_next.at[15].set(e2_end)

    # clamps / projections
    POS = jnp.array([0,1,2, 3,4, 5,9, 10,11, 12])
    x_next = x_next.at[POS].set(jnp.maximum(x_next[POS], 0.0))

    return x_next


def integrate_t2d(
    x: jnp.ndarray,
    dt: float,
    action: jnp.ndarray,
    params: PatientParams,
    last_Qsto: float,
    last_foodtaken: float,
) -> jnp.ndarray:
    """Advance deterministic T2D dynamics and project physical pools."""
    # 3) RK4 on static params
    # E2 can decay on a 0.01-minute scale. Use its stable affine solution
    # at every RK stage, not just at the endpoint (which leaves stiff stages).
    e2_half = _exercise_e2_step(x, dt / 2.0, params)
    e2_end = _exercise_e2_step(x, dt, params)
    k1 = hybrid_t2d(x, action, params, last_Qsto, last_foodtaken)
    stage2 = (x + (dt / 2.0) * k1).at[15].set(e2_half)
    k2 = hybrid_t2d(stage2, action, params, last_Qsto, last_foodtaken)
    stage3 = (x + (dt / 2.0) * k2).at[15].set(e2_half)
    k3 = hybrid_t2d(stage3, action, params, last_Qsto, last_foodtaken)
    stage4 = (x + dt * k3).at[15].set(e2_end)
    k4 = hybrid_t2d(stage4, action, params, last_Qsto, last_foodtaken)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    x_next = x_next.at[15].set(e2_end)

    # clamps / projections similar to your t2d_rk4_step
    POS = jnp.array([0,1,2, 3,4, 5,9, 10,11, 12, 16])
    x_next = x_next.at[POS].set(jnp.maximum(x_next[POS], 0.0))

    return x_next
