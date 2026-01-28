import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial

# Cohort-specific meal profiles (mean grams and variability).
CHILD_MEAL_MU = jnp.asarray([30, 10, 40, 10, 35, 10], dtype=jnp.float32)
CHILD_MEAL_SIGMA = jnp.asarray([5, 5, 5, 5, 5, 5], dtype=jnp.float32)

ADULT_MEAL_MU = jnp.array([45, 10, 70, 10, 80, 10], dtype=jnp.float32)
ADULT_MEAL_SIGMA = jnp.array([10, 5, 10, 5, 10, 5], dtype=jnp.float32)
        
ADOLESCENT_MEAL_MU = jnp.array([45, 10, 70, 10, 80, 10], dtype=jnp.float32)
ADOLESCENT_MEAL_SIGMA = jnp.array([10, 5, 10, 5, 10, 5], dtype=jnp.float32)


def get_meal_profile_for_cohort(cohort_key: str):
    """Return (mu, sigma) arrays for the requested cohort key."""
    cohort_key = (cohort_key or "").lower()
    if cohort_key.startswith("child"):
        return CHILD_MEAL_MU, CHILD_MEAL_SIGMA
    if cohort_key.startswith("adolescent"):
        return ADOLESCENT_MEAL_MU, ADOLESCENT_MEAL_SIGMA
    # Default to adult/baseline if unknown
    return ADULT_MEAL_MU, ADULT_MEAL_SIGMA


@partial(jit, static_argnums=(3,))
def create_multiday_scenario_jax(
    key: jnp.ndarray,
    amount_mu: jnp.ndarray,
    amount_sigma: jnp.ndarray,
    num_days: int
) -> jnp.ndarray:
    """
    Creates a JAX-native random meal scenario for multiple days.
    Cohort-specific meal sizes are passed in via amount_mu/amount_sigma.
    """
    
    def create_one_day(day_key):
        """Generates a meal schedule for a single day."""
        probs = jnp.array([0.8, 0.3, 0.95, 0.3, 0.95, 0.3])
        time_mu = jnp.array([7, 9.5, 12, 15, 18, 21.5]) * 60
        time_sigma = jnp.array([60, 30, 60, 30, 60, 30])
        
        dkey, subkey_prob, subkey_time, subkey_amount = random.split(day_key, 4)
        occurs = random.uniform(subkey_prob, shape=probs.shape) < probs
        
        time_normals = random.normal(subkey_time, shape=time_mu.shape)
        meal_times = time_mu + time_normals * time_sigma
        
        amount_normals = random.normal(subkey_amount, shape=amount_mu.shape)
        meal_amounts = jnp.maximum(0, jnp.round(amount_mu + amount_normals * amount_sigma))
        
        final_times = jnp.where(occurs, meal_times, -1)
        final_amounts = jnp.where(occurs, meal_amounts, 0)
        
        return jnp.stack([final_times, final_amounts], axis=1)

    day_keys = random.split(key, num_days)
    all_days_scenarios = jax.vmap(create_one_day)(day_keys)
    return all_days_scenarios
