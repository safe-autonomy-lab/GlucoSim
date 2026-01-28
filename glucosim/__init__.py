from functools import partial
from glucosim.simglucose.sim.env import JaxSimEnv
from jax import random
from .gym_env.utils.registration import register

def _make_env_fn(env_id, simulation_hours=24, **kwargs):
    key = random.PRNGKey(0)
    simulation_minutes = kwargs.pop("simulation_minutes", simulation_hours * 60)
    return JaxSimEnv(env_id, key, simulation_minutes=simulation_minutes, **kwargs)


register(
    id='t1d-v0',
    entry_point=partial(_make_env_fn, env_id='t1d-v0'),
)

register(
    id='t2d-v0',
    entry_point=partial(_make_env_fn, env_id='t2d-v0', simulation_hours=72),
)

register(
    id='t2d_no_pump-v0',
    entry_point=partial(_make_env_fn, env_id='t2d_no_pump-v0', simulation_hours=72),
)
