from ..gym_env.utils.registration import register

register(
    id='simglucose-v0',
    entry_point='envs.simglucose.sim.env:JaxSimEnv',
)
