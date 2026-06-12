"""Regression test: background noise must not trigger spurious exercise.

`disturb_action` adds noise to the heart-rate / exercise input. Before the fix
it did so unconditionally and clipped the result to [0, 1], rectifying the
zero-mean noise into a positive mean. Because exercise glucose sinks are
one-directional, this silently depressed the basal fasting steady state by
~30 mg/dL (130 -> ~99). With no exercise command, the input must stay zero, so
the fasting equilibrium is preserved.
"""
import dataclasses

import numpy as np

import glucosim
from glucosim import gym_env as gym


def _noise_minus_circadian(base):
    """All noise sources at defaults, but circadian disabled, so this isolates
    the (now-gated) heart-rate/exercise noise path from the circadian swing."""
    return dataclasses.replace(base, circadian_egp_amp_rel=0.0)


def test_basal_operation_does_not_trigger_exercise():
    env = gym.make("t1d-v0", simulation_minutes=24 * 60, sample_time=5,
                   patient_name="adult#001")
    u = env.unwrapped
    u.env_params = dataclasses.replace(
        u.env_params, noise_config=_noise_minus_circadian(u.env_params.noise_config)
    )
    type(u)._warmup_cache.clear()
    obs, _ = env.reset(seed=0)
    cgm0 = float(obs[0])
    env.close()
    # No-exercise fasting equilibrium for adult#001 is ~130 mg/dL. Before the fix
    # the spurious exercise pulled this to ~99; require it to stay near the
    # equilibrium now that the exercise input is gated.
    assert cgm0 >= 120.0, (
        f"basal fasting start {cgm0:.1f} mg/dL is depressed; background noise may "
        "be triggering spurious exercise again"
    )
