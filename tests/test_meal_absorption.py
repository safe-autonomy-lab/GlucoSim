"""Regression test: meal absorption must not depend on the controller interval.

`sample_time` is a controller/observation choice. With process noise disabled,
an identical meal must produce an identical carbs-on-board (COB) trajectory in
wall-clock time regardless of `sample_time`. Before the fix to `sim/step.py`
(which divided `eat_rate` by `sample_time`), an 80 g meal left 3.6 g on board at
2 h for `sample_time=1` but 16.2 g for `sample_time=10`.
"""
import dataclasses

import numpy as np
import pytest

import glucosim
from glucosim import gym_env as gym

MAX_MEAL_LEVEL = 4  # action_dim=5 -> level 4 = max meal (80 g for adult#001)


def _cob_at_minutes(sample_time, minutes=120, patient="adult#001"):
    """Drive an 80 g meal at t=0 (no bolus, noise OFF) and return COB sampled
    at every `sample_time` minutes as (time_min, cob_g) arrays."""
    env = gym.make("t1d-v0", simulation_minutes=24 * 60,
                   sample_time=sample_time, patient_name=patient)
    u = env.unwrapped
    nc = dataclasses.replace(u.env_params.noise_config, enable=False)
    u.env_params = dataclasses.replace(u.env_params, noise_config=nc)
    obs, _ = env.reset(seed=0)
    times, cob = [0], [float(obs[2])]
    n_steps = minutes // sample_time
    for i in range(n_steps):
        action = np.array([0, MAX_MEAL_LEVEL]) if i == 0 else np.array([0, 0])
        obs, *_ = env.step(action)
        times.append((i + 1) * sample_time)
        cob.append(float(obs[2]))
    env.close()
    return np.array(times), np.array(cob)


def test_meal_absorption_independent_of_sample_time():
    # Reference at the finest interval, compared at common 10-min grid points.
    t1, cob1 = _cob_at_minutes(1)
    t5, cob5 = _cob_at_minutes(5)
    t10, cob10 = _cob_at_minutes(10)

    ref = dict(zip(t1.tolist(), cob1.tolist()))
    for t, c in zip(t5, cob5):
        if t in ref:
            assert c == pytest.approx(ref[t], abs=1.0), (
                f"sample_time=5 COB at {t} min = {c:.2f} g but sample_time=1 = {ref[t]:.2f} g"
            )
    for t, c in zip(t10, cob10):
        if t in ref:
            assert c == pytest.approx(ref[t], abs=1.0), (
                f"sample_time=10 COB at {t} min = {c:.2f} g but sample_time=1 = {ref[t]:.2f} g"
            )
