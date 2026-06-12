"""Shared pytest fixtures for GlucoSim tests.

JAX is forced onto CPU before it is imported so the suite runs reliably on
CI machines without a GPU and avoids slow cuDNN autotuning locally.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import glucosim  # noqa: F401  (registers t1d-v0 / t2d-v0 / t2d_no_pump-v0)
from glucosim import gym_env as gym

# One day at a 5-minute controller interval = 288 steps.
SIMULATION_MINUTES = 24 * 60
SAMPLE_TIME = 5
PATIENT_NAME = "adolescent#001"
EPISODE_STEPS = SIMULATION_MINUTES // SAMPLE_TIME


@pytest.fixture(scope="session")
def env():
    """A single shared t1d env; reset(seed=...) gives independent episodes.

    Session-scoped because the first reset triggers an expensive JIT compile
    plus a 72-hour basal-only warmup (cached per patient configuration).
    """
    env = gym.make(
        "t1d-v0",
        simulation_minutes=SIMULATION_MINUTES,
        sample_time=SAMPLE_TIME,
        patient_name=PATIENT_NAME,
    )
    yield env
    env.close()


def rollout(env, seed, actions=None, n_steps=20):
    """Run a seeded rollout and return per-step records as a dict of arrays.

    ``actions`` may be None (no-op actions), a fixed action applied at step 0
    only, or a callable ``step_idx -> action``.
    """
    obs, info = env.reset(seed=seed)
    records = {
        "obs": [obs],
        "reward": [],
        "cost": [],
        "terminated": [],
        "truncated": [],
        "info": [info],
    }
    noop = np.array([0, 0])
    for i in range(n_steps):
        if actions is None:
            action = noop
        elif callable(actions):
            action = actions(i)
        else:
            action = actions if i == 0 else noop
        obs, reward, cost, terminated, truncated, info = env.step(action)
        records["obs"].append(obs)
        records["reward"].append(reward)
        records["cost"].append(cost)
        records["terminated"].append(terminated)
        records["truncated"].append(truncated)
        records["info"].append(info)
        if terminated or truncated:
            break
    records["obs"] = np.array(records["obs"])
    records["reward"] = np.array(records["reward"])
    records["cost"] = np.array(records["cost"])
    return records
