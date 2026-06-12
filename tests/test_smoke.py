"""Smoke tests: import, reset, step, seeding, rollout, termination."""
import numpy as np
import pytest

from conftest import EPISODE_STEPS, rollout


def test_import_registers_envs():
    import glucosim  # noqa: F401
    from glucosim.gym_env.utils.registration import registry

    for env_id in ("t1d-v0", "t2d-v0", "t2d_no_pump-v0"):
        assert env_id in registry


def test_reset_returns_valid_observation(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))
    assert isinstance(info, dict)
    # CGM (first feature) should start in a plausible fasting range.
    assert 40.0 <= obs[0] <= 300.0


def test_step_returns_six_tuple(env):
    env.reset(seed=0)
    out = env.step(np.array([0, 0]))
    assert len(out) == 6
    obs, reward, cost, terminated, truncated, info = out
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(cost, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_before_reset_raises():
    from glucosim import gym_env as gym

    fresh = gym.make("t1d-v0")
    with pytest.raises(RuntimeError):
        fresh.unwrapped.step(np.array([0, 0]))


def test_short_rollout_no_crash_and_finite(env):
    rec = rollout(env, seed=1, n_steps=50)
    assert np.all(np.isfinite(rec["obs"]))
    assert np.all(np.isfinite(rec["reward"]))
    assert np.all(np.isfinite(rec["cost"]))


def test_same_seed_reproduces_trajectory(env):
    a = rollout(env, seed=42, n_steps=20)
    b = rollout(env, seed=42, n_steps=20)
    np.testing.assert_allclose(a["obs"], b["obs"], rtol=0, atol=0)
    np.testing.assert_allclose(a["reward"], b["reward"], rtol=0, atol=0)
    np.testing.assert_allclose(a["cost"], b["cost"], rtol=0, atol=0)


def test_different_seeds_differ(env):
    a = rollout(env, seed=42, n_steps=20)
    b = rollout(env, seed=123, n_steps=20)
    assert not np.allclose(a["obs"], b["obs"])


def test_truncation_at_horizon(env):
    rec = rollout(env, seed=7, n_steps=EPISODE_STEPS + 10)
    n_steps_taken = len(rec["reward"])
    # Either the patient terminated early (allowed) or the episode truncated
    # exactly at the configured horizon.
    if rec["terminated"][-1]:
        assert n_steps_taken <= EPISODE_STEPS
    else:
        assert rec["truncated"][-1]
        assert n_steps_taken == EPISODE_STEPS
        assert rec["info"][-1]["termination_cause"] == 3  # timeout code
