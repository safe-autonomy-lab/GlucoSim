"""Reset caching must preserve seeded state, outputs and effective parameters."""
import jax
import numpy as np
import pytest

import glucosim
from glucosim import gym_env as gym


def snapshot(env, seed):
    reset_output = env.reset(seed=seed)
    before = env.unwrapped._jax_state
    outputs = [env.step(np.array([0, 0])) for _ in range(2)]
    return jax.tree.map(lambda x: np.asarray(x).copy() if not isinstance(x, str) else x,
                        (reset_output, before, outputs, env.unwrapped._jax_state,
                         env.unwrapped.key))


def assert_tree_equal(expected, actual):
    leaves_a, tree_a = jax.tree.flatten(expected)
    leaves_b, tree_b = jax.tree.flatten(actual)
    assert tree_a == tree_b
    for a, b in zip(leaves_a, leaves_b):
        np.testing.assert_array_equal(a, b)


@pytest.fixture(params=['t1d', 't2d', 't2d_no_pump'])
def fresh_env(request):
    env = gym.make(f'{request.param}-v0', patient_name='adolescent#001', sample_time=5)
    type(env.unwrapped)._warmup_cache.clear()
    yield env
    type(env.unwrapped)._warmup_cache.clear()
    env.close()


def test_cold_warm_and_mixed_seed_resets_match(fresh_env, monkeypatch):
    env = fresh_env
    hits = []
    get_cached = env.unwrapped._get_warm_state
    def observe_cache(key):
        result = get_cached(key)
        hits.append(result is not None)
        return result
    monkeypatch.setattr(env.unwrapped, '_get_warm_state', observe_cache)
    cold = snapshot(env, 42)
    assert_tree_equal(cold, snapshot(env, 42))
    type(env.unwrapped)._warmup_cache.clear()
    env.reset(seed=1)
    assert_tree_equal(cold, snapshot(env, 42))
    assert hits == [False, True, False, False]


def test_unseeded_reset_advances_and_replays(fresh_env):
    env = fresh_env
    env.reset(seed=42)
    starting_key = env.unwrapped.key
    first = snapshot(env, None)
    second = snapshot(env, None)
    assert not np.array_equal(first[-1], second[-1])
    type(env.unwrapped)._warmup_cache.clear()
    env.unwrapped.key = starting_key
    assert_tree_equal(first, snapshot(env, None))
    assert_tree_equal(second, snapshot(env, None))


def test_getter_exposes_effective_parameters(fresh_env):
    env = fresh_env
    env.reset(seed=42)
    assert env.unwrapped.get_patient_params() is env.unwrapped.env_params.patient_params
    assert env.unwrapped.patient_params is env.unwrapped.env_params.patient_params
