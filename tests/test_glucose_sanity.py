"""Physiological sanity checks for the blood glucose simulator.

These tests pin down directional behavior (meals raise glucose, insulin
lowers it), threshold behavior of the safety cost, and numerical health
over a full episode. They intentionally avoid asserting exact values so
they do not over-fit to simulator constants.
"""
import numpy as np
import pytest

from conftest import EPISODE_STEPS, rollout

MAX_ACTION_LEVEL = 4  # action_dim=5 -> levels 0..4, 4 = max bolus / max meal


def test_basal_only_glucose_plausible(env):
    rec = rollout(env, seed=3, n_steps=60)
    cgm = rec["obs"][:, 0]
    assert np.all(np.isfinite(cgm))
    assert np.all(cgm > 10.0)
    assert np.all(cgm < 600.0)


def test_meal_raises_glucose(env):
    """A large meal at t=0 should raise CGM relative to a no-op rollout.

    Both rollouts share the seed, so the scenario meals and noise draws are
    identical; only the agent meal differs.
    """
    seed = 11
    horizon = 24  # 2 hours: covers the carb absorption peak
    base = rollout(env, seed=seed, n_steps=horizon)
    meal = rollout(env, seed=seed, actions=np.array([0, MAX_ACTION_LEVEL]), n_steps=horizon)
    assert bool(meal["info"][1]["meal_accepted"]), "max meal at t=0 should be accepted"
    # Compare mean CGM over the absorption window (30min - 2h after the meal).
    base_cgm = base["obs"][6:, 0].mean()
    meal_cgm = meal["obs"][6:, 0].mean()
    assert meal_cgm > base_cgm + 5.0, (
        f"80g meal should visibly raise CGM: meal={meal_cgm:.1f}, base={base_cgm:.1f}"
    )


def test_bolus_lowers_glucose(env):
    """A large bolus at t=0 should lower CGM relative to a no-op rollout."""
    seed = 11
    horizon = 36  # 3 hours: covers insulin action onset and peak
    base = rollout(env, seed=seed, n_steps=horizon)
    bolus = rollout(env, seed=seed, actions=np.array([MAX_ACTION_LEVEL, 0]), n_steps=horizon)
    assert bool(bolus["info"][1]["bolus_accepted"]), "max bolus at t=0 should be accepted"
    base_cgm = base["obs"][12:, 0].mean()
    bolus_cgm = bolus["obs"][12:, 0].mean()
    assert bolus_cgm < base_cgm - 5.0, (
        f"10U bolus should visibly lower CGM: bolus={bolus_cgm:.1f}, base={base_cgm:.1f}"
    )


def test_bolus_increases_iob(env):
    seed = 11
    base = rollout(env, seed=seed, n_steps=3)
    bolus = rollout(env, seed=seed, actions=np.array([MAX_ACTION_LEVEL, 0]), n_steps=3)
    # Observation feature 1 is bolus insulin-on-board (U).
    assert bolus["obs"][1, 1] > base["obs"][1, 1] + 0.5


def test_full_episode_random_actions_numerically_healthy(env):
    """A full seeded 24h episode with random actions must stay finite."""
    rng = np.random.default_rng(0)
    actions = rng.integers(0, MAX_ACTION_LEVEL + 1, size=(EPISODE_STEPS, 2))
    rec = rollout(env, seed=5, actions=lambda i: actions[i], n_steps=EPISODE_STEPS)
    assert np.all(np.isfinite(rec["obs"]))
    assert np.all(np.isfinite(rec["reward"]))
    assert np.all(np.isfinite(rec["cost"]))
    assert np.all(rec["cost"] >= 0.0)
    cgm = rec["obs"][:, 0]
    assert np.all(cgm > 0.0) and np.all(cgm < 700.0)


class TestDangerCostThresholds:
    """Unit tests for the graded danger cost at clinical thresholds."""

    @staticmethod
    def _cost(level, n=24, dt=5.0):
        import jax.numpy as jnp

        from glucosim.simglucose.rl.reward_and_cost_functions import (
            calculate_graded_danger_cost_raw,
        )

        trace = jnp.full((n,), float(level), dtype=jnp.float32)
        return float(calculate_graded_danger_cost_raw(trace, dt=dt))

    def test_in_range_glucose_has_zero_cost(self):
        for level in (80.0, 100.0, 140.0, 180.0):
            assert self._cost(level) == pytest.approx(0.0, abs=1e-5)

    def test_hypoglycemia_cost_below_70(self):
        assert self._cost(69.0) > 0.0
        assert self._cost(60.0) > self._cost(69.0)
        # Severe hypo (<54) ramps up much faster.
        assert self._cost(45.0) > self._cost(60.0)

    def test_hyperglycemia_cost_above_180(self):
        assert self._cost(181.0) > 0.0
        assert self._cost(250.0) > self._cost(200.0)
        # Severe hyper (>250) adds a quadratic term.
        assert self._cost(350.0) > self._cost(250.0)

    def test_hypo_costs_more_than_symmetric_hyper(self):
        # Hypoglycemia is clinically more dangerous; the weights reflect it.
        assert self._cost(50.0) > self._cost(250.0)

    def test_cost_is_clamped(self):
        assert self._cost(1000.0) <= 50.0


def test_zero_meal_basal_only_consistency(env):
    """Two basal-only rollouts with the same seed are step-for-step identical
    and keep IOB/COB observations non-negative."""
    a = rollout(env, seed=21, n_steps=30)
    b = rollout(env, seed=21, n_steps=30)
    np.testing.assert_array_equal(a["obs"], b["obs"])
    assert np.all(a["obs"][:, 1] >= -0.1)  # IOB
    assert np.all(a["obs"][:, 2] >= -0.1)  # COB
