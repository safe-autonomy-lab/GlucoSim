"""Preserve legacy entry points and the stochastic step's ordering contract."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from glucosim.simglucose.core.params import create_env_params
from glucosim.simglucose.physiology import glucose_dynamics, integration, vector_fields
from glucosim.simglucose.sim import physiology_step


@pytest.mark.parametrize('name', ['hovorka_t1d', 'hybrid_t2d'])
def test_legacy_vector_field_is_same_jitted_function(name):
    assert getattr(glucose_dynamics, name) is getattr(vector_fields, name)
    assert callable(getattr(glucose_dynamics, name).lower)


@pytest.mark.parametrize('name', ['t1d_rk4_step', 't2d_rk4_step'])
def test_legacy_step_is_same_jitted_function(name):
    assert getattr(glucose_dynamics, name) is getattr(physiology_step, name)
    assert callable(getattr(glucose_dynamics, name).lower)


def test_legacy_exercise_helper_import():
    assert glucose_dynamics._exercise_e2_step is integration._exercise_e2_step


@pytest.mark.parametrize('kind', ['t1d', 't2d', 't2d_no_pump'])
def test_step_orders_disturbance_integration_circadian_and_noise(monkeypatch, kind):
    env = create_env_params('adolescent#001', diabetes_type=kind)
    p, cfg = env.patient_params, env.noise_config
    model = 't1d' if kind == 't1d' else 't2d'
    state = jnp.zeros(18).at[8].set(0.05)
    action = jnp.array([2., 0.1, 0.4])
    key = jax.random.PRNGKey(42)
    dt, minute, stomach_mg, food_g = 0.25, 37., 1200., 12.
    ou = jnp.array(0.3)
    calls = []

    def disturb(a, params, k, config):
        calls.append('disturb')
        assert params is p and config is cfg
        np.testing.assert_array_equal(a, action)
        np.testing.assert_array_equal(k, key)
        return a + 0.01, k + 1

    def factors(t, config):
        calls.append('factors')
        assert float(t) == minute and config is cfg
        return {'circadian': jnp.array(1.25)}

    def integrate(x, step_dt, a, params, stomach, food):
        calls.append('integrate')
        assert params is p and step_dt == dt
        assert stomach == stomach_mg and food == food_g
        np.testing.assert_array_equal(x, state)
        np.testing.assert_array_equal(a, action + 0.01)
        return x.at[3].set(100.).at[8].set(0.25)

    def noise(x, step_dt, k, config, params, prior_ou):
        calls.append('noise')
        assert config is cfg and params is p and float(step_dt) == dt
        np.testing.assert_array_equal(k, key + 1)
        np.testing.assert_array_equal(prior_ou, ou)
        # T2D suppression uses the integrated x3, not the incoming x3.
        flow = p.kp1 if kind == 't1d' else (p.EGP_0 * 180. / p.BW) * 0.75
        assert float(x[3]) == pytest.approx(100. + dt * 0.25 * flow, abs=1e-5)
        return x, k + 1, prior_ou + 0.5

    monkeypatch.setattr(physiology_step, 'disturb_action', disturb)
    monkeypatch.setattr(physiology_step, 'dynamic_factors_for_step', factors)
    monkeypatch.setattr(physiology_step, 'integrate_' + model, integrate)
    monkeypatch.setattr(physiology_step, 'add_process_noise_structured', noise)
    step = getattr(physiology_step, model + '_rk4_step')
    # Exercise Python orchestration without reusing previously compiled JITs.
    _, next_key, next_ou = step.__wrapped__(
        state, dt, action, p, stomach_mg, food_g, minute, key, cfg, ou,
    )
    assert calls == ['disturb', 'factors', 'integrate', 'noise']
    np.testing.assert_array_equal(next_key, key + 2)
    np.testing.assert_array_equal(next_ou, ou + 0.5)
