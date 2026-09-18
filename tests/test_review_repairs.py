"""Behavioral checks for the post-review balance, timing and unit repairs."""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples import run_scenarios as runner
from glucosim.simglucose.core.params import create_env_params, create_patient_params
from glucosim.simglucose.core.types import Action
from glucosim.simglucose.physiology.glucose_dynamics import (
    hovorka_t1d, hybrid_t2d, t1d_rk4_step, t2d_rk4_step,
)
from glucosim.simglucose.physiology.initialization import tune_initial_state
from glucosim.simglucose.sim.patient_transition import patient_step
from glucosim.simglucose.sim.realism import _ou_exchange_eps
from glucosim.simglucose.sim.reset import _build_base_state


@pytest.fixture(scope='module', params=['t1d', 't2d', 't2d_no_pump'])
def model(request):
    env, x = tune_initial_state(create_env_params(diabetes_type=request.param))
    cfg = dataclasses.replace(env.noise_config, enable=False)
    ode, step = (hovorka_t1d, t1d_rk4_step) if request.param == 't1d' else (hybrid_t2d, t2d_rk4_step)
    return env, x, cfg, ode, step


@pytest.mark.parametrize('cohort', ['adolescent', 'adult', 'child'])
def test_pump_basal_attains_target_insulin(cohort):
    for number in range(1, 11):
        env, x = tune_initial_state(create_env_params(
            patient_name=f'{cohort}#{number:03d}', diabetes_type='t2d'))
        p = env.patient_params
        assert p.basal > 0
        assert float(x[5]) / p.Vi == pytest.approx(p.Ib, rel=2e-6)
        dx = np.asarray(hybrid_t2d(x, jnp.zeros(3), p, 0., 0.))
        np.testing.assert_allclose(dx[3:18], 0., atol=2e-5)


def test_exercise_starts_at_resting_equilibrium(model):
    env, x, cfg, ode, step = model
    dx = ode(x, jnp.zeros(3), env.patient_params, 0., 0.)
    np.testing.assert_allclose(dx[13:16], 0., atol=1e-7)
    y, _, _ = step(x, 1., jnp.zeros(3), env.patient_params, 0., 0., 0.,
                   jax.random.PRNGKey(42), cfg, jnp.array(0.))
    np.testing.assert_allclose(y[13:16], x[13:16], atol=1e-7)


@pytest.mark.parametrize('duration', [.01, .5, 100.])
def test_exercise_duration_follows_ode_without_projection(model, duration):
    env, x, cfg, ode, step = model
    p = env.patient_params
    x = x.at[14].set(duration)
    dt = .1
    y, _, _ = step(x, dt, jnp.zeros(3), p, 0., 0., 0.,
                   jax.random.PRNGKey(42), cfg, jnp.array(0.))
    expected = p.c2 + (duration - p.c2) * np.exp(-dt / p.tau_ex)
    assert float(y[14]) == pytest.approx(expected, rel=2e-6, abs=1e-7)


def test_fast_e2_decay_is_stable_at_one_minute(model):
    env, x, cfg, ode, step = model
    p = env.patient_params
    # E1 and T_E are at their constant-input equilibrium; the E2 solution is exact.
    hr = .5
    E1 = hr * (220. - p.age - p.HR0)
    z = (E1 / (p.alpha_HR * p.HR0)) ** p.n_power
    f = z / (1. + z)
    TE = p.c1 * f + p.c2
    x = x.at[13].set(E1).at[14].set(TE).at[15].set(1.)
    a = f / p.tau_in + 1. / max(TE, .001)
    b = f * max(TE, .001) / (p.c1 + p.c2)
    y, _, _ = step(x, 1., jnp.array([0., 0., hr]), p, 0., 0., 0.,
                   jax.random.PRNGKey(42), cfg, jnp.array(0.))
    assert np.isfinite(np.asarray(y)).all()
    assert float(y[15]) == pytest.approx(np.exp(-a) + b / a * (-np.expm1(-a)), rel=2e-5)
    assert float(y[3]) > .95 * float(x[3])  # no stiff intermediate E2 sink explosion


def test_internal_exercise_session_survives_multiple_minutes(model):
    env, x, cfg, ode, step = model
    key = jax.random.PRNGKey(42)
    state = _build_base_state(env, x, jnp.zeros((1, 3, 2)), key)
    state['planned_exercise_min'] = jnp.array(3.)
    state['exercise_intensity'] = jnp.array(.5)
    delivered = []
    for minute in range(5):
        intensity = .5 if float(state['planned_exercise_min']) > 0 else 0.
        delivered.append(intensity)
        state = patient_step(state, Action(meal=0., bolus=0., exercise=intensity),
                             env.patient_params, key, cfg, 10.)
        assert float(state['planned_exercise_min']) == max(2. - minute, 0.)
    assert delivered == [.5, .5, .5, 0., 0.]


def test_runner_remainder_uses_actual_time_and_mass(model, monkeypatch):
    env, x, cfg, ode, step = model
    calls = []
    def capture(**kwargs):
        calls.append((kwargs['t_min'], kwargs['dt'], kwargs['last_foodtaken']))
        return kwargs['x'], kwargs['key'], kwargs['ou_state_dL']
    monkeypatch.setattr(runner, 't1d_rk4_step', capture)
    monkeypatch.setattr(runner, 't2d_rk4_step', capture)
    times, states = runner.simulate(1., .3, x, lambda t: jnp.array([2., 0., 0.]),
                                   env.patient_params, cfg, jax.random.PRNGKey(42), t0_min=10.)
    np.testing.assert_allclose(times, [0., .3, .6, .9, 1.])
    np.testing.assert_allclose([c[0] for c in calls], times[:-1] + 10.)
    np.testing.assert_allclose([c[1] for c in calls], [.3, .3, .3, .1])
    assert calls[-1][2] == pytest.approx(2.)


@pytest.mark.parametrize('name,value', [('tau_in', 0.), ('tau_ex', -1.), ('c2', 0.),
                                       ('c1', -1.), ('tau_HR', float('nan'))])
def test_invalid_exercise_parameters_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        create_patient_params('adolescent#001', diabetes_type='t1d', **{name: value})


def test_ou_exchange_telescopes_to_offset_change():
    key = jax.random.PRNGKey(42)
    state = jnp.array(0.)
    total = 0.
    for dt in [.1, .3, .6, 1., .25]:
        delta, state, key = _ou_exchange_eps(key, jnp.array(dt), .1, 2., 1.7, state)
        total += float(delta)
    assert total == pytest.approx(float(state) * 1.7, abs=1e-6)
    delta, unchanged, _ = _ou_exchange_eps(key, jnp.array(0.), .1, 2., 1.7, state)
    assert float(delta) == 0.
    assert float(unchanged) == float(state)
