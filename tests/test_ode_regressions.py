"""Regression checks for the reviewed ODE bookkeeping and initialization bugs."""
import dataclasses
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples import run_scenarios as runner
from glucosim.simglucose.core.params import (
    _mgdl_to_mM, _mM_to_mgdl, adapt_params_for_t2d, create_env_params, create_patient_params,
)
from glucosim.simglucose.core.types import Action, PatientType
from glucosim.simglucose.physiology.glucose_dynamics import (
    hovorka_t1d, hybrid_t2d, t1d_rk4_step,
)
from glucosim.simglucose.physiology.initialization import tune_initial_state
from glucosim.simglucose.sim.patient_transition import patient_step
from glucosim.simglucose.sim.reset import _build_base_state


@pytest.fixture(scope="module", params=["t1d", "t2d", "t2d_no_pump"])
def patient(request):
    env, state = tune_initial_state(create_env_params(diabetes_type=request.param))
    env = dataclasses.replace(
        env, noise_config=dataclasses.replace(env.noise_config, enable=False)
    )
    return env, state


def test_initial_glucose_and_insulin_are_stationary(patient):
    env, state = patient
    p = env.patient_params
    ode = hovorka_t1d if p.diabetes_type == PatientType.t1d else hybrid_t2d
    residual = np.asarray(ode(state, jnp.zeros(3), p, 0.0, 0.0))
    # Exercise placeholders/projections are outside this equilibrium contract.
    np.testing.assert_allclose(residual[3:13], 0.0, atol=2e-5)
    if p.diabetes_type != PatientType.t1d:
        np.testing.assert_allclose(residual[16:18], 0.0, atol=2e-5)


def test_meal_history_includes_final_bite_and_survives_digestion(patient):
    env, x = patient
    key = jax.random.PRNGKey(42)
    state = _build_base_state(env, x, jnp.zeros((1, 3, 2)), key)
    for minute in range(10):
        state = patient_step(
            state, Action(meal=80.0 if minute == 0 else 0.0, bolus=0.0, exercise=0.0),
            env.patient_params, key, env.noise_config, 10.0,
        )
        assert float(state["last_foodtaken"]) == min((minute + 1) * 10.0, 80.0)
        assert int(state["is_eating"]) == int(minute < 8)
    # A new meal captures the remaining stomach contents and starts a fresh count.
    remaining_stomach = float(state["patient_state"][:2].sum())
    state = patient_step(
        state, Action(meal=5.0, bolus=0.0, exercise=0.0),
        env.patient_params, key, env.noise_config, 10.0,
    )
    assert float(state["last_foodtaken"]) == 5.0
    assert float(state["last_Qsto"]) == pytest.approx(remaining_stomach)


@pytest.mark.parametrize("dt", [1.0, 0.25])
def test_runner_meal_history_uses_grams_and_is_retained(patient, monkeypatch, dt):
    env, x = patient
    history = []

    def capture(**kwargs):
        history.append(float(kwargs["last_foodtaken"]))
        return kwargs["x"], kwargs["key"], kwargs["ou_state_dL"]

    monkeypatch.setattr(runner, "t1d_rk4_step", capture)
    monkeypatch.setattr(runner, "t2d_rk4_step", capture)
    runner.simulate(
        21.0, dt, x,
        lambda t: jnp.array([5.0 if t < 15.0 else (2.0 if t >= 20.0 else 0.0), 0., 0.]),
        env.patient_params, env.noise_config, jax.random.PRNGKey(42),
    )
    assert history[int(15 / dt) - 1] == 75.0
    assert history[int(20 / dt) - 1] == 75.0
    assert history[-1] == 2.0


@pytest.mark.parametrize("scenario", ["basal", "exercise"])
def test_runner_does_not_add_basal_twice(patient, monkeypatch, tmp_path, scenario):
    env, x = patient
    actions = []

    def capture(**kwargs):
        actions.extend(np.asarray(kwargs["action_fn"](t)) for t in (0., 120.))
        return np.array([0.0]), np.asarray([x])

    monkeypatch.setattr(runner, "create_env_params", lambda **kwargs: env)
    monkeypatch.setattr(runner, "tune_initial_state", lambda _: (env, x))
    monkeypatch.setattr(runner, "simulate", capture)
    monkeypatch.setattr(runner, "plot_states", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["run_scenarios.py", "--scenario", scenario,
                                    "--output_dir", str(tmp_path)])
    runner.main()
    assert all(a[1] == 0.0 for a in actions)
    if scenario == "exercise":
        assert actions[1][2] == 0.5


@pytest.mark.parametrize("duration_state", [0.5, 1.0, 60.0, 100.0])
@pytest.mark.parametrize("effect_state", [0.0, 5.0])
def test_t1d_exercise_step_matches_vector_field(duration_state, effect_state):
    env, x = tune_initial_state(create_env_params(diabetes_type="t1d"))
    cfg = dataclasses.replace(env.noise_config, enable=False)
    x = x.at[13].set(30.0).at[14].set(duration_state).at[15].set(effect_state)
    action = jnp.array([0., 0., .5])
    dt = 1e-4
    following, _, _ = t1d_rk4_step(
        x, dt, action, env.patient_params, 0., 0., 0.,
        jax.random.PRNGKey(42), cfg, jnp.array(0.),
    )
    expected = float(hovorka_t1d(x, action, env.patient_params, 0., 0.)[15])
    assert float((following[15] - x[15]) / dt) == pytest.approx(expected, rel=5e-3)


@pytest.mark.parametrize("kind", ["t2d", "t2d_no_pump"])
def test_scaling_applied_once(kind):
    base = create_patient_params("adolescent#001", diabetes_type=kind)
    scaled = create_patient_params(
        "adolescent#001", diabetes_type=kind,
        carb_absorption_scale=2., eat_rate_scale=2., insulin_sensitivity_scale=2.,
    )
    for field in ("kmax", "kabs", "eat_rate", "Vmx"):
        assert getattr(scaled, field) == pytest.approx(2 * getattr(base, field))


def test_no_pump_sensitivities_use_final_resistance():
    pump = create_patient_params("adolescent#001", diabetes_type="t2d")
    no_pump = create_patient_params("adolescent#001", diabetes_type="t2d_no_pump")
    for field in ("S_I1", "S_I2", "S_I3"):
        assert getattr(no_pump, field) * no_pump.insulin_resistance_factor == pytest.approx(
            getattr(pump, field) * pump.insulin_resistance_factor
        )


def test_pump_conversion_uses_declared_resistance():
    base = create_patient_params("adolescent#001", diabetes_type="t1d")
    pump = adapt_params_for_t2d(dataclasses.replace(base, insulin_resistance_factor=4.0))
    for field in ("S_I1", "S_I2", "S_I3"):
        assert getattr(pump, field) == pytest.approx(getattr(base, field) / 2.5)


def test_glucose_conversion_round_trip():
    for glucose in (70., 149.02, 250.):
        assert _mM_to_mgdl(_mgdl_to_mM(glucose)) == pytest.approx(glucose, abs=1e-10)
