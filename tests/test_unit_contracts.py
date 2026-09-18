"""Numerical dose/concentration conversions and physical-pool balances.

These check arithmetic contracts, not the clinical calibration of the models.
"""
import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from glucosim.simglucose.core.params import _mgdl_to_mM, _mM_to_mgdl, create_env_params
from glucosim.simglucose.physiology.glucose_dynamics import hovorka_t1d, hybrid_t2d
from glucosim.simglucose.physiology.initialization import tune_initial_state
from glucosim.simglucose.physiology.steady_state_solvers import iir_exogenous_pmolkgmin


@pytest.fixture(scope="module", params=["t1d", "t2d", "t2d_no_pump"])
def model(request):
    env, state = tune_initial_state(create_env_params(diabetes_type=request.param))
    ode = hovorka_t1d if request.param == "t1d" else hybrid_t2d
    return request.param, env.patient_params, state, ode


def test_glucose_conversion_has_correct_absolute_scale():
    assert _mgdl_to_mM(180.0) == pytest.approx(10.0, rel=0, abs=1e-12)
    assert _mM_to_mgdl(10.0) == pytest.approx(180.0, rel=0, abs=1e-12)


@pytest.mark.parametrize("weight", [40.0, 80.0])
@pytest.mark.parametrize("pump", [True, False])
def test_insulin_dose_rate_conversion(model, weight, pump):
    _, p, _, _ = model
    p = dataclasses.replace(p, BW=weight, basal=0.6, use_pump=pump)
    # Additional 3 U delivered over 15 minutes; basal is U/hour.
    rate = iir_exogenous_pmolkgmin(p, 3.0 / 15.0)
    expected_U = 3.0 + (0.6 * 15.0 / 60.0 if pump else 0.0)
    assert rate * weight * 15.0 / 6000.0 == pytest.approx(expected_U)


def test_gut_mass_balance_and_appearance_units(model):
    _, p, state, ode = model
    state = state.at[:3].set(jnp.array([12000., 4000., 5000.]))
    dx = np.asarray(ode(state, jnp.array([2.5, 0., 0.]), p, 0., 21.))
    # All gut pools are whole-patient mg, not mg/kg.
    assert dx[:3].sum() == pytest.approx(2500.0 - p.kabs * 5000.0, abs=1e-3)
    empty_intestine = state.at[2].set(0.)
    no_appearance = np.asarray(ode(empty_intestine, jnp.array([2.5, 0., 0.]), p, 0., 21.))
    assert dx[3] - no_appearance[3] == pytest.approx(p.f * p.kabs * 5000.0 / p.BW, abs=1e-5)


def test_total_insulin_pool_balance(model):
    kind, p, state, ode = model
    state = state.at[5].set(8.).at[9].set(4.).at[10].set(20.).at[11].set(30.)
    state = state.at[16].set(3.)  # mU/min secretion state, for T2D only
    dx = np.asarray(ode(state, jnp.array([0., 0.2, 0.]), p, 0., 0.))
    secretion = 0. if kind == "t1d" else 6. * (p.Sb_per_kg + p.beta_cell_function * 3. / p.BW)
    inflow = iir_exogenous_pmolkgmin(p, 0.2)
    # Plasma/liver/SC exchanges cancel; each pool is pmol/kg.
    expected = inflow + secretion - p.m4 * 8. - p.m30 * 4.
    assert dx[[5, 9, 10, 11]].sum() == pytest.approx(expected, abs=1e-5)


def test_glucose_pool_balance_without_exercise(model):
    kind, p, state, ode = model
    state = state.at[3].set(350.).at[4].set(200.).at[2].set(5000.)
    state = state.at[8].set(.3 if kind != "t1d" else state[8])
    dx = np.asarray(ode(state, jnp.zeros(3), p, 0., 5.))
    insulin_pmol_L = float(state[5]) / p.Vi
    if kind == "t1d":
        production = max(p.kp1 - p.kp2 * 350. - p.kp3 * insulin_pmol_L, 0.)
        brain = p.Fsnc
    else:
        production = p.EGP_0 * 180. / p.BW * .7
        brain = p.F_cns0 * 180. / p.BW
    appearance = p.f * p.kabs * 5000. / p.BW
    renal = p.ke1 * max(350. - p.ke2, 0.)
    utilization = (p.Vm0 + p.Vmx * insulin_pmol_L) * 200. / (p.Km0 + 200.)
    assert dx[3:5].sum() == pytest.approx(production + appearance - brain - renal - utilization, abs=1e-5)
