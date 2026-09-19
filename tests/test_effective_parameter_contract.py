"""Independent arithmetic and initialization contracts for effective inputs.

These tests exercise public construction without trajectories or reset warmup.
The synthetic source retains a bundled row's kinetics and supplies simple units.
"""
import dataclasses
from fractions import Fraction
from pathlib import Path

import jax.numpy as jnp
import pytest

from glucosim.simglucose.core import params, patient_loader
from glucosim.simglucose.physiology import calibration, initialization, vector_fields


KINDS = ('t1d', 't2d', 't2d_no_pump')
EFFECTIVE = dict(BW=100.0, Vg=2.0, Vi=0.05, Gpb=180.0,
                 Fsnc=1.8, EGPb=2.0)


@pytest.mark.parametrize('kind', KINDS[1:])
def test_real_scalar_inputs_are_normalized_before_construction(synthetic_source, kind):
    patient = create(kind, BW=Fraction(100), insulin_resistance_factor=Fraction(5))
    assert type(patient.BW) is float
    assert type(patient.insulin_resistance_factor) is float
    assert patient.BW == 100.0
    assert patient.insulin_resistance_factor == 5.0
    assert (patient.S_I1, patient.S_I2, patient.S_I3) == pytest.approx(
        (0.0002, 0.002, 0.001))


@pytest.fixture(scope='module')
def source_row():
    path = Path(patient_loader.__file__).parent.parent / 'params/vpatient_params.csv'
    row = patient_loader.load_patient_parameters_from_csv(str(path))['adolescent#001']
    return dict(row, BW=80.0, Vg=2.0, Vi=0.05, Fsnc=1.8,
                S_I1=0.001, S_I2=0.01, S_I3=0.005)


@pytest.fixture
def synthetic_source(monkeypatch, source_row):
    monkeypatch.setattr(patient_loader, 'load_patient_parameters_from_csv',
                        lambda path: {'adolescent#001': source_row.copy()})
    return source_row


@pytest.fixture(scope='module')
def env_template():
    return params.create_env_params('adolescent#001', diabetes_type='t1d')


def create(kind, **overrides):
    return params.create_patient_params('adolescent#001', diabetes_type=kind, **overrides)


@pytest.mark.parametrize('kind', KINDS)
def test_effective_weight_controls_weight_dependent_construction(synthetic_source, kind):
    patient = create(kind, BW=100.0, autobalance_enabled=False)
    assert patient.BW == 100.0
    if kind == 't1d':
        assert patient.basal == pytest.approx(1.1)
    else:
        # 1.8 mg/kg/min * 100 kg / 180 mg/mmol = 1 mmol/min.
        assert patient.F_cns0 == pytest.approx(1.0)
        assert patient.F_cns0 * 180.0 / patient.BW == pytest.approx(1.8)
        assert patient.V_G == pytest.approx(20.0)
        assert patient.V_I == pytest.approx(5.0)
        # Synchronizing the original reporting aliases is deliberately deferred.
        assert patient.V_G_L == pytest.approx(16.0)
        assert patient.V_I_L == pytest.approx(4.0)


@pytest.mark.parametrize('kind,resistance,weight,pump,beta', [
    ('t1d', 1.0, 80.0, True, 0.0),
    ('t2d', 2.5, 92.0, True, 0.25),
    ('t2d_no_pump', 2.8, 92.0, False, 0.3),
])
def test_no_override_presets_keep_source_scaling(synthetic_source, kind, resistance,
                                                weight, pump, beta):
    patient = create(kind)
    assert patient.BW == pytest.approx(weight)
    assert patient.insulin_resistance_factor == resistance
    assert patient.use_pump is pump
    assert patient.beta_cell_function == beta
    if kind != 't1d':
        assert (patient.S_I1, patient.S_I2, patient.S_I3) == pytest.approx(
            (0.001 / resistance, 0.01 / resistance, 0.005 / resistance))
    if kind == 't2d_no_pump':
        assert patient.basal == 0.0
        assert patient.K_deriv == 30.0


@pytest.mark.parametrize('kind', KINDS[1:])
def test_resistance_five_uses_original_gains_once(synthetic_source, kind):
    patient = create(kind, insulin_resistance_factor=5.0)
    assert patient.insulin_resistance_factor == 5.0
    assert (patient.S_I1, patient.S_I2, patient.S_I3) == pytest.approx(
        (0.0002, 0.002, 0.001))
    state = jnp.zeros(18).at[3].set(180.0).at[4].set(100.0)
    state = state.at[5].set(60.0 * patient.Vi).at[14].set(patient.c2)
    derivative = vector_fields.hybrid_t2d(state, jnp.zeros(3), patient, 0.0, 0.0)
    # Ip/Vi/6 = 10 mU/L; zero effect states leave gain * concentration.
    assert list(derivative[6:9]) == pytest.approx((0.002, 0.02, 0.01))


@pytest.mark.parametrize('kind', KINDS[1:])
def test_glucose_mass_and_volume_determine_secretion_threshold(synthetic_source, kind):
    patient = create(kind, Vg=2.0, Gpb=180.0)
    assert patient.h == pytest.approx(5.0)  # 180 mg/kg / 2 dL/kg / 18.


@pytest.mark.parametrize('kind', KINDS)
def test_all_covered_inputs_and_dependencies_survive_direct_initialization(
        synthetic_source, env_template, kind):
    requested = dict(EFFECTIVE, insulin_resistance_factor=1.0 if kind == 't1d' else 5.0)
    patient = create(kind, **requested)
    tuned_env, state = initialization.tune_initial_state(
        dataclasses.replace(env_template, patient_params=patient))
    tuned = tuned_env.patient_params
    for name, value in requested.items():
        assert getattr(patient, name) == value
        assert getattr(tuned, name) == value
    assert float(state[3]) == pytest.approx(180.0)
    if kind != 't1d':
        for p in (patient, tuned):
            assert p.h == pytest.approx(5.0)
            assert p.F_cns0 == pytest.approx(1.0)
            # Portal secretion must deliver the fixed systemic target 0.4 U/h.
            assert p.Sb_per_kg * 100.0 * (1.0 - p.HEb) * 60.0 / 1000.0 == pytest.approx(0.4)
            assert (p.S_I1, p.S_I2, p.S_I3) == pytest.approx((0.0002, 0.002, 0.001))
        # Reset owns EGP_0, but does not retune these conversion/calibration values.
        for name in ('Sb_per_kg', 'Vm0', 'Vmx', 'V_G', 'V_I'):
            assert getattr(tuned, name) == getattr(patient, name)
    else:
        # Reset owns kp1/Vm0 and restores the requested production target.
        insulin = float(state[5]) / tuned.Vi
        production = tuned.kp1 - tuned.kp2 * float(state[3]) - tuned.kp3 * insulin
        assert production == pytest.approx(2.0)
        for name in ('basal', 'Vmx', 'S_I1', 'S_I2', 'S_I3'):
            assert getattr(tuned, name) == getattr(patient, name)


FORBIDDEN = [('t1d', field) for field in ('kp1', 'Vm0')]
FORBIDDEN += [(kind, field) for kind in KINDS[1:]
              for field in ('h', 'F_cns0', 'Sb_per_kg', 'S_I1', 'S_I2', 'S_I3', 'EGP_0')]


@pytest.mark.parametrize('kind,field', FORBIDDEN)
@pytest.mark.parametrize('extra', [{}, {'BW': 100.0, 'autobalance_enabled': False}])
def test_output_overrides_rejected_before_loading(monkeypatch, kind, field, extra):
    def unexpected_load(*args):
        pytest.fail('Invalid factory overrides must fail before CSV loading')
    monkeypatch.setattr(patient_loader, 'load_patient_parameters_from_csv', unexpected_load)
    with pytest.raises(ValueError, match=field):
        create(kind, **dict(extra, **{field: 123.0}))


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('field,value', [
    *[(name, float('nan')) for name in (*EFFECTIVE, 'insulin_resistance_factor')],
    *[(name, -1.0) for name in (*EFFECTIVE, 'insulin_resistance_factor')],
    *[(name, 0.0) for name in ('BW', 'Vg', 'Vi', 'insulin_resistance_factor')],
    ('BW', True), ('Fsnc', '1.8'), ('Gpb', float('inf')),
])
def test_invalid_effective_inputs_fail_before_loading(monkeypatch, kind, field, value):
    def unexpected_load(*args):
        pytest.fail('Invalid effective input reached CSV loading')
    monkeypatch.setattr(patient_loader, 'load_patient_parameters_from_csv', unexpected_load)
    with pytest.raises(ValueError, match=field):
        create(kind, **{field: value})


def test_t1d_resistance_has_no_unimplemented_gain_policy(synthetic_source):
    patient = create('t1d', insulin_resistance_factor=1.0)
    assert (patient.S_I1, patient.S_I2, patient.S_I3) == (0.001, 0.01, 0.005)
    with pytest.raises(ValueError, match='insulin_resistance_factor'):
        create('t1d', insulin_resistance_factor=5.0)


def test_t1d_autobalance_false_skips_only_factory_tuning(monkeypatch, synthetic_source,
                                                       env_template):
    def unexpected_balance(*args, **kwargs):
        pytest.fail('Factory autobalance was explicitly disabled')
    monkeypatch.setattr(calibration, 'autobalance_basal_t1d', unexpected_balance)
    patient = create('t1d', autobalance_enabled=False)
    patient = dataclasses.replace(patient, Gpb=100.0, Gtb=100.0, Vi=0.05,
                                  EGPb=2.0, kp2=0.01, kp3=0.02, Fsnc=1.0,
                                  Km0=100.0, Vmx=0.05, ke2=200.0,
                                  kp1=123.0, Vm0=456.0)
    state = jnp.zeros(18).at[3].set(100.0).at[4].set(100.0).at[5].set(0.5)
    monkeypatch.setattr(initialization, 'init_state_t1d', lambda p: state)
    tuned, actual_state = initialization.tune_initial_state(
        dataclasses.replace(env_template, patient_params=patient))
    # I=10 pmol/L. Production target: 2 + .01*100 + .02*10 = 3.2.
    assert tuned.patient_params.kp1 == pytest.approx(3.2)
    # (production 2 - CNS 1) * (100+100)/100 - .05*10 = 1.5.
    assert tuned.patient_params.Vm0 == pytest.approx(1.5)
    assert actual_state is state


@pytest.mark.parametrize('kind', KINDS[1:])
def test_t2d_initialization_recomputes_egp_from_independent_balance(
        monkeypatch, synthetic_source, env_template, kind):
    patient = dataclasses.replace(create(kind), BW=90.0, Vg=2.0, Vi=0.05,
                                  Gpb=100.0, Gtb=100.0, Km0=100.0,
                                  Vm0=2.0, Vmx=0.0, k1=0.02, k2=0.01,
                                  F_cns0=0.5, S_I3=0.0, ke2=200.0, EGP_0=123.0)
    state = jnp.zeros(18).at[3].set(100.0).at[4].set(100.0).at[5].set(0.3)
    monkeypatch.setattr(initialization, 'init_state_t2d', lambda p: state)
    tuned, _ = initialization.tune_initial_state(
        dataclasses.replace(env_template, patient_params=patient))
    # Uptake=2*100/200=1 and CNS=.5*180/90=1 mg/kg/min.
    # Tissue exchange=.02*100-.01*100-1=0, so EGP=(1+1)*90/180=1.
    assert tuned.patient_params.EGP_0 == pytest.approx(1.0)
    assert tuned.patient_params.Vm0 == 2.0
    assert tuned.patient_params.F_cns0 == 0.5


@pytest.mark.parametrize('kind', KINDS)
def test_late_behavior_and_stress_scaling_remain_once(synthetic_source, kind):
    effective = dict(EFFECTIVE, insulin_resistance_factor=1.0 if kind == 't1d' else 5.0)
    baseline = create(kind, **effective)
    stressed = create(kind, **effective, carb_absorption_scale=1.4,
                      insulin_sensitivity_scale=0.8, eat_rate_scale=1.2,
                      bolus_acceptance_prob=0.37)
    assert stressed.kabs == baseline.kabs * 1.4
    assert stressed.kmax == baseline.kmax * 1.4
    assert stressed.Vmx == baseline.Vmx * 0.8
    assert stressed.eat_rate == baseline.eat_rate * 1.2
    assert stressed.bolus_acceptance_prob == 0.37


@pytest.mark.parametrize('kind', KINDS[1:])
def test_positive_effective_resistance_below_legacy_floor_is_not_clamped(
        synthetic_source, kind):
    patient = create(kind, insulin_resistance_factor=5e-7)
    assert patient.insulin_resistance_factor == 5e-7
    # Explicit effective resistance is positive, so source gains / 0.0000005.
    assert (patient.S_I1, patient.S_I2, patient.S_I3) == pytest.approx(
        (2000.0, 20000.0, 10000.0))


def test_direct_conversion_keeps_legacy_resistance_floor(synthetic_source):
    source = dataclasses.replace(create('t1d'), insulin_resistance_factor=5e-7)
    converted = params.patient_to_t2d_params(source)
    # The existing direct conversion interface keeps its historical 1e-6 floor.
    assert (converted.S_I1, converted.S_I2, converted.S_I3) == pytest.approx(
        (1000.0, 10000.0, 5000.0))


@pytest.mark.parametrize('field', ('Gpb', 'Fsnc', 'EGPb'))
def test_zero_is_valid_for_nonnegative_effective_inputs(synthetic_source, field):
    # No initialization or equilibrium claim: this checks the input boundary.
    patient = create('t1d', autobalance_enabled=False, **{field: 0.0})
    assert getattr(patient, field) == 0.0


@pytest.mark.parametrize('kind', KINDS[1:])
def test_changed_volume_and_cns_inputs_survive_conversion_and_initialization(
        synthetic_source, env_template, kind):
    # Every quantity below differs from the synthetic source (Vg2, Vi.05,
    # Fsnc1.8, BW80). This detects an override accidentally left until after
    # conversion even when the requested value might otherwise equal source.
    patient = create(kind, BW=100.0, Vg=4.0, Vi=0.1, Gpb=180.0, Fsnc=3.6)
    tuned_env, _ = initialization.tune_initial_state(
        dataclasses.replace(env_template, patient_params=patient))
    for candidate in (patient, tuned_env.patient_params):
        assert candidate.Vg == 4.0
        assert candidate.Vi == 0.1
        assert candidate.Fsnc == 3.6
        assert candidate.h == pytest.approx(2.5)  # 180 / 4 / 18 mM.
        assert candidate.V_I == pytest.approx(10.0)  # .1 L/kg * 100 kg.
        assert candidate.V_I_L == pytest.approx(4.0)  # Deferred source alias.
        assert candidate.F_cns0 == pytest.approx(2.0)  # 3.6 * 100 / 180.
        assert candidate.F_cns0 * 180.0 / candidate.BW == pytest.approx(3.6)


def test_changed_t1d_egp_target_survives_construction_and_initialization(
        synthetic_source, env_template):
    assert synthetic_source['EGPb'] != 3.0
    patient = create('t1d', EGPb=3.0)
    tuned_env, state = initialization.tune_initial_state(
        dataclasses.replace(env_template, patient_params=patient))
    tuned = tuned_env.patient_params
    assert patient.EGPb == tuned.EGPb == 3.0
    # Initialization sets linear hepatic production to the requested target;
    # the state supplies actual glucose mass and insulin concentration.
    production = (tuned.kp1 - tuned.kp2 * float(state[3])
                  - tuned.kp3 * float(state[5]) / tuned.Vi)
    assert production == pytest.approx(3.0)
