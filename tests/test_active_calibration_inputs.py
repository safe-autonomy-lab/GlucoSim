"""Effective calibration inputs checked against independent balance equations.

The source fixture uses simple masses/rates; no expected value calls conversion,
calibration, or a steady-state solver from the implementation.
"""
import dataclasses
from pathlib import Path

import pytest

from glucosim.simglucose.core import params, patient_loader
from glucosim.simglucose.physiology import initialization


KINDS = ('t1d', 't2d', 't2d_no_pump')
T1D_INPUTS = dict(k1=.04, k2=.015, Km0=150., Gtb=80., kp2=.02,
                  kp3=.03, ke1=.02, ke2=50.)
T2D_INPUTS = dict(HEb=.5, m1=.3, m2=.15, m30=.25, m4=.15,
                  k1=.04, k2=.015, Km0=150., ke1=.02, ke2=50., k_a3=.1)
CASES = [('t1d', k, v) for k, v in T1D_INPUTS.items()]
CASES += [(kind, k, v) for kind in KINDS[1:] for k, v in T2D_INPUTS.items()]
CASES += [('t2d', 'Ib', 400.)]


@pytest.fixture(scope='module')
def source_row():
    path = Path(patient_loader.__file__).parent.parent / 'params/vpatient_params.csv'
    row = patient_loader.load_patient_parameters_from_csv(str(path))['adolescent#001']
    return dict(row, BW=80., Vg=2., Vi=.05, Gpb=100., Gtb=100.,
                Ipb=1., Ilb=1., Ib=200., HEb=.4, m1=.2, m2=.1,
                m30=.2, m4=.1, k1=.03, k2=.01, Km0=100.,
                kp2=.01, kp3=.02, ke1=.01, ke2=80., Vm0=1.,
                Fsnc=1., EGPb=3., S_I1=.001, S_I2=.01, S_I3=.0005,
                k_a3=.05)


@pytest.fixture
def source(monkeypatch, source_row):
    monkeypatch.setattr(patient_loader, 'load_patient_parameters_from_csv',
                        lambda path: {'adolescent#001': source_row.copy()})
    return source_row


@pytest.fixture(scope='module')
def env_template():
    return params.create_env_params('adolescent#001', diabetes_type='t1d')


def create(kind, **overrides):
    return params.create_patient_params('adolescent#001', diabetes_type=kind, **overrides)


def assert_factory_balances(patient, kind, source, overrides):
    """Conservation identities, evaluated from requested inputs and raw source."""
    inputs = dict(source)
    if kind != 't1d':
        inputs.update(BW=92., HEb=.34, Ib=250.)
    inputs.update(overrides)
    uptake = inputs['k1'] * 100. - inputs['k2'] * inputs['Gtb']
    saturation = inputs['Gtb'] / (inputs['Km0'] + inputs['Gtb'])
    renal = inputs['ke1'] * max(100. - inputs['ke2'], 0.)
    if kind == 't1d':
        # Source insulin concentration = 1 pmol/kg / .05 L/kg = 20 pmol/L.
        assert patient.Vmx == pytest.approx((uptake / saturation - 1.) / 20.)
        assert patient.kp1 == pytest.approx(inputs['kp2'] * 100. + inputs['kp3'] * 20.
                                            + 1. + renal + uptake)
        return
    secretion = (400. / 60.) / ((1. - inputs['HEb']) * 92.)
    assert patient.Sb_per_kg == pytest.approx(secretion)
    # Solve the two linear insulin balances by eliminating liver mass.
    clearance = inputs['m2'] + inputs['m4'] - inputs['m1'] * inputs['m2'] / (inputs['m1'] + inputs['m30'])
    endogenous_plasma = inputs['m1'] * 6. * secretion / (inputs['m1'] + inputs['m30'])
    plasma_mass = endogenous_plasma / clearance
    liver_mass = (inputs['m2'] * plasma_mass + 6. * secretion) / (inputs['m1'] + inputs['m30'])
    assert patient.Ipb == pytest.approx(plasma_mass)
    assert patient.Ilb == pytest.approx(liver_mass)
    if kind == 't2d':
        expected_basal = max(0., (clearance * inputs['Ib'] * .05 - endogenous_plasma) * 92. / 100.)
        assert patient.basal == pytest.approx(expected_basal)
        # Pump glucose calibration precedes its insulin-pool replacement.
        calibration_insulin = 25.  # source Ipb * 1.25 / Vi
        resistance = 2.5
    else:
        assert patient.basal == 0.
        calibration_insulin = plasma_mass / .05
        resistance = 2.8
    assert patient.Vmx == pytest.approx((uptake / saturation - 1.) / calibration_insulin)
    suppression = (.0005 / resistance) / inputs['k_a3'] * calibration_insulin / 6.
    assert patient.EGP_0 == pytest.approx((uptake + 1. + renal) * 92. / 180. / (1. - suppression))


@pytest.mark.parametrize('kind,field,value', CASES)
def test_each_effective_input_reaches_its_factory_dependencies(source, kind, field, value):
    patient = create(kind, **{field: value})
    assert getattr(patient, field) == value
    assert_factory_balances(patient, kind, source, {field: value})


@pytest.mark.parametrize('kind', KINDS)
def test_combined_effective_inputs_reach_factory_dependencies(source, kind):
    overrides = dict(T1D_INPUTS if kind == 't1d' else T2D_INPUTS)
    if kind == 't2d':
        overrides['Ib'] = 400.
    patient = create(kind, **overrides)
    for field, value in overrides.items():
        assert getattr(patient, field) == value
    assert_factory_balances(patient, kind, source, overrides)


@pytest.mark.parametrize('kind,field,value', CASES)
def test_direct_initialization_retains_effective_inputs_but_retunes_owned_outputs(
        source, env_template, kind, field, value):
    patient = create(kind, **{field: value})
    tuned_env, state = initialization.tune_initial_state(dataclasses.replace(env_template, patient_params=patient))
    tuned = tuned_env.patient_params
    assert getattr(tuned, field) == value
    assert tuned.Vmx == patient.Vmx
    glucose, tissue, insulin = float(state[3]), float(state[4]), float(state[5]) / tuned.Vi
    if kind == 't1d':
        # Initialization targets EGPb, not the factory's exchange-based kp1.
        assert tuned.kp1 - tuned.kp2 * glucose - tuned.kp3 * insulin == pytest.approx(3.)
        required_uptake = max(3. - 1. - tuned.ke1 * max(glucose - tuned.ke2, 0.), 0.)
        assert tuned.Vm0 == pytest.approx(max(required_uptake * (tuned.Km0 + tissue) / tissue - tuned.Vmx * insulin, 0.))
    else:
        assert tuned.Vm0 == patient.Vm0
        assert tuned.Sb_per_kg == patient.Sb_per_kg
        uptake = (tuned.Vm0 + tuned.Vmx * insulin) * tissue / (tuned.Km0 + tissue)
        assert tuned.k1 * glucose - tuned.k2 * tissue == pytest.approx(uptake, abs=2e-6)
        suppression = min(tuned.S_I3 / tuned.k_a3 * insulin / 6., .95)
        hepatic = tuned.EGP_0 * 180. / tuned.BW * (1. - suppression)
        assert hepatic == pytest.approx(1. + uptake + tuned.ke1 * max(glucose - tuned.ke2, 0.), rel=2e-6)
        # Insulin is recomputed from delivery + secretion; factory Ipb is not an
        # explicit initial-state promise for the pump model.
        delivered = patient.basal * 100. / patient.BW if patient.use_pump else 0.
        plasma, liver = float(state[5]), float(state[9])
        assert -(patient.m2 + patient.m4) * plasma + patient.m1 * liver + delivered == pytest.approx(0., abs=2e-6)
        assert patient.m2 * plasma - (patient.m1 + patient.m30) * liver + 6. * patient.Sb_per_kg == pytest.approx(0., abs=2e-6)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('basal', [.5, -.5])
def test_nonzero_basal_without_pump_is_rejected_after_simultaneous_overrides(source, kind, basal):
    with pytest.raises(ValueError, match='(?i)(use_pump.*basal|basal.*use_pump)'):
        create(kind, use_pump=False, basal=basal)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('pump,basal', [(False, 0.), (True, .5)])
def test_consistent_delivery_overrides_remain_supported(source, kind, pump, basal):
    patient = create(kind, use_pump=pump, basal=basal)
    assert patient.use_pump is pump
    assert patient.basal == basal


@pytest.mark.parametrize('kind', ('t1d', 't2d'))
def test_disabling_pump_rejects_resolved_nonzero_preset_basal(source, kind):
    with pytest.raises(ValueError, match='(?i)(use_pump.*basal|basal.*use_pump)'):
        create(kind, use_pump=False)


def test_no_pump_nonzero_basal_alone_is_rejected(source):
    with pytest.raises(ValueError, match='(?i)(use_pump.*basal|basal.*use_pump)'):
        create('t2d_no_pump', basal=.5)


@pytest.mark.parametrize('kind', KINDS)
def test_calibration_output_override_acceptance_is_not_broadened_or_removed(source, kind):
    patient = create(kind, autobalance_enabled=False, Vmx=.123, k1=.04)
    assert patient.Vmx == .123
    if kind != 't1d':
        assert create(kind, autobalance_enabled=False, Vm0=7.).Vm0 == 7.


@pytest.mark.parametrize('kind', KINDS[1:])
def test_t2d_gtb_and_linear_hepatic_fields_remain_unlisted_late_overrides(source, kind):
    default = create(kind)
    altered = create(kind, Gtb=85., kp2=.2, kp3=.3)
    assert (altered.Gtb, altered.kp2, altered.kp3) == (85., .2, .3)
    assert (altered.Vmx, altered.EGP_0) == (default.Vmx, default.EGP_0)


def test_no_pump_ib_remains_late_without_a_final_target_consumer(source):
    default = create('t2d_no_pump')
    altered = create('t2d_no_pump', Ib=400.)
    assert altered.Ib == 400.
    for field in ('basal', 'Sb_per_kg', 'Ipb', 'Ilb', 'Vmx', 'EGP_0'):
        assert getattr(altered, field) == getattr(default, field)


def test_no_pump_effective_inputs_keep_original_gains_and_one_stress_pass(source):
    baseline = create('t2d_no_pump', **T2D_INPUTS, insulin_resistance_factor=5.)
    stressed = create('t2d_no_pump', **T2D_INPUTS, insulin_resistance_factor=5.,
                      insulin_sensitivity_scale=.8, carb_absorption_scale=1.4)
    assert (baseline.S_I1, baseline.S_I2, baseline.S_I3) == pytest.approx((.0002, .002, .0001))
    assert stressed.Vmx == baseline.Vmx * .8
    assert stressed.kabs == baseline.kabs * 1.4
    assert stressed.kmax == baseline.kmax * 1.4


@pytest.mark.parametrize('kind', KINDS)
def test_default_presets_still_satisfy_original_factory_balances(source, kind):
    patient = create(kind)
    assert_factory_balances(patient, kind, source, {})
    assert patient.BW == pytest.approx(80. if kind == 't1d' else 92.)
    if kind != 't1d':
        assert patient.HEb == pytest.approx(.34)
        assert patient.Ib == 250.
