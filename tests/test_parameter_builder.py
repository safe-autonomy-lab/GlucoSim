"""Legacy builder contracts: entry state, source values and precedence."""
import dataclasses
import subprocess
import sys

import pytest

from examples.characterize_simulator import digest
from glucosim import gym_env as gym
from glucosim.simglucose.core import params, patient_loader
from glucosim.simglucose.core.configuration import LegacyBuildOptions


@pytest.fixture(scope='module')
def supplied():
    return params.create_patient_params('adolescent#001', diabetes_type='t1d')


def test_options_are_immutable_and_acceptance_is_explicit():
    with pytest.raises(TypeError):
        LegacyBuildOptions()
    options = LegacyBuildOptions(acceptance_probability=0.35)
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.acceptance_probability = 1.0


@pytest.mark.parametrize('kind', ['t1d', 't2d', 't2d_no_pump'])
def test_live_acceptance_and_factory_late_override(monkeypatch, supplied, kind):
    adapt = getattr(params, 'adapt_params_for_' + kind)
    for probability in (0.35, 0.7):
        monkeypatch.setattr(params, 'ACCEPTANCE_PROB_DEFAULT', probability)
        adapted = adapt(supplied)
        created = params.create_patient_params('adolescent#001', diabetes_type=kind,
                                               bolus_acceptance_prob=0.8)
        assert adapted.meal_acceptance_prob == adapted.bolus_acceptance_prob == probability
        assert created.meal_acceptance_prob == created.exercise_acceptance_prob == probability
        assert created.bolus_acceptance_prob == 0.8


@pytest.mark.parametrize('kind', ['t1d', 't2d', 't2d_no_pump'])
def test_gym_default_precedence_is_separate(monkeypatch, kind):
    monkeypatch.setattr(params, 'ACCEPTANCE_PROB_DEFAULT', 0.35)
    env = gym.make(kind + '-v0', patient_overrides={'bolus_acceptance_prob': 0.6})
    try:
        patient = env.unwrapped.env_params.patient_params
        assert patient.meal_acceptance_prob == patient.exercise_acceptance_prob == 1.0
        assert patient.bolus_acceptance_prob == 0.6
    finally:
        env.close()


@pytest.mark.parametrize('kind', ['t1d', 't2d', 't2d_no_pump'])
def test_direct_input_never_reloads_csv_or_mutates(monkeypatch, supplied, kind):
    def no_csv(*args, **kwargs):
        pytest.fail('Direct adapter attempted to reconstruct the input from CSV')
    monkeypatch.setattr(patient_loader, 'load_patient_parameters_from_csv', no_csv)
    direct = dataclasses.replace(supplied, BW=81.25, Vi=0.065, S_I1=supplied.S_I1 * 1.7)
    before = digest(direct)
    adapt = getattr(params, 'adapt_params_for_' + kind)
    kwargs = dict(carb_absorption_scale=1.4, insulin_sensitivity_scale=0.8, eat_rate_scale=1.2)
    config = dict(Ib_factor=1.1, Ipb_factor=1.2, HEb_factor=0.9, BW_factor=1.05)
    original = config.copy()
    if kind != 't1d':
        kwargs['config'] = config
    first = adapt(direct, **kwargs)
    assert digest(first) == digest(adapt(direct, **kwargs))
    assert digest(direct) == before and config == original
    if kind == 't1d':
        assert first.BW == direct.BW
        assert first.basal == direct.BW * 0.011
    else:
        assert first.BW == direct.BW * config['BW_factor']
        assert first.V_I == direct.Vi * first.BW
        assert first.V_G == direct.Vg * first.BW / 10.0
        assert first.S_I1 == direct.S_I1 / (2.5 if kind == 't2d' else 2.8)
        # Preserve legacy original-volume aliases, even on a modified input.
        assert first.V_G_L == direct.V_G_L and first.V_I_L == direct.V_I_L


@pytest.mark.parametrize('kind', ['t2d', 't2d_no_pump'])
def test_legacy_config_fallback_and_partial_error(supplied, kind):
    adapt = getattr(params, 'adapt_params_for_' + kind)
    assert digest(adapt(supplied, config=None)) == digest(adapt(supplied, config={}))
    config = {'BW_factor': 1.05}
    with pytest.raises(KeyError, match='Ib_factor'):
        adapt(supplied, config=config)
    assert config == {'BW_factor': 1.05}


@pytest.mark.parametrize('kind', ['t1d', 't2d', 't2d_no_pump'])
def test_repeated_raw_build_does_not_accumulate_scales(kind):
    kwargs = dict(carb_absorption_scale=1.4, insulin_sensitivity_scale=0.8, eat_rate_scale=1.2)
    first = params.create_patient_params('adolescent#001', diabetes_type=kind, **kwargs)
    assert digest(first) == digest(params.create_patient_params('adolescent#001', diabetes_type=kind, **kwargs))
    base = params.create_patient_params('adolescent#001', diabetes_type=kind)
    assert first.kabs == base.kabs * 1.4
    assert first.eat_rate == base.eat_rate * 1.2
    assert first.Vmx == base.Vmx * 0.8


@pytest.mark.parametrize('order', [
    ('params', 'parameter_builder', 'conversion'),
    ('parameter_builder', 'conversion', 'params'),
    ('conversion', 'params', 'parameter_builder'),
])
def test_fresh_import_order_and_runtime_class_identity(order):
    script = '\n'.join('from glucosim.simglucose.core import ' + name for name in order)
    script += '\nassert parameter_builder.PatientParams is params.PatientParams\n'
    script += 'from glucosim.simglucose.core import PatientParams\nassert PatientParams is params.PatientParams\n'
    subprocess.run([sys.executable, '-c', script], check=True)
