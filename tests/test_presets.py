"""Keep legacy configuration semantics through the preset-data extraction."""
import pytest

from examples.characterize_simulator import digest
from glucosim.simglucose.core import params, presets


def test_default_scaling_is_not_shared():
    config = presets.t2d_scaling_defaults()
    config['BW_factor'] = 99.0
    assert presets.t2d_scaling_defaults()['BW_factor'] == 1.15


@pytest.mark.parametrize('kind', ['t1d', 't2d', 't2d_no_pump'])
def test_acceptance_setting_and_final_override(monkeypatch, kind):
    monkeypatch.setattr(params, 'ACCEPTANCE_PROB_DEFAULT', 0.35)
    patient = params.create_patient_params('adolescent#001', diabetes_type=kind,
                                           bolus_acceptance_prob=0.8)
    assert patient.bolus_acceptance_prob == 0.8
    assert patient.meal_acceptance_prob == 0.35
    assert patient.exercise_acceptance_prob == 0.35


@pytest.mark.parametrize('kind', ['t2d', 't2d_no_pump'])
def test_empty_and_custom_scaling_config(kind):
    base = params.create_patient_params('adolescent#001', diabetes_type='t1d')
    adapt = getattr(params, f'adapt_params_for_{kind}')
    assert digest(adapt(base, config={})) == digest(adapt(base))
    config = dict(Ib_factor=1.1, Ipb_factor=1.2, HEb_factor=0.9, BW_factor=1.05)
    original = config.copy()
    patient = adapt(base, config=config)
    assert config == original
    assert patient.BW == base.BW * config['BW_factor']
