"""Construction reaches implementation owners without reentering public facades."""
import pytest

from examples.characterize_simulator import digest
from glucosim.simglucose.core import conversion, params, patient_loader
from glucosim.simglucose.physiology import calibration
from glucosim.simglucose.sim import scenario_gen


@pytest.mark.parametrize('kind,expected', [
    ('t1d', ['load', 't1d']),
    ('t2d', ['load', 'convert', 'pump']),
    ('t2d_no_pump', ['load', 'convert', 'pump', 'no_pump']),
])
def test_patient_construction_calls_owners_without_facade_callbacks(monkeypatch, kind, expected):
    reference = params.create_patient_params('adolescent#001', diabetes_type=kind)
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail('Builder called a compatibility facade instead of the implementation owner')

    for name in ('load_patient_parameters_from_csv', 'patient_to_t2d_params',
                 'autobalance_basal_t1d', 'autobalance_basal_t2d',
                 '_steady_state_insulin_from_Sb'):
        monkeypatch.setattr(params, name, forbidden)

    for owner, name, label in [
        (patient_loader, 'load_patient_parameters_from_csv', 'load'),
        (conversion, 'patient_to_t2d_params', 'convert'),
        (calibration, 'autobalance_basal_t1d', 't1d'),
        (calibration, 'calibrate_t2d_pump', 'pump'),
        (calibration, 'calibrate_t2d_no_pump', 'no_pump'),
    ]:
        original = getattr(owner, name)

        def tracked(*args, _original=original, _label=label, **kwargs):
            calls.append(_label)
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, tracked)

    result = params.create_patient_params('adolescent#001', diabetes_type=kind)
    assert calls == expected
    assert type(result) is params.PatientParams
    assert digest(result) == digest(reference)


@pytest.mark.parametrize('kind', ['t1d', 't2d', 't2d_no_pump'])
def test_environment_construction_does_not_reenter_patient_factory(monkeypatch, kind):
    overrides = dict(carb_absorption_scale=1.2, insulin_sensitivity_scale=0.8,
                     bolus_acceptance_prob=0.6, BW=79.0)
    monkeypatch.setattr(params, 'ACCEPTANCE_PROB_DEFAULT', 0.35)
    reference = params.create_env_params('adolescent#001', diabetes_type=kind,
                                         simulation_minutes=720, sample_time=10, **overrides)

    def forbidden(*args, **kwargs):
        pytest.fail('Environment builder called the public patient factory')

    monkeypatch.setattr(params, 'create_patient_params', forbidden)
    result = params.create_env_params('adolescent#001', diabetes_type=kind,
                                      simulation_minutes=720, sample_time=10, **overrides)
    assert type(result) is params.EnvParams
    assert type(result.patient_params) is params.PatientParams
    assert type(result.noise_config) is params.NoiseConfig
    assert result.patient_params.meal_acceptance_prob == 0.35
    assert result.patient_params.bolus_acceptance_prob == 0.6
    assert digest(result) == digest(reference)


def test_environment_resolves_meal_profile_before_patient_type_validation(monkeypatch):
    class ProfileFailure(Exception):
        pass

    received = []

    def fail_profile(cohort):
        received.append(cohort)
        raise ProfileFailure('meal profile reached first')

    def forbidden_load(*args, **kwargs):
        pytest.fail('Invalid environment type reached CSV loading')

    monkeypatch.setattr(scenario_gen, 'get_meal_profile_for_cohort', fail_profile)
    monkeypatch.setattr(patient_loader, 'load_patient_parameters_from_csv', forbidden_load)
    with pytest.raises(ProfileFailure, match='meal profile reached first'):
        params.create_env_params('ADOLESCENT#001', diabetes_type='invalid')
    assert received == ['adolescent']
