"""Public factories resolve defaults and reject invalid types before CSV I/O."""
import pytest

from examples.characterize_simulator import digest
from glucosim.simglucose.core import params
from glucosim.simglucose.core.types import PatientType


@pytest.mark.parametrize('factory', [params.create_patient_params, params.create_env_params])
@pytest.mark.parametrize('type_kwargs', [{}, {'diabetes_type': None}])
def test_omitted_or_none_type_matches_explicit_t1d(factory, type_kwargs):
    kwargs = dict(patient_name='adolescent#001', bolus_acceptance_prob=0.8,
                  carb_absorption_scale=1.2)
    assert digest(factory(**kwargs, **type_kwargs)) == digest(factory(**kwargs, diabetes_type='t1d'))


def test_environment_documented_no_argument_call():
    assert digest(params.create_env_params()) == digest(params.create_env_params(diabetes_type='t1d'))


@pytest.mark.parametrize('factory', [params.create_patient_params, params.create_env_params])
@pytest.mark.parametrize('kind', ['t1d', 't2d', 't2d_no_pump'])
def test_explicit_supported_types(factory, kind):
    result = factory('adolescent#001', diabetes_type=kind)
    patient = result.patient_params if isinstance(result, params.EnvParams) else result
    assert patient.diabetes_type == getattr(PatientType, kind)


@pytest.mark.parametrize('factory', [params.create_patient_params, params.create_env_params])
@pytest.mark.parametrize('kind', ['', 'T1D', 'unknown', '__dict__', 0, False, [], {}])
def test_invalid_type_is_rejected_before_loading(monkeypatch, factory, kind):
    def unexpected_load(*args, **kwargs):
        pytest.fail('Invalid diabetes type reached CSV loading')

    monkeypatch.setattr(params, 'load_patient_parameters_from_csv', unexpected_load)
    with pytest.raises(ValueError, match="Invalid diabetes_type:.*Must be 't1d', 't2d', or 't2d_no_pump'"):
        factory('adolescent#001', diabetes_type=kind, csv_path='unused.csv')
