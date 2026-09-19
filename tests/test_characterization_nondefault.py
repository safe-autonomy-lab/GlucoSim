"""Ensure the small differential matrix can expose lost public factory inputs."""
from copy import deepcopy
import csv
import io
import json

import pytest

from examples import characterize_simulator as harness
from examples.characterization_contract import factory_record_stages, expected_rejection, rejection_digest


@pytest.fixture(scope='module')
def manifest():
    return json.loads(harness.MANIFEST.read_text())


@pytest.fixture(scope='module')
def records(manifest):
    return harness.capture_nondefault(manifest)


def test_matrix_keys_and_distinguishable_factory_cases(manifest, records):
    expected = set()
    for kind in manifest['types']:
        prefix = f'nondefault/adolescent#001/{kind}/factory/'
        for name, case in manifest['nondefault']['factory_cases'].items():
            if kind not in case.get('types', manifest['types']):
                continue
            expected.update(prefix + name + '/' + stage for stage in factory_record_stages(case, kind))
            if expected_rejection(case, kind) is not None:
                continue
            if name != 'default' and not name.startswith('t2d_autobalance'):
                for stage in ('patient', 'created'):
                    assert records[prefix + name + '/' + stage] != records[prefix + 'default/' + stage]
            if name.startswith('t2d_autobalance'):
                # Characterize the current ignored options without changing semantics.
                for stage in ('patient', 'created', 'tuned'):
                    assert records[prefix + name + '/' + stage] == records[prefix + 'default/' + stage]
        if kind != 't1d':
            adapter = f'nondefault/adolescent#001/{kind}/adapter/'
            expected.update(adapter + name for name in manifest['nondefault']['adapter_configs'])
            assert records[adapter + 'empty'] != records[adapter + 'custom']
    assert set(records) == expected


def test_historical_no_pump_basal_case_is_explicitly_rejected(manifest):
    subset = deepcopy(manifest)
    subset['types'] = ['t2d_no_pump']
    subset['nondefault']['factory_cases'] = {
        'final_override': subset['nondefault']['factory_cases']['final_override']}
    assert subset['nondefault']['factory_cases']['final_override']['kwargs']['basal'] == 0.35
    captured = harness.capture_nondefault(subset)
    prefix = 'nondefault/adolescent#001/t2d_no_pump/factory/final_override/'
    spec = expected_rejection(subset['nondefault']['factory_cases']['final_override'], 't2d_no_pump')
    for stage in ('patient_rejection', 'created_rejection'):
        assert captured[prefix + stage] == rejection_digest(stage, spec)
    assert not any(prefix + stage in captured for stage in ('patient', 'created', 'tuned'))


@pytest.mark.parametrize('case,argument', [
    ('carb_scale', 'carb_absorption_scale'),
    ('insulin_scale', 'insulin_sensitivity_scale'),
    ('eat_scale', 'eat_rate_scale'),
    ('autobalance_basal', 'autobalance_basal_scale'),
    ('autobalance_hepatic', 'autobalance_hepatic_scale'),
    ('autobalance_disabled', 'autobalance_enabled'),
    ('final_override', 'kabs'),
    ('final_override', 'basal'),
    ('acceptance', 'bolus_acceptance_prob'),
    ('alternative_csv', 'csv_path'),
])
def test_matrix_detects_dropped_env_factory_argument(monkeypatch, manifest, records, case, argument):
    subset = deepcopy(manifest)
    subset['types'] = ['t1d']
    subset['nondefault']['factory_cases'] = {case: subset['nondefault']['factory_cases'][case]}
    original = harness.create_env_params

    def dropping(*args, **kwargs):
        kwargs.pop(argument, None)
        return original(*args, **kwargs)

    monkeypatch.setattr(harness, 'create_env_params', dropping)
    mutated = harness.capture_nondefault(subset)
    prefix = f'nondefault/adolescent#001/t1d/factory/{case}'
    assert mutated[prefix + '/patient'] == records[prefix + '/patient']
    assert mutated[prefix + '/created'] != records[prefix + '/created']


def test_csv_fixture_is_deterministic_and_distinct(manifest):
    inputs = harness.patient_inputs(manifest)
    assert inputs == harness.patient_inputs(manifest)
    original = list(csv.DictReader(io.StringIO(inputs[harness.PATIENT_CSV].decode())))
    alternative = list(csv.DictReader(io.StringIO(inputs['generated/alternative_patient.csv'].decode())))
    for old, new in zip(original, alternative):
        if old['Name'] == manifest['nondefault']['patient']:
            expected = dict(old, **{key: str(value) for key, value in manifest['nondefault']['csv_changes'].items()})
            assert new == expected and new != old
        else:
            assert new == old


def test_acceptance_restored_on_capture_failure(monkeypatch, manifest):
    original = harness.patient_factory.ACCEPTANCE_PROB_DEFAULT
    subset = deepcopy(manifest)
    subset['nondefault']['factory_cases'] = {'acceptance': {'acceptance': 0.27}}

    def failing(*args, **kwargs):
        assert harness.patient_factory.ACCEPTANCE_PROB_DEFAULT == 0.27
        raise RuntimeError('injected failure')

    monkeypatch.setattr(harness, 'create_env_params', failing)
    with pytest.raises(RuntimeError, match='injected'):
        harness.capture_nondefault(subset)
    assert harness.patient_factory.ACCEPTANCE_PROB_DEFAULT == original


def rejection_manifest(manifest):
    subset = deepcopy(manifest)
    subset['types'] = ['t2d_no_pump']
    subset['nondefault']['factory_cases'] = {
        'final_override': subset['nondefault']['factory_cases']['final_override']}
    subset['nondefault']['adapter_configs'] = {}
    return subset


@pytest.mark.parametrize('entrypoint', ['patient', 'created'])
@pytest.mark.parametrize('failure', ['success', 'wrong_type', 'wrong_reason', 'wrong_location', 'subclass'])
def test_rejection_requires_exact_factory_guard(monkeypatch, manifest, entrypoint, failure):
    subset = rejection_manifest(manifest)

    def invalid_factory(*args, **kwargs):
        if failure == 'success':
            return object()
        if failure == 'wrong_type':
            raise RuntimeError('use_pump=False requires basal=0')
        if failure == 'wrong_reason':
            raise ValueError('different error')
        if failure == 'subclass':
            class OtherError(ValueError):
                pass
            raise OtherError('use_pump=False requires basal=0')
        raise ValueError('use_pump=False requires basal=0')

    target = harness.patient_factory if entrypoint == 'patient' else harness
    name = 'create_patient_params' if entrypoint == 'patient' else 'create_env_params'
    monkeypatch.setattr(target, name, invalid_factory)
    with pytest.raises((AssertionError, RuntimeError)):
        harness.capture_nondefault(subset)


def test_same_message_loader_failure_is_not_expected_rejection(monkeypatch, manifest):
    from glucosim.simglucose.core import parameter_builder

    def fail(*args, **kwargs):
        raise ValueError('use_pump=False requires basal=0')

    monkeypatch.setattr(parameter_builder.patient_loader, 'load_patient_parameters_from_csv', fail)
    with pytest.raises(AssertionError, match='did not match'):
        harness.capture_nondefault(rejection_manifest(manifest))


def test_undeclared_invalid_case_still_fails(manifest):
    subset = rejection_manifest(manifest)
    del subset['nondefault']['factory_cases']['final_override']['expected_rejection']
    with pytest.raises(ValueError, match='use_pump=False requires basal=0'):
        harness.capture_nondefault(subset)


def test_valid_no_pump_late_override_values(manifest):
    case = manifest['nondefault']['factory_cases']['final_override_no_pump_valid']
    for factory in (harness.patient_factory.create_patient_params, harness.create_env_params):
        value = factory('adolescent#001', diabetes_type='t2d_no_pump', **case['kwargs'])
        patient = value.patient_params if hasattr(value, 'patient_params') else value
        assert patient.kabs == 0.071
        assert patient.basal == 0
        assert not patient.use_pump


@pytest.mark.parametrize('field,value', [
    ('exception', 'RuntimeError'), ('message', 'anything'),
    ('types', ['t1d']), ('origin', 'loader'),
])
def test_unknown_rejection_declarations_fail(manifest, field, value):
    subset = rejection_manifest(manifest)
    subset['nondefault']['factory_cases']['final_override']['expected_rejection'][field] = value
    with pytest.raises(ValueError, match='Unsupported expected_rejection'):
        harness.capture_nondefault(subset)


def test_rejection_failure_restores_acceptance(monkeypatch, manifest):
    subset = rejection_manifest(manifest)
    subset['nondefault']['factory_cases']['final_override']['acceptance'] = 0.27
    original = harness.patient_factory.ACCEPTANCE_PROB_DEFAULT

    def succeeding(*args, **kwargs):
        assert harness.patient_factory.ACCEPTANCE_PROB_DEFAULT == 0.27
        return object()

    monkeypatch.setattr(harness, 'create_env_params', succeeding)
    with pytest.raises(AssertionError, match='construction succeeded'):
        harness.capture_nondefault(subset)
    assert harness.patient_factory.ACCEPTANCE_PROB_DEFAULT == original


@pytest.mark.parametrize('calibrator', ['calibrate_t2d_pump', 'calibrate_t2d_no_pump'])
def test_same_message_calibration_failure_is_not_expected_rejection(monkeypatch, manifest, calibrator):
    from glucosim.simglucose.core import parameter_builder

    def fail(*args, **kwargs):
        raise ValueError('use_pump=False requires basal=0')

    monkeypatch.setattr(parameter_builder.calibration, calibrator, fail)
    with pytest.raises(AssertionError, match='did not match'):
        harness.capture_nondefault(rejection_manifest(manifest))


def test_serial_cli_comparison_failure_does_not_publish_output(monkeypatch, tmp_path):
    """A failed comparison cannot leave an apparently successful capture file."""
    output = tmp_path / 'failed-capture.json'
    reference = tmp_path / 'reference.json'
    reference.write_text(json.dumps({'runtime': {}}))
    monkeypatch.setattr(harness.sys, 'argv', [
        'characterize_simulator.py', '--output', str(output), '--compare', str(reference)])
    monkeypatch.setattr(harness.jax, 'default_backend', lambda: 'cpu')
    # Keep the CLI's provenance and comparison checks real, but avoid simulation.
    monkeypatch.setattr(harness.sharding, 'job_records', lambda *args: {'fixture': ['record']})
    calls = []

    def captured(*args):
        calls.append(args)
        return {'record': '0' * 64}

    monkeypatch.setattr(harness, 'capture', captured)
    with pytest.raises(ValueError, match='Cannot compare different runtime'):
        harness.main()
    assert len(calls) == 1
    assert not output.exists()


def test_v4_comparison_rejects_shared_manifest_shrink(manifest):
    subset = deepcopy(manifest)
    subset['patients']['numbers'] = [1]
    artifact = dict(runtime={}, manifest=subset, smoke=False,
                    harness_sha256='0' * 64, input_hashes={}, records={})
    with pytest.raises(ValueError, match='Manifest differs from the frozen comparison manifest'):
        harness.compare(artifact, deepcopy(artifact))
