"""Ensure the small differential matrix can expose lost public factory inputs."""
from copy import deepcopy
import csv
import io
import json

import pytest

from examples import characterize_simulator as harness


@pytest.fixture(scope='module')
def manifest():
    return json.loads(harness.MANIFEST.read_text())


@pytest.fixture(scope='module')
def constructible_manifest(manifest):
    # The frozen historical manifest deliberately retains its now-invalid
    # no-pump + basal=.35 case. Test that rejection separately below.
    subset = deepcopy(manifest)
    subset['nondefault']['factory_cases']['final_override']['types'] = ['t1d', 't2d']
    return subset


@pytest.fixture(scope='module')
def records(constructible_manifest):
    return harness.capture_nondefault(constructible_manifest)


def test_matrix_keys_and_distinguishable_factory_cases(constructible_manifest, records):
    manifest = constructible_manifest
    expected = set()
    for kind in manifest['types']:
        prefix = f'nondefault/adolescent#001/{kind}/factory/'
        for name, case in manifest['nondefault']['factory_cases'].items():
            if kind not in case.get('types', manifest['types']):
                continue
            expected.update(prefix + name + '/' + stage for stage in ('patient', 'created', 'tuned'))
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
    with pytest.raises(ValueError, match='use_pump=False requires basal=0'):
        harness.capture_nondefault(subset)


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
