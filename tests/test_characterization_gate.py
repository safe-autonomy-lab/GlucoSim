"""The characterization gate must reject changes outside the reset repair."""
from copy import deepcopy
import hashlib

import pytest

from examples.characterize_simulator import compare


@pytest.fixture(autouse=True)
def baseline_source(monkeypatch):
    # Unit tests exercise validation without requiring historical Git objects.
    monkeypatch.setattr('examples.characterize_simulator.legacy_source',
                        lambda revision: ('baseline-commit', {'glucosim/example.py': 'baseline-hash'}))
    monkeypatch.setattr('examples.characterize_simulator.subprocess.check_output', lambda *args, **kwargs: b'csv')


def captures():
    records = {'patient/ode/meal': 'unchanged'}
    for history in ('cold', 'repeat_same_seed', 'after_other_seed'):
        for field in ('reset', 'steps', 'exposed', 'effective'):
            records[f'patient/gym/5/default/42/{history}/{field}'] = 'cold'
    commit, hashes = 'baseline-commit', {'glucosim/example.py': 'baseline-hash'}
    current = dict(runtime={}, manifest={'legacy_commit': 'a5bd537'}, smoke=True, records=records,
                   harness_sha256='same-harness', source_commit=commit, source_hashes=hashes,
                   input_hashes={'glucosim/simglucose/params/vpatient_params.csv': hashlib.sha256(b'csv').hexdigest()})
    reference = deepcopy(current)
    for key in records:
        if key.endswith('/exposed') or ('/after_other_seed/' in key and not key.endswith('/effective')):
            reference['records'][key] = 'legacy'
    return reference, current


def test_reset_exceptions_are_explicit():
    reference, current = captures()
    with pytest.raises(AssertionError):
        compare(reference, current)
    assert len(compare(reference, current, allow_reset_fixes=True)) == 5


@pytest.mark.parametrize('key', [
    'patient/ode/meal',
    'patient/gym/5/default/42/cold/reset',
    'patient/gym/5/default/42/after_other_seed/steps',
])
def test_changed_dynamics_or_cache_history_fail(key):
    reference, current = captures()
    current['records'][key] = 'unexpected'
    with pytest.raises(AssertionError):
        compare(reference, current, allow_reset_fixes=True)


def test_different_runtime_fails():
    reference, current = captures()
    current['runtime']['jax'] = 'different'
    with pytest.raises(ValueError):
        compare(reference, current, allow_reset_fixes=True)


def test_consistently_wrong_getter_fails():
    reference, current = captures()
    for key in current['records']:
        if key.endswith('/exposed'):
            current['records'][key] = 'wrong-across-all-histories'
    with pytest.raises(AssertionError, match='Exposed parameters'):
        compare(reference, current, allow_reset_fixes=True)


def test_different_harness_fails():
    reference, current = captures()
    current['harness_sha256'] = 'different-harness'
    with pytest.raises(ValueError, match='harness_sha256'):
        compare(reference, current, allow_reset_fixes=True)


@pytest.mark.parametrize('field', ['source_commit', 'source_hashes'])
def test_wrong_legacy_source_fails(field):
    reference, current = captures()
    reference[field] = 'wrong' if field == 'source_commit' else {}
    with pytest.raises(ValueError, match='baseline source'):
        compare(reference, current, allow_reset_fixes=True)


def test_effective_parameters_cannot_use_mixed_seed_exception():
    reference, current = captures()
    for field in ('effective', 'exposed'):
        current['records'][f'patient/gym/5/default/42/after_other_seed/{field}'] = 'wrong'
    with pytest.raises(AssertionError, match='Unexpected changes'):
        compare(reference, current, allow_reset_fixes=True)


def test_changed_or_missing_csv_provenance_fails():
    _, current = captures()
    reference = deepcopy(current)
    reference['input_hashes'] = {}
    with pytest.raises(ValueError, match='input_hashes'):
        compare(reference, current)
    del reference['input_hashes']
    with pytest.raises(ValueError, match='recapture'):
        compare(reference, current)


def test_common_csv_drift_fails_git_verification():
    reference, current = captures()
    for artifact in (reference, current):
        artifact['input_hashes']['glucosim/simglucose/params/vpatient_params.csv'] = 'shared-drift'
    with pytest.raises(ValueError, match='CSV'):
        compare(reference, current, allow_reset_fixes=True)


def test_source_verification_uses_content_not_capture_head():
    from examples.characterize_simulator import verify_source
    _, current = captures()
    current['source_commit'] = 'head-before-final-commit'
    verify_source(current, 'final-commit')
    current['source_hashes'] = {}
    with pytest.raises(ValueError, match='source hashes'):
        verify_source(current, 'final-commit')


def test_nondefault_records_have_no_repair_exception():
    reference, current = captures()
    key = 'nondefault/adolescent#001/t1d/factory/carb_scale/tuned'
    reference['records'][key] = 'before'
    current['records'][key] = 'after'
    with pytest.raises(AssertionError, match='Unexpected changes'):
        compare(reference, current, allow_reset_fixes=True)
