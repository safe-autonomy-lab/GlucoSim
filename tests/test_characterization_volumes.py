"""Fail-closed tests for the bounded field-level volume transition."""
from copy import deepcopy
import dataclasses
import json

import pytest

from examples import characterization_volumes as v


def patient_pair():
    old = {'dataclass': v.PATIENT, 'fields': {'BW': 100., 'Vg': 2., 'Vi': .05, 'gain': 3.,
            'V_G': 15., 'V_I': 4., 'V_G_L': 15., 'V_I_L': 4.},
           'volumes': {'V_G_L': 15., 'V_I_L': 4.},
           'expected_volumes': {'V_G_L': 20., 'V_I_L': 5.}}
    new = deepcopy(old)
    for name in v.REMOVED:
        del new['fields'][name]
    new['volumes'] = deepcopy(new['expected_volumes'])
    return old, new


def test_declared_removal_and_properties_only():
    old, new = patient_pair()
    assert v.compare_fields(old, new) == {'patients': 1, 'removed_fields': 4, 'verified_properties': 2}


@pytest.mark.parametrize('change', ['gain', 'BW', 'remove', 'add', 'property', 'stored', 'wrong_class'])
def test_parameter_drift_rejected(change):
    old, new = patient_pair()
    if change in ('gain', 'BW'):
        new['fields'][change] += 1
    elif change == 'remove':
        del new['fields']['gain']
    elif change == 'add':
        new['fields']['unrelated'] = 1
    elif change == 'property':
        new['volumes']['V_G_L'] = 21.
    elif change == 'stored':
        new['fields']['V_I'] = 5.
    else:
        new['dataclass'] = 'OtherPatient'
    with pytest.raises(AssertionError):
        v.compare_fields(old, new)


@pytest.mark.parametrize('leaf', ['physical_state', 'observation', 'reward', 'terminated', 'random_key'])
def test_mixed_record_keeps_every_physical_leaf(leaf):
    old, new = patient_pair()
    a = {'patient': old, leaf: {'dtype': '<f4', 'shape': [2], 'bytes_sha256': 'a' * 64}}
    b = {'patient': new, leaf: {'dtype': '<f4', 'shape': [2], 'bytes_sha256': 'b' * 64}}
    with pytest.raises(AssertionError, match='Retained value'):
        v.compare_fields(a, b)


def test_nested_frame_digests_expand():
    import numpy as np
    encoded = v.encode([np.array([1., 2.], dtype=np.float32), 3.], {})
    result = v.encode(['a' * 64], {'a' * 64: encoded})
    assert result['items'][0] == {'nested_digest': encoded}
    changed = deepcopy(result)
    changed['items'][0]['nested_digest']['items'][0]['bytes_sha256'] = 'b' * 64
    with pytest.raises(AssertionError):
        v.compare_fields(result, changed)


def test_dataclass_uses_named_fields():
    @dataclasses.dataclass
    class Example:
        physical: float
    result = v.encode(Example(3.), {})
    assert set(result['fields']) == {'physical'}
    assert result['fields']['physical']['dtype'] == '<f8'


@pytest.mark.parametrize('receipt', [None, {}, {'exit_code': 1}, {'exit_code': False}, {'exit_code': 0, 'sha256': 'bad'}])
def test_failed_missing_or_unbound_process_rejected(tmp_path, receipt):
    path = tmp_path / 'worker.json'
    path.write_text('{}')
    receipts = {} if receipt is None else {str(path.resolve()): receipt}
    with pytest.raises(ValueError):
        v.require_receipt(path, receipts)


def test_valid_process_receipt(tmp_path):
    path = tmp_path / 'worker.json'
    path.write_text('{}')
    v.require_receipt(path, {str(path.resolve()): {'exit_code': 0, 'sha256': v.file_hash(path)}})


def test_outputs_never_overwrite(tmp_path):
    path = tmp_path / 'result.json'
    v.publish(path, {'one': 1})
    with pytest.raises(FileExistsError):
        v.publish(path, {'two': 2})
    assert json.loads(path.read_text()) == {'one': 1}


def test_duplicate_json_rejected(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"record": 1, "record": 2}')
    with pytest.raises(ValueError, match='Duplicate'):
        v.shards.load_capture(path)


def test_merge_requires_receipts_before_loading(tmp_path):
    with pytest.raises(ValueError, match='receipt'):
        v.merge([tmp_path / 'absent.json'], {})


def test_empty_merge_rejected():
    with pytest.raises(ValueError, match='Missing shards'):
        v.merge([], {})


def evidence_pair():
    manifest = v.shards.load_capture(v.shards.ROOT / 'tests/characterization_manifest.json')
    jobs = v.shards.job_records(manifest, True)
    records = {key: 'a' * 64 for keys in jobs.values() for key in keys}
    from examples.characterization_contract import expected_rejection, factory_record_stages, rejection_digest
    spec = manifest['nondefault']
    for kind in manifest['types']:
        for name, case in spec['factory_cases'].items():
            rejection = expected_rejection(case, kind)
            if rejection:
                for stage in factory_record_stages(case, kind):
                    records[f'nondefault/{spec["patient"]}/{kind}/factory/{name}/{stage}'] = rejection_digest(stage, rejection)
    source = {'glucosim/simglucose/core/params.py': 'old'}
    capture = dict(manifest=manifest, smoke=True, runtime={'cpu': True}, execution_settings={},
                   input_hashes={}, source_commit='same', source_hashes=source,
                   harness_files=v.shards.harness_files(), harness_sha256=v.shards.harness_digest(v.shards.harness_files()),
                   records=records)
    fields = {key: {'fingerprint': digest, 'payload': {'physical': 42}} for key, digest in records.items()}
    old, new = patient_pair()
    first = next(iter(fields))
    fields[first]['payload'] = old
    before = dict(schema=1, verifier_sha256=v.file_hash(v.__file__), capture=capture, fields=fields)
    after = deepcopy(before)
    after['fields'][first]['payload'] = new
    after['capture']['source_hashes'] = {next(iter(source)): 'new'}
    return before, after


def test_complete_comparison_binds_baseline_source_and_all_records():
    before, after = evidence_pair()
    result = v.compare(before, after, before['capture'], before['capture']['source_hashes'], after['capture']['source_hashes'])
    assert result['status'] == 'passed'
    assert result['records'] == 243


@pytest.mark.parametrize('mutation', ['missing', 'fingerprint', 'harness', 'manifest', 'source', 'baseline', 'unexpected_source', 'verifier'])
def test_comparison_provenance_and_coverage_fail_closed(mutation):
    before, after = evidence_pair()
    baseline = deepcopy(before['capture'])
    old_source = deepcopy(before['capture']['source_hashes'])
    new_source = deepcopy(after['capture']['source_hashes'])
    if mutation == 'missing':
        after['fields'].pop(next(iter(after['fields'])))
    elif mutation == 'fingerprint':
        next(iter(after['fields'].values()))['fingerprint'] = 'b' * 64
    elif mutation == 'harness':
        after['capture']['harness_sha256'] = 'bad'
    elif mutation == 'manifest':
        after['capture']['manifest']['version'] = 99
    elif mutation == 'source':
        after['capture']['source_hashes'] = {'wrong': 'wrong'}
    elif mutation == 'baseline':
        baseline['records'][next(iter(baseline['records']))] = 'bad'
    elif mutation == 'unexpected_source':
        old_source['glucosim/ode.py'] = 'old'
        new_source['glucosim/ode.py'] = 'new'
        before['capture']['source_hashes'] = old_source
        after['capture']['source_hashes'] = new_source
        baseline['source_hashes'] = old_source
    else:
        after['verifier_sha256'] = 'bad'
    with pytest.raises((ValueError, AssertionError)):
        v.compare(before, after, baseline, old_source, new_source)


def write_with_receipt(path, item, kind):
    v.publish(path, item)
    return {str(path.resolve()): {'exit_code': 0, 'sha256': v.file_hash(path), 'capture_kind': kind}}


@pytest.mark.parametrize('kind', [None, 'worker', 'merged'])
def test_outer_zero_receipt_cannot_reclassify_missing_worker_evidence(tmp_path, kind):
    before, _ = evidence_pair()
    # All merger/worker markers deliberately stripped; launcher kind is authoritative.
    path = tmp_path / 'stripped.json'
    receipts = write_with_receipt(path, before, kind)
    with pytest.raises(ValueError):
        v.completed_evidence(path, receipts)


def test_serial_receipt_accepts_only_serial_artifact(tmp_path):
    before, _ = evidence_pair()
    path = tmp_path / 'serial.json'
    receipts = write_with_receipt(path, before, 'serial')
    assert v.completed_evidence(path, receipts) == before


def test_merge_reconstruction_checks_original_worker_receipts(tmp_path):
    before, _ = evidence_pair()
    jobs = v.shards.membership(before['capture']['manifest'], True, 0, 1)
    before['capture']['shard'] = dict(schema=1, status='complete', count=1, index=0, jobs=jobs)
    worker = tmp_path / 'worker.json'
    receipts = write_with_receipt(worker, before, 'worker')
    merged = v.merge([worker], receipts)
    path = tmp_path / 'merged.json'
    receipts.update(write_with_receipt(path, merged, 'merged'))
    assert v.completed_evidence(path, receipts) == merged
    receipts[str(worker.resolve())]['exit_code'] = 1
    with pytest.raises(ValueError, match='Failed'):
        v.completed_evidence(path, receipts)


def test_duplicate_workers_fail_even_with_valid_receipt(tmp_path):
    before, _ = evidence_pair()
    jobs = v.shards.membership(before['capture']['manifest'], True, 0, 1)
    before['capture']['shard'] = dict(schema=1, status='complete', count=1, index=0, jobs=jobs)
    worker = tmp_path / 'worker.json'
    receipts = write_with_receipt(worker, before, 'worker')
    with pytest.raises(ValueError, match='Duplicate shard'):
        v.merge([worker, worker], receipts)


def test_runtime_drift_fails():
    before, after = evidence_pair()
    after['capture']['runtime']['cpu'] = False
    with pytest.raises(ValueError, match='runtime'):
        v.compare(before, after, before['capture'], before['capture']['source_hashes'], after['capture']['source_hashes'])


def test_verifier_change_from_a_to_b_to_b_is_detected_before_sidecar_publish(tmp_path, monkeypatch):
    import sys
    import types
    import examples
    source = tmp_path / 'hashes.json'
    source.write_text('{}')
    standard = tmp_path / 'standard.json'
    sidecar = tmp_path / 'sidecar.json'
    fake = types.SimpleNamespace(MANIFEST=tmp_path / 'tests/manifest.json', digest=lambda value: 'a' * 64,
                                 main=lambda: standard.write_text('{"records": {}}'))
    monkeypatch.setitem(sys.modules, 'examples.characterize_simulator', fake)
    monkeypatch.setattr(examples, 'characterize_simulator', fake, raising=False)
    versions = iter(['a', 'b', 'b'])
    monkeypatch.setattr(v, 'file_hash', lambda path: next(versions) if path == v.__file__ else 'artifact')
    monkeypatch.setattr(v, 'validate', lambda *args: None)
    args = types.SimpleNamespace(expected_verifier_sha256='a', evidence=str(sidecar), expected_source_json=str(source),
                                 harness_args=['--output', str(standard)])
    with pytest.raises(ValueError, match='changed during capture'):
        v.capture(args)
    assert not sidecar.exists()


@pytest.mark.parametrize('mutation', ['missing_both', 'missing_before', 'extra_property', 'extra_metadata'])
def test_property_schema_and_patient_metadata_are_exact(mutation):
    old, new = patient_pair()
    if mutation == 'missing_both':
        for node in (old, new):
            node['volumes'] = {}
            node['expected_volumes'] = {}
    elif mutation == 'missing_before':
        del old['volumes']['V_G_L']
    elif mutation == 'extra_property':
        new['volumes']['extra'] = 1
        new['expected_volumes']['extra'] = 1
    else:
        old['extra'] = 1
        new['extra'] = 2
    with pytest.raises(AssertionError):
        v.compare_fields(old, new)


@pytest.mark.parametrize('mode', ['serial', 'worker', 'baseline'])
def test_replacement_between_authentication_and_parse_cannot_change_consumed_payload(tmp_path, monkeypatch, mode):
    before, _ = evidence_pair()
    if mode == 'worker':
        jobs = v.shards.membership(before['capture']['manifest'], True, 0, 1)
        before['capture']['shard'] = dict(schema=1, status='complete', count=1, index=0, jobs=jobs)
    path = tmp_path / (mode + '.json')
    receipts = write_with_receipt(path, before, mode)
    authenticated_bytes = path.read_bytes()
    authenticated_hash = v.file_hash(path)
    replacement = deepcopy(before)
    first = next(iter(replacement['fields']))
    replacement['fields'][first]['payload']['fields']['gain'] = 999.
    replacement_bytes = json.dumps(replacement).encode()
    original_loads = json.loads
    replaced = False
    def replace_before_parse(data, *args, **kwargs):
        nonlocal replaced
        if data == authenticated_bytes:
            path.write_bytes(replacement_bytes)
            replaced = True
        return original_loads(data, *args, **kwargs)
    monkeypatch.setattr(v.json, 'loads', replace_before_parse)
    if mode == 'serial':
        consumed = v.completed_evidence(path, receipts)
    elif mode == 'worker':
        consumed = v.merge([path], receipts)
    else:
        consumed = v.authenticated_json(path, authenticated_hash)
    assert replaced
    assert path.read_bytes() == replacement_bytes
    assert consumed['fields'] == before['fields']
    assert consumed['fields'] != replacement['fields']


def test_authenticated_snapshot_still_rejects_duplicate_keys(tmp_path):
    path = tmp_path / 'duplicate.json'
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(ValueError, match='Duplicate'):
        v.authenticated_json(path, v.file_hash(path))


@pytest.mark.parametrize('form', ['separate', 'equals', 'missing', 'harness_failure'])
def test_forwarded_output_parsed_before_harness_and_digest_restored(tmp_path, monkeypatch, form):
    import sys
    import types
    import examples
    source = tmp_path / 'hashes.json'
    source.write_text('{}')
    standard = tmp_path / 'standard.json'
    sidecar = tmp_path / 'sidecar.json'
    original_digest = lambda value: 'a' * 64
    calls = []
    def main():
        calls.append(True)
        if form == 'harness_failure':
            raise RuntimeError('worker failed')
        standard.write_text('{"records": {}}')
    fake = types.SimpleNamespace(MANIFEST=tmp_path / 'tests/manifest.json', digest=original_digest, main=main)
    monkeypatch.setitem(sys.modules, 'examples.characterize_simulator', fake)
    monkeypatch.setattr(examples, 'characterize_simulator', fake, raising=False)
    monkeypatch.setattr(v, 'validate', lambda *args: None)
    forwarded = ['--output=' + str(standard)] if form == 'equals' else ['--output', str(standard)]
    if form == 'missing':
        forwarded = ['--smoke']
    args = types.SimpleNamespace(expected_verifier_sha256=v.file_hash(v.__file__), evidence=str(sidecar),
                                 expected_source_json=str(source), harness_args=['--', *forwarded])
    old_argv = sys.argv
    if form == 'missing':
        with pytest.raises(SystemExit):
            v.capture(args)
        assert not calls
    elif form == 'harness_failure':
        with pytest.raises(RuntimeError, match='worker failed'):
            v.capture(args)
        assert calls == [True]
        assert not sidecar.exists()
    else:
        v.capture(args)
        assert calls == [True]
        assert sidecar.exists()
    assert fake.digest is original_digest
    assert sys.argv is old_argv
