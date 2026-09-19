"""Fail-closed tests of the single declared v3/v4 bridge (no trajectories)."""
from copy import deepcopy
import hashlib
import json

import pytest

from examples import characterization_migration as migration
from examples import characterization_shards as shards
from examples import characterize_simulator as harness


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    from examples.characterization_contract import rejection_digest
    old_manifest, new_manifest = migration.expected_manifest()
    old_files = {name: hashlib.sha256(migration.git_bytes(migration.NEW_SOURCE, name)).hexdigest()
                 for name in migration.OLD_HARNESS}
    files = shards.harness_files()
    shared = dict(runtime={'backend': 'cpu'}, execution_settings={'cpu': 'test'},
                  input_hashes={'csv': 'test'}, source_hashes={'source': 'test'}, smoke=False)
    old = dict(deepcopy(shared), manifest=old_manifest, source_commit=migration.OLD_SOURCE,
               harness_files=old_files, harness_sha256=shards.harness_digest(old_files),
               records={key: 'a' * 64 for keys in shards.job_records(old_manifest).values() for key in keys})
    new = dict(deepcopy(shared), manifest=new_manifest, source_commit=migration.NEW_SOURCE,
               harness_files=files, harness_sha256=shards.harness_digest(files),
               records={key: 'a' * 64 for keys in shards.job_records(new_manifest).values() for key in keys})
    for key in new['records']:
        if key.endswith('_rejection'):
            new['records'][key] = rejection_digest(key.rsplit('/', 1)[1], migration.SPEC)
    replay = {key: value for key, value in new['records'].items() if '/final_override_no_pump_valid/' in key}
    monkeypatch.setattr(migration, 'runtime', lambda: shared['runtime'])
    monkeypatch.setattr(shards, 'execution_settings', lambda: shared['execution_settings'])
    monkeypatch.setattr(migration, 'live_sources', lambda: shared['source_hashes'])
    monkeypatch.setattr(harness, 'input_hashes', lambda manifest: shared['input_hashes'])
    def verify_source(capture, revision):
        if capture['source_hashes'] != shared['source_hashes']:
            raise ValueError('Source hash differs')
    monkeypatch.setattr(harness, 'verify_source', verify_source)
    monkeypatch.setattr(migration, 'replay_counterpart', lambda manifest: replay)
    old_path, new_path = tmp_path / 'old.json', tmp_path / 'new.json'
    old_path.write_text(json.dumps(old))
    monkeypatch.setattr(migration, 'OLD_BASELINE_SHA256', hashlib.sha256(old_path.read_bytes()).hexdigest())
    def run(receipt_change=None):
        new_path.write_text(json.dumps(new))
        receipt = {'exit_code': 0, 'sha256': hashlib.sha256(new_path.read_bytes()).hexdigest(),
                   'capture_kind': 'serial'}
        receipt.update(receipt_change or {})
        return migration.verify_migration(old_path, new_path, receipt)
    return old, new, run, old_path


def test_exact_accounting(bridge):
    _, _, run, _ = bridge
    result = run()
    assert result['unaffected_exact_records'] == 6268
    assert len(result['removed_records']) == 3
    assert len(result['expected_rejection_records']) == 2
    assert len(result['added_bounded_replay_records']) == 3


@pytest.mark.parametrize('field,value', [
    ('runtime', {'backend': 'gpu'}), ('execution_settings', {}),
    ('source_commit', migration.OLD_SOURCE), ('source_hashes', {}),
    ('input_hashes', {}), ('harness_files', {}), ('harness_sha256', '0' * 64),
    ('smoke', True), ('shard', {'status': 'complete'})])
def test_incompatible_provenance(bridge, field, value):
    _, new, run, _ = bridge
    new[field] = value
    with pytest.raises(ValueError):
        run()


@pytest.mark.parametrize('receipt', [{'exit_code': 1}, {'exit_code': False},
                                    {'exit_code': None}, {'sha256': 'bad'}])
def test_failed_process_or_unbound_receipt(bridge, receipt):
    with pytest.raises(ValueError):
        bridge[2](receipt)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'changed', 'rejection', 'added'])
def test_record_failures(bridge, mutation):
    old, new, run, _ = bridge
    common = next(key for key in old['records'] if key in new['records'])
    if mutation == 'missing':
        del new['records'][common]
    elif mutation == 'extra':
        new['records']['unexpected'] = 'b' * 64
    elif mutation == 'changed':
        new['records'][common] = 'b' * 64
    elif mutation == 'rejection':
        key = next(key for key in new['records'] if key.endswith('_rejection'))
        new['records'][key] = 'b' * 64
    else:
        key = next(key for key in new['records'] if '/final_override_no_pump_valid/' in key)
        new['records'][key] = 'b' * 64
    with pytest.raises((ValueError, AssertionError)):
        run()


@pytest.mark.parametrize('mutation', ['case', 'expectation', 'version'])
def test_manifest_drift(bridge, mutation):
    _, new, run, _ = bridge
    if mutation == 'case':
        new['manifest']['nondefault']['factory_cases'][migration.VALID_CASE]['kwargs']['basal'] = 0.1
    elif mutation == 'expectation':
        new['manifest']['nondefault']['factory_cases'][migration.CASE]['expected_rejection']['message'] = 'other'
    else:
        new['manifest']['version'] = 5
    with pytest.raises(ValueError, match='Manifest'):
        run()


def test_original_baseline_bytes_are_pinned(bridge):
    _, _, run, old_path = bridge
    old_path.write_text(old_path.read_text() + '\n')
    with pytest.raises(ValueError, match='pinned untouched'):
        run()


def test_duplicate_json_keys_rejected(bridge):
    _, _, _, old_path = bridge
    # JSON loading itself retains the old fail-closed duplicate-key safeguard.
    with pytest.raises(ValueError, match='Duplicate'):
        json.loads('{"records":{},"records":{}}', object_pairs_hook=shards.reject_duplicate_pairs)


def test_replay_must_use_capture_runtime(bridge, monkeypatch):
    monkeypatch.setattr(migration, 'runtime', lambda: {'backend': 'cpu', 'jax': 'different'})
    with pytest.raises(ValueError, match='capture CPU runtime'):
        bridge[2]()


def test_replay_must_use_capture_source(bridge, monkeypatch):
    monkeypatch.setattr(migration, 'live_sources', lambda: {'changed.py': '0' * 64})
    with pytest.raises(ValueError, match='frozen capture source'):
        bridge[2]()


def test_changes_during_replay_are_rejected(bridge, monkeypatch):
    _, new, run, _ = bridge
    def replay(manifest):
        monkeypatch.setattr(migration, 'live_sources', lambda: {'changed.py': '0' * 64})
        return {key: value for key, value in new['records'].items()
                if '/final_override_no_pump_valid/' in key}
    monkeypatch.setattr(migration, 'replay_counterpart', replay)
    with pytest.raises(ValueError, match='changed during verification'):
        run()


def test_harness_change_between_validation_and_replay(bridge, monkeypatch):
    # A -> B -> B used to pass when the verifier sampled a second starting point.
    original = shards.harness_files()
    changed = dict(original, **{'examples/characterization_migration.py': '0' * 64})
    snapshots = iter([original, changed, changed])
    monkeypatch.setattr(shards, 'harness_files', lambda: next(snapshots))
    with pytest.raises(ValueError, match='changed during verification'):
        bridge[2]()


@pytest.fixture
def merged_bridge(bridge, tmp_path):
    _, new, run, _ = bridge
    jobs = shards.job_records(new['manifest'])
    workers = []
    artifacts, receipts = {}, {}
    for index in range(2):
        names = shards.membership(new['manifest'], False, index, 2)
        worker = {field: deepcopy(new[field]) for field in shards.PROVENANCE}
        worker['records'] = {key: new['records'][key] for name in names for key in jobs[name]}
        worker['shard'] = {'schema': shards.SCHEMA, 'status': 'complete',
                           'index': index, 'count': 2, 'jobs': names}
        path = tmp_path / f'worker-{index}.json'
        workers.append((path, worker))
    def publish():
        artifacts.clear()
        receipts.clear()
        for path, worker in workers:
            path.write_text(json.dumps(worker))
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            artifacts[str(path)] = checksum
            receipts[str(path)] = {'exit_code': 0, 'sha256': checksum}
        new['shard_artifacts'] = deepcopy(artifacts)
        new['completion_receipts'] = deepcopy(receipts)
    publish()
    rebuilt = shards.merge([worker for _, worker in workers])
    new.update(rebuilt)
    def run_merged(receipt_change=None):
        return run({'capture_kind': 'merged', **(receipt_change or {})})
    return new, run_merged, workers, publish


def test_valid_merged_capture_reconstructed(merged_bridge):
    assert merged_bridge[1]()['unaffected_exact_records'] == 6268


@pytest.mark.parametrize('failure', ['failed_exit', 'missing_receipt', 'failed_status',
                                     'missing_worker', 'metadata', 'hash', 'records'])
def test_merged_worker_failures_despite_successful_outer_process(merged_bridge, failure):
    new, run, workers, publish = merged_bridge
    path, worker = workers[0]
    if failure == 'failed_status':
        worker['shard']['status'] = 'failed'
        publish()
    elif failure == 'failed_exit':
        new['completion_receipts'][str(path)]['exit_code'] = 17
    elif failure == 'missing_receipt':
        del new['completion_receipts'][str(path)]
    elif failure == 'missing_worker':
        del new['completion_receipts'][str(path)]
        del new['shard_artifacts'][str(path)]
    elif failure == 'metadata':
        new['merged_shards'][0]['status'] = 'failed'
    elif failure == 'hash':
        new['shard_artifacts'][str(path)] = '0' * 64
    else:
        key = next(iter(worker['records']))
        worker['records'][key] = 'b' * 64
        publish()
    # run() always supplies a valid outer receipt with integer exit_code == 0.
    with pytest.raises(ValueError):
        run()


@pytest.mark.parametrize('field', ['merged_shards', 'shard_artifacts', 'completion_receipts', 'shard_resources'])
def test_partial_merge_evidence_rejected(merged_bridge, field):
    new, run, _, _ = merged_bridge
    del new[field]
    with pytest.raises(ValueError, match='complete worker evidence'):
        run()


@pytest.mark.parametrize('keep_resources', [True, False])
def test_stripped_merge_evidence_still_requires_workers(merged_bridge, keep_resources):
    new, run, workers, _ = merged_bridge
    workers[0][0].unlink()  # Missing worker must not become a serial capture.
    for field in ('merged_shards', 'shard_artifacts', 'completion_receipts'):
        del new[field]
    if not keep_resources:
        del new['shard_resources']
    with pytest.raises(ValueError, match='complete worker evidence'):
        run()  # Trusted launcher still declares merged, with a matching new hash.


@pytest.mark.parametrize('kind', [None, '', 'worker', False, {}, []])
def test_invalid_or_missing_capture_kind(bridge, kind):
    with pytest.raises(ValueError, match='capture_kind'):
        bridge[2]({'capture_kind': kind})


@pytest.mark.parametrize('field', ['merged_shards', 'shard_artifacts', 'completion_receipts', 'shard_resources'])
def test_serial_receipt_rejects_each_merge_marker(bridge, field):
    _, new, run, _ = bridge
    new[field] = [] if field == 'merged_shards' else {}
    with pytest.raises(ValueError, match='Serial capture contains'):
        run()


def test_receipt_without_capture_kind_rejected(bridge):
    _, _, run, old_path = bridge
    run()
    new_path = old_path.with_name('new.json')
    receipt = {'exit_code': 0, 'sha256': hashlib.sha256(new_path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match='capture_kind'):
        migration.verify_migration(old_path, new_path, receipt)
