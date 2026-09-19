"""Sharding integrity tests use synthetic digests, never simulator rollouts."""
from copy import deepcopy
import hashlib
import json

import pytest

from examples import characterization_shards as shards


@pytest.fixture
def manifest():
    return json.loads((shards.ROOT / 'tests/characterization_manifest.json').read_text())


def synthetic_records(manifest, keys):
    from examples.characterization_contract import expected_rejection, factory_record_stages, rejection_digest
    records = {key: hashlib.sha256(key.encode()).hexdigest() for key in keys}
    for kind in manifest['types']:
        for name, case in manifest['nondefault']['factory_cases'].items():
            spec = expected_rejection(case, kind)
            if spec is None:
                continue
            for stage in factory_record_stages(case, kind):
                key = f'nondefault/{manifest["nondefault"]["patient"]}/{kind}/factory/{name}/{stage}'
                if key in records:
                    records[key] = rejection_digest(stage, spec)
    return records


def make_shards(manifest, count=3, smoke=False):
    jobs = shards.job_records(manifest, smoke)
    files = shards.harness_files()
    common = dict(runtime={'backend': 'cpu', 'x64': False}, manifest=manifest, smoke=smoke,
                  harness_files=files, harness_sha256=shards.harness_digest(files),
                  input_hashes={'patients.csv': 'input'}, source_hashes={'source.py': 'source'},
                  source_commit='test-commit', execution_settings={'cpu_model': 'test-cpu'})
    output = []
    for index in range(count):
        members = shards.membership(manifest, smoke, index, count)
        artifact = deepcopy(common)
        artifact['shard'] = dict(schema=shards.SCHEMA, status='complete', index=index,
                                 count=count, jobs=members)
        artifact['records'] = synthetic_records(manifest, [key for job in members for key in jobs[job]])
        output.append(artifact)
    return output


def test_full_manifest_has_exact_unique_coverage(manifest):
    jobs = shards.job_records(manifest)
    keys = [key for group in jobs.values() for key in group]
    assert len(keys) == len(set(keys)) == 6273
    assert len(jobs) == 13
    assert len(jobs['nondefault']) == 99


@pytest.mark.parametrize('smoke', [False, True])
def test_membership_covers_each_logical_job_once_for_every_worker_count(manifest, smoke):
    jobs = shards.job_records(manifest, smoke)
    original = deepcopy(manifest)
    for count in range(1, len(jobs) + 1):
        assigned = [job for index in reversed(range(count))
                    for job in shards.membership(manifest, smoke, index, count)]
        assert len(assigned) == len(set(assigned)) == len(jobs)
        assert set(assigned) == set(jobs)
    assert manifest == original


@pytest.mark.parametrize('smoke', [False, True])
def test_gym_jobs_keep_all_reset_histories_and_seeds_together(manifest, smoke):
    jobs = shards.job_records(manifest, smoke)
    seeds = manifest['smoke']['seeds'] if smoke else manifest['seeds']
    for job, keys in jobs.items():
        if not job.startswith('gym/'):
            continue
        assert {int(key.split('/')[5]) for key in keys} == set(seeds)
        for key in keys:
            parts = key.split('/')
            assert '/'.join(parts[:2]) == job.removeprefix('gym/')
            for history in manifest['gym']['cache_histories']:
                assert '/'.join(parts[:6] + [history, parts[7]]) in keys


@pytest.mark.parametrize('index,count', [(-1, 2), (2, 2), (0, 0), (0, 14),
                                         (True, 2), (0, True), (0.0, 2), (0, 2.0), (None, 2)])
def test_invalid_shard_coordinates_rejected(manifest, index, count):
    with pytest.raises(ValueError, match='index/count'):
        shards.membership(manifest, False, index, count)


def test_duplicate_manifest_cases_rejected(manifest):
    manifest['seeds'].append(manifest['seeds'][0])
    with pytest.raises(ValueError, match='duplicate logical'):
        shards.job_records(manifest)


@pytest.mark.parametrize('smoke', [False, True])
def test_merge_is_exact_order_independent_and_preserves_inputs(manifest, smoke):
    artifacts = make_shards(manifest, smoke=smoke)
    before = deepcopy(artifacts)
    merged = shards.merge(list(reversed(artifacts)))
    expected = synthetic_records(manifest, [key for keys in shards.job_records(manifest, smoke).values() for key in keys])
    assert merged['records'] == expected
    assert merged == shards.merge(artifacts)
    assert artifacts == before
    assert [receipt['index'] for receipt in merged['merged_shards']] == [0, 1, 2]
    merged['manifest']['seeds'].append(100)
    assert artifacts == before


def test_missing_shards_rejected(manifest):
    with pytest.raises(ValueError, match='Missing shards'):
        shards.merge([])
    with pytest.raises(ValueError, match='Missing shards'):
        shards.merge(make_shards(manifest)[:-1])


def test_duplicate_shard_rejected_even_when_identical(manifest):
    artifacts = make_shards(manifest)
    with pytest.raises(ValueError, match='Duplicate shard'):
        shards.merge(artifacts + [deepcopy(artifacts[0])])


@pytest.mark.parametrize('field', shards.PROVENANCE)
@pytest.mark.parametrize('operation', ['missing', 'different'])
def test_missing_or_incompatible_provenance_rejected(manifest, field, operation):
    artifacts = make_shards(manifest)
    if operation == 'missing':
        del artifacts[1][field]
    else:
        artifacts[1][field] = 'different'
    with pytest.raises(ValueError, match='provenance'):
        shards.merge(artifacts)


@pytest.mark.parametrize('field', shards.PROVENANCE)
def test_first_shard_requires_all_provenance(manifest, field):
    artifacts = make_shards(manifest)
    del artifacts[0][field]
    with pytest.raises(ValueError, match='Missing provenance'):
        shards.merge(artifacts)


def test_common_foreign_harness_and_inconsistent_hash_rejected(manifest):
    artifacts = make_shards(manifest)
    for artifact in artifacts:
        artifact['harness_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='Inconsistent harness'):
        shards.merge(artifacts)
    for artifact in artifacts:
        artifact['harness_files'] = {'foreign.py': 'hash'}
        artifact['harness_sha256'] = shards.harness_digest(artifact['harness_files'])
    with pytest.raises(ValueError, match='same frozen harness'):
        shards.merge(artifacts)


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('status', 'running'),
                                         ('schema', 99), ('count', 4), ('index', 99)])
def test_failed_or_incompatible_shard_metadata_rejected(manifest, field, value):
    artifacts = make_shards(manifest)
    artifacts[1]['shard'][field] = value
    with pytest.raises(ValueError):
        shards.merge(artifacts)


@pytest.mark.parametrize('operation', ['missing', 'extra', 'duplicate', 'reordered'])
def test_forged_job_membership_rejected(manifest, operation):
    artifacts = make_shards(manifest)
    members = artifacts[0]['shard']['jobs']
    if operation == 'missing':
        members.pop()
    elif operation == 'extra':
        members.append('unknown/job')
    elif operation == 'duplicate':
        members.append(members[0])
    else:
        members.reverse()
    with pytest.raises(ValueError, match='membership'):
        shards.merge(artifacts)


@pytest.mark.parametrize('operation', ['missing', 'extra', 'cross_shard', 'missing_history'])
def test_record_coverage_is_computed_from_manifest(manifest, operation):
    artifacts = make_shards(manifest)
    records = artifacts[0]['records']
    if operation == 'missing':
        records.pop(next(iter(records)))
    elif operation == 'extra':
        records['invented/key'] = 'a' * 64
    elif operation == 'cross_shard':
        key, digest = next(iter(artifacts[1]['records'].items()))
        records[key] = digest
    else:
        key = next(key for key in records if '/after_other_seed/' in key)
        records.pop(key)
    with pytest.raises(ValueError, match='coverage'):
        shards.merge(artifacts)


@pytest.mark.parametrize('digest', ['', 'a' * 63, 'A' * 64, 'g' * 64, 42, None])
def test_malformed_digests_rejected(manifest, digest):
    artifacts = make_shards(manifest)
    artifacts[0]['records'][next(iter(artifacts[0]['records']))] = digest
    with pytest.raises(ValueError, match='digest'):
        shards.merge(artifacts)


@pytest.mark.parametrize('payload', ['{"records": {"key": "a", "key": "b"}}',
                                     '{"runtime": {}, "runtime": {}}'])
def test_duplicate_json_fields_rejected(tmp_path, payload):
    path = tmp_path / 'duplicate.json'
    path.write_text(payload)
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        shards.load_capture(path)


def test_capture_reader_roundtrip(tmp_path, manifest):
    artifact = make_shards(manifest, count=1)[0]
    path = tmp_path / 'capture.json'
    path.write_text(json.dumps(artifact))
    assert shards.load_capture(path) == artifact


def completion_fixture(tmp_path):
    paths = [tmp_path / f'shard-{index}.json' for index in range(2)]
    for index, path in enumerate(paths):
        path.write_text(json.dumps({'index': index}))
    receipts = {str(path.resolve()): {'exit_code': 0, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in paths}
    return paths, receipts


def test_completion_receipts_bind_exact_successful_artifacts(tmp_path):
    paths, receipts = completion_fixture(tmp_path)
    shards.verify_completions(list(reversed(paths)), receipts)
    paths[0].write_text('changed after exit')
    with pytest.raises(ValueError, match='hash mismatch'):
        shards.verify_completions(paths, receipts)


@pytest.mark.parametrize('exit_code', [1, -9, None, False, '0'])
def test_failed_or_unreported_exit_rejected(tmp_path, exit_code):
    paths, receipts = completion_fixture(tmp_path)
    receipts[str(paths[0].resolve())]['exit_code'] = exit_code
    with pytest.raises(ValueError, match='Failed shard process'):
        shards.verify_completions(paths, receipts)


@pytest.mark.parametrize('operation', ['missing', 'extra', 'duplicate_path', 'aliased_path'])
def test_completion_receipts_require_exact_path_coverage(tmp_path, operation):
    paths, receipts = completion_fixture(tmp_path)
    if operation == 'missing':
        receipts.pop(str(paths[0].resolve()))
    elif operation == 'extra':
        receipts[str(tmp_path / 'foreign.json')] = {'exit_code': 0, 'sha256': 'a' * 64}
    elif operation == 'duplicate_path':
        paths.append(paths[0])
    else:
        paths.append(tmp_path / 'unused' / '..' / paths[0].name)
    with pytest.raises(ValueError, match='exactly once'):
        shards.verify_completions(paths, receipts)


@pytest.fixture
def migration_captures(manifest, monkeypatch):
    from examples import characterize_simulator as harness
    current = shards.merge(make_shards(manifest))
    # Preserve the original v3 serial-to-sharding migration fixture.
    manifest = deepcopy(manifest)
    manifest['version'] = 3
    cases = manifest['nondefault']['factory_cases']
    cases['final_override'].pop('expected_rejection')
    cases.pop('final_override_no_pump_valid')
    current['manifest'] = deepcopy(manifest)
    current['records'] = synthetic_records(manifest,
        [key for group in shards.job_records(manifest).values() for key in group])
    current['source_commit'] = shards.MIGRATION_COMMIT
    reference = deepcopy(current)
    del reference['merged_shards']
    del reference['harness_files']
    del reference['execution_settings']
    reference['harness_sha256'] = hashlib.sha256(b'pinned-serial').hexdigest()

    def git_read(command, **kwargs):
        if command[-1].endswith(':tests/characterization_manifest.json'):
            return json.dumps(manifest).encode()
        assert command[-1] == f'{shards.MIGRATION_COMMIT}:examples/characterize_simulator.py'
        return b'pinned-serial'

    def verify_source(capture, revision):
        assert revision == shards.MIGRATION_COMMIT
        if capture['source_hashes'] != {'source.py': 'source'}:
            raise ValueError('Source drift')
        if capture['input_hashes'] != {'patients.csv': 'input'}:
            raise ValueError('Input drift')

    monkeypatch.setattr(shards.subprocess, 'check_output', git_read)
    monkeypatch.setattr(harness, 'verify_source', verify_source)
    receipt = dict(exit_code=0, capture_digest=hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
                   execution_settings=deepcopy(current['execution_settings']))
    return reference, current, receipt


def test_migration_is_explicit_exact_and_does_not_mutate_captures(migration_captures):
    from examples.characterize_simulator import compare
    reference, current, receipt = migration_captures
    before = deepcopy(migration_captures)
    with pytest.raises(ValueError, match='harness'):
        compare(reference, current)
    result = shards.verify_serial_migration(reference, current, receipt)
    assert result['records'] == 6271
    assert result['source_commit'] == shards.MIGRATION_COMMIT
    assert migration_captures == before


@pytest.mark.parametrize('side,field,value', [
    (0, 'manifest', {}), (1, 'manifest', {}),
    (0, 'harness_sha256', 'foreign'), (1, 'harness_sha256', 'foreign'),
    (1, 'harness_files', {}),
    (0, 'source_commit', 'foreign'), (1, 'source_commit', 'foreign'),
    (0, 'source_hashes', {}), (1, 'source_hashes', {}),
    (0, 'input_hashes', {}), (1, 'input_hashes', {}),
    (0, 'smoke', True), (1, 'smoke', True), (1, 'merged_shards', []),
    (1, 'runtime', {'backend': 'gpu'}), (1, 'execution_settings', {}),
    (2, 'capture_digest', 'foreign'), (2, 'execution_settings', {}),
])
def test_migration_cannot_relax_other_provenance(migration_captures, side, field, value):
    migration_captures[side][field] = value
    with pytest.raises(ValueError):
        shards.verify_serial_migration(*migration_captures)


@pytest.mark.parametrize('operation', ['changed_value', 'missing_key', 'extra_key'])
def test_migration_requires_exact_record_equivalence(migration_captures, operation):
    records = migration_captures[1]['records']
    if operation == 'changed_value':
        key = next(key for key in records if '/after_other_seed/' in key and key.endswith('/steps'))
        records[key] = 'f' * 64
    elif operation == 'missing_key':
        records.pop(next(iter(records)))
    else:
        records['unexpected'] = 'f' * 64
    with pytest.raises((ValueError, AssertionError)):
        shards.verify_serial_migration(*migration_captures)


def test_merger_cli_never_overwrites_existing_output(tmp_path, monkeypatch):
    import sys
    output = tmp_path / 'existing.json'
    output.write_text('preserve this artifact')
    monkeypatch.setattr(sys, 'argv', ['characterization_shards', 'unused-shard.json',
                                    '--output', str(output), '--completion-receipts', 'unused-receipts.json'])
    with pytest.raises(SystemExit) as error:
        shards.main()
    assert error.value.code == 2
    assert output.read_text() == 'preserve this artifact'


def test_common_manifest_shrink_cannot_forge_complete_coverage(manifest):
    manifest['patients']['numbers'] = [1]
    artifacts = make_shards(manifest)
    with pytest.raises(ValueError, match='manifest'):
        shards.merge(artifacts)


@pytest.mark.parametrize('smoke', [0, 1, 'false', None])
def test_smoke_provenance_requires_boolean(manifest, smoke):
    artifacts = make_shards(manifest)
    for artifact in artifacts:
        artifact['smoke'] = smoke
    with pytest.raises(ValueError):
        shards.merge(artifacts)


@pytest.mark.parametrize('schema', [True, 1.0, '1'])
def test_schema_provenance_requires_integer(manifest, schema):
    artifacts = make_shards(manifest)
    for artifact in artifacts:
        artifact['shard']['schema'] = schema
    with pytest.raises(ValueError):
        shards.merge(artifacts)


@pytest.mark.parametrize('exit_code', [None, 1, -9, False, True, '0', 0.0])
def test_migration_rejects_failed_or_malformed_serial_exit(migration_captures, exit_code):
    migration_captures[2]['exit_code'] = exit_code
    with pytest.raises(ValueError, match='completion status'):
        shards.verify_serial_migration(*migration_captures)


def test_migration_requires_serial_exit_status(migration_captures):
    del migration_captures[2]['exit_code']
    with pytest.raises(ValueError, match='completion status'):
        shards.verify_serial_migration(*migration_captures)


@pytest.mark.parametrize('replacement', ['overwrite', 'symlink_retarget'])
def test_merger_parses_and_reports_exact_verified_snapshot(tmp_path, monkeypatch, manifest, replacement):
    import sys
    artifact = make_shards(manifest, count=1)[0]
    original_bytes = json.dumps(artifact).encode()
    original = tmp_path / 'original.json'
    original.write_bytes(original_bytes)
    shard_path = original
    if replacement == 'symlink_retarget':
        shard_path = tmp_path / 'shard-link.json'
        shard_path.symlink_to(original)
    verified_path = str(original.resolve())
    verified_hash = hashlib.sha256(original_bytes).hexdigest()
    receipts = {verified_path: {'exit_code': 0, 'sha256': verified_hash}}
    receipt_path = tmp_path / 'receipts.json'
    receipt_path.write_text(json.dumps(receipts))
    output = tmp_path / 'merged.json'
    verify_completions = shards.verify_completions

    def verify_then_replace(paths, completion_receipts):
        snapshots = verify_completions(paths, completion_receipts)
        assert snapshots == {verified_path: original_bytes}
        if replacement == 'overwrite':
            original.write_text('this replacement is deliberately invalid JSON')
        else:
            replacement_path = tmp_path / 'unverified.json'
            replacement_path.write_text('this replacement is deliberately invalid JSON')
            shard_path.unlink()
            shard_path.symlink_to(replacement_path)
        return snapshots

    monkeypatch.setattr(shards, 'verify_completions', verify_then_replace)
    monkeypatch.setattr(sys, 'argv', ['characterization_shards', str(shard_path),
                                    '--output', str(output), '--completion-receipts', str(receipt_path)])
    shards.main()
    merged = shards.load_capture(output)
    assert merged['records'] == artifact['records']
    assert merged['completion_receipts'] == receipts
    assert merged['shard_artifacts'] == {verified_path: verified_hash}


@pytest.mark.parametrize('field,value,changed_key_fragment', [
    ('seed', 0.0, '/ode/default/0.0/'),
    ('seed', False, '/ode/default/False/'),
    ('sample_time', 1.0, '/gym/1.0/'),
])
def test_common_manifest_type_coercion_cannot_change_record_keys(
        manifest, field, value, changed_key_fragment):
    original = deepcopy(manifest)
    if field == 'seed':
        manifest['seeds'][0] = value
    else:
        manifest['gym']['sample_time_min'][0] = value
    # This is the failure mechanism: ordinary Python equality overlooks the
    # type change, but the formatted record keys no longer describe the manifest.
    assert manifest == original
    artifacts = make_shards(manifest)
    assert any(changed_key_fragment in key for artifact in artifacts for key in artifact['records'])
    original_keys = {key for keys in shards.job_records(original).values() for key in keys}
    altered_keys = {key for artifact in artifacts for key in artifact['records']}
    assert altered_keys != original_keys
    with pytest.raises(ValueError, match='[Mm]anifest'):
        shards.merge(artifacts)


@pytest.mark.parametrize('field,value', [('seed', 0.0), ('seed', False), ('sample_time', 1.0)])
def test_migration_pinned_manifest_rejects_common_type_coercion(migration_captures, field, value):
    reference, current, receipt = migration_captures
    original_manifest = deepcopy(current['manifest'])
    for artifact in (reference, current):
        if field == 'seed':
            artifact['manifest']['seeds'][0] = value
        else:
            artifact['manifest']['gym']['sample_time_min'][0] = value
        artifact['records'] = {key: hashlib.sha256(key.encode()).hexdigest()
                               for keys in shards.job_records(artifact['manifest']).values() for key in keys}
        assert artifact['manifest'] == original_manifest
    receipt['capture_digest'] = hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()
    assert reference['records'] == current['records']
    with pytest.raises(ValueError, match='pinned manifest'):
        shards.verify_serial_migration(reference, current, receipt)


@pytest.mark.parametrize('field,value', [('smoke', 0), ('runtime', {'backend': 'cpu', 'x64': 0})])
def test_later_shard_provenance_cannot_coerce_boolean_values(manifest, field, value):
    artifacts = make_shards(manifest)
    assert artifacts[1][field] == value
    artifacts[1][field] = value
    with pytest.raises(ValueError, match='provenance'):
        shards.merge(artifacts)


def test_later_shard_count_requires_integer(manifest):
    artifacts = make_shards(manifest)
    artifacts[1]['shard']['count'] = 3.0
    with pytest.raises(ValueError):
        shards.merge(artifacts)


@pytest.mark.parametrize('field,value', [('smoke', 0), ('runtime', {'backend': 'cpu', 'x64': 0})])
def test_serial_comparison_preserves_provenance_types(manifest, field, value):
    from examples.characterize_simulator import compare
    reference = shards.merge(make_shards(manifest))
    current = deepcopy(reference)
    assert current[field] == value
    current[field] = value
    with pytest.raises(ValueError, match=field):
        compare(reference, current)


def test_serial_comparison_preserves_manifest_number_types(manifest):
    from examples.characterize_simulator import compare
    reference = shards.merge(make_shards(manifest))
    current = deepcopy(reference)
    current['manifest']['seeds'][0] = 0.0
    assert current['manifest'] == reference['manifest']
    with pytest.raises(ValueError, match='manifest'):
        compare(reference, current)


def test_migration_execution_receipt_preserves_value_types(migration_captures):
    reference, current, receipt = migration_captures
    current['execution_settings']['future_boolean_setting'] = False
    receipt['execution_settings']['future_boolean_setting'] = 0
    assert current['execution_settings'] == receipt['execution_settings']
    with pytest.raises(ValueError, match='execution settings'):
        shards.verify_serial_migration(reference, current, receipt)


@pytest.mark.parametrize('smoke', [False, True])
def test_merge_rejects_forged_rejection_result(manifest, smoke):
    artifacts = make_shards(manifest, smoke=smoke)
    key = next(k for k in artifacts[0]['records'] if k.endswith('/patient_rejection'))
    artifacts[0]['records'][key] = 'f' * 64
    with pytest.raises(ValueError, match='expected-rejection digest'):
        shards.merge(artifacts)


def test_v4_comparison_rejects_common_missing_results(manifest):
    from examples.characterize_simulator import compare
    full = shards.merge(make_shards(manifest))
    partial = deepcopy(full)
    partial['records'].pop(next(iter(partial['records'])))
    with pytest.raises(ValueError, match='coverage'):
        compare(partial, deepcopy(partial))


def test_v4_comparison_rejects_common_forged_rejection(manifest):
    from examples.characterize_simulator import compare
    full = shards.merge(make_shards(manifest))
    key = next(k for k in full['records'] if k.endswith('/created_rejection'))
    full['records'][key] = '0' * 64
    with pytest.raises(ValueError, match='expected-rejection digest'):
        compare(full, deepcopy(full))
