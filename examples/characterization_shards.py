"""Deterministic characterization jobs and fail-closed artifact merging.

Use characterize_simulator.py --shard-index I --shard-count N for workers.
Merge with python examples/characterization_shards.py --output NEW \
    --completion-receipts receipts.json shard*.json.
The launcher must wait for each process and record receipts as a JSON map:
{absolute_artifact_path: {"exit_code": 0, "sha256": SHA256_OF_FILE_BYTES}}.
The initial migration also requires --migrate-serial-reference OLD and
--serial-runtime-receipt RECEIPT. That receipt binds SHA256 of the sorted-key
JSON encoding of OLD (capture_digest) to the measured execution_settings and
a launcher-observed integer exit_code of zero after the serial process exits.
Keep the old artifact intact; the merged artifact records the validated bridge.
All outputs are exclusive creations. A worker publishes only after full coverage
validation; a failed worker has no completed artifact to contribute.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MIGRATION_COMMIT = '0b8c087a27e40c5b179106458577527fa992f9a1'
SCHEMA = 1
PROVENANCE = ('runtime', 'manifest', 'smoke', 'harness_sha256', 'harness_files',
              'input_hashes', 'source_hashes', 'source_commit', 'execution_settings')


def job_records(manifest, smoke=False):
    """Stable jobs independent of worker count; keys computed without simulation."""
    subset = manifest['smoke'] if smoke else {}
    jobs = {'nondefault': []}
    spec = manifest['nondefault']
    for kind in manifest['types']:
        for name, case in spec['factory_cases'].items():
            if kind in case.get('types', manifest['types']):
                jobs['nondefault'].extend(f'nondefault/{spec["patient"]}/{kind}/factory/{name}/{stage}'
                                          for stage in ('patient', 'created', 'tuned'))
        if kind != 't1d':
            jobs['nondefault'].extend(f'nondefault/{spec["patient"]}/{kind}/adapter/{name}'
                                      for name in spec['adapter_configs'])
    seeds = subset.get('seeds', manifest['seeds'])
    modes = subset.get('noise_modes', manifest['noise_modes'])
    for kind in manifest['types']:
        keys = jobs[f'ode/{kind}'] = []
        for cohort in manifest['patients']['cohorts']:
            for number in subset.get('patient_numbers', manifest['patients']['numbers']):
                prefix = f'{cohort}#{number:03d}/{kind}'
                keys.extend(prefix + '/' + stage for stage in ('created', 'tuned'))
                keys.extend(f'{prefix}/ode/{mode}/{seed}/{case}' for mode in modes for seed in seeds
                            for case in subset.get('ode_cases', list(manifest['ode']['cases'])))
    spec = manifest['gym']
    for cohort in manifest['patients']['cohorts']:
        for number in spec['patient_numbers']:
            for kind in manifest['types']:
                prefix = f'{cohort}#{number:03d}/{kind}'
                jobs[f'gym/{prefix}'] = [
                    f'{prefix}/gym/{sample}/{mode}/{seed}/{history}/{stage}'
                    for sample in subset.get('sample_time_min', spec['sample_time_min'])
                    for mode in modes for seed in seeds for history in spec['cache_histories']
                    for stage in ('exposed', 'effective', 'reset', 'steps')]
    flattened = [key for keys in jobs.values() for key in keys]
    if len(flattened) != len(set(flattened)):
        raise ValueError('Manifest contains duplicate logical record keys')
    return jobs


def membership(manifest, smoke, index, count):
    jobs = list(job_records(manifest, smoke))
    if type(count) is not int or type(index) is not int or not 1 <= count <= len(jobs) or not 0 <= index < count:
        raise ValueError('Invalid shard index/count')
    return jobs[index::count]


def harness_files():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ('examples/characterize_simulator.py', 'examples/characterization_shards.py')}


def harness_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def execution_settings():
    import os
    import platform
    # CPU model/ISA and numerical/thread settings must match. Affinity IDs and
    # hostnames may differ; record those as resources, not numerical provenance.
    cpu = Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else ''
    model = next((line.split(':', 1)[1].strip() for line in cpu.splitlines() if line.startswith('model name')), platform.machine())
    flags = next((line.split(':', 1)[1].strip() for line in cpu.splitlines() if line.startswith('flags')), '')
    return dict(cpu_model=model, cpu_flags=flags, environment={key: value for key, value in sorted(os.environ.items())
                if key.startswith(('JAX_', 'XLA_', 'TF_')) or key in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})


def validate_records(records, expected):
    if not isinstance(records, dict) or set(records) != set(expected):
        raise ValueError('Record coverage differs from expected manifest coverage')
    if any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value) for value in records.values()):
        raise ValueError('Invalid record digest')


def reject_duplicate_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def load_capture(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=reject_duplicate_pairs)


def exact_json(value):
    """Type-sensitive JSON representation (unlike Python's 0 == 0.0 == False)."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def merge(shards):
    if not shards:
        raise ValueError('Missing shards')
    first = shards[0]
    for field in PROVENANCE:
        if field not in first:
            raise ValueError(f'Missing provenance: {field}')
    trusted_manifest = load_capture(ROOT / 'tests/characterization_manifest.json')
    if exact_json(first['manifest']) != exact_json(trusted_manifest):
        raise ValueError('Manifest differs from the frozen merger manifest')
    if type(first['smoke']) is not bool:
        raise ValueError('Invalid smoke mode')
    if first['harness_sha256'] != harness_digest(first['harness_files']):
        raise ValueError('Inconsistent harness provenance')
    if first['harness_files'] != harness_files():
        raise ValueError('Merger must use the same frozen harness as workers')
    expected_jobs = job_records(trusted_manifest, first['smoke'])
    count = first.get('shard', {}).get('count')
    seen, records, receipts, resources = set(), {}, [], {}
    for shard in shards:
        for field in PROVENANCE:
            if field not in shard or exact_json(shard[field]) != exact_json(first[field]):
                raise ValueError(f'Incompatible shard provenance: {field}')
        meta = shard.get('shard', {})
        if type(meta.get('schema')) is not int or meta.get('schema') != SCHEMA or meta.get('status') != 'complete' or type(meta.get('count')) is not int or meta.get('count') != count:
            raise ValueError('Failed, incomplete, or incompatible shard')
        index = meta.get('index')
        jobs = membership(trusted_manifest, first['smoke'], index, count)
        if index in seen:
            raise ValueError('Duplicate shard index')
        seen.add(index)
        if meta.get('jobs') != jobs:
            raise ValueError('Incorrect shard membership')
        validate_records(shard.get('records'), [key for job in jobs for key in expected_jobs[job]])
        if records.keys() & shard['records'].keys():
            raise ValueError('Duplicate record keys')
        records.update(shard['records'])
        receipts.append(deepcopy(meta))
        resources[str(index)] = deepcopy(shard.get('capture_resources', {}))
    if seen != set(range(count)):
        raise ValueError('Missing shards')
    validate_records(records, [key for keys in expected_jobs.values() for key in keys])
    output = {field: deepcopy(first[field]) for field in PROVENANCE}
    output.update(records=records, merged_shards=sorted(receipts, key=lambda item: item['index']), shard_resources=resources)
    return output


def verify_serial_migration(reference, current, serial_receipt):
    """One-time 0b8c087 serial -> current sharded bridge; no generic hash bypass."""
    from examples import characterize_simulator as harness
    pinned_manifest = json.loads(subprocess.check_output(['git', 'show', f'{MIGRATION_COMMIT}:tests/characterization_manifest.json'], cwd=ROOT))
    if exact_json(reference.get('manifest')) != exact_json(pinned_manifest) or exact_json(current.get('manifest')) != exact_json(pinned_manifest):
        raise ValueError('Migration requires the pinned manifest')
    old_bytes = subprocess.check_output(['git', 'show', f'{MIGRATION_COMMIT}:examples/characterize_simulator.py'], cwd=ROOT)
    if reference.get('harness_sha256') != hashlib.sha256(old_bytes).hexdigest():
        raise ValueError('Migration requires the pinned original serial harness')
    if current.get('harness_files') != harness_files() or current.get('harness_sha256') != harness_digest(harness_files()):
        raise ValueError('Migration requires the current frozen harness')
    if reference.get('source_commit') != MIGRATION_COMMIT or current.get('source_commit') != MIGRATION_COMMIT:
        raise ValueError('Migration requires the pinned source commit')
    if reference.get('smoke') is not False or current.get('smoke') is not False:
        raise ValueError('Migration requires full captures')
    if not current.get('merged_shards'):
        raise ValueError('Migration requires a merged capture')
    harness.verify_source(reference, MIGRATION_COMMIT)
    harness.verify_source(current, MIGRATION_COMMIT)
    expected = [key for keys in job_records(pinned_manifest).values() for key in keys]
    validate_records(reference['records'], expected)
    validate_records(current['records'], expected)
    if type(serial_receipt.get('exit_code')) is not int or serial_receipt['exit_code'] != 0:
        raise ValueError('Serial baseline process failed or has no completion status')
    reference_digest = hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()
    if serial_receipt.get('capture_digest') != reference_digest:
        raise ValueError('Serial runtime receipt is not bound to this baseline')
    if exact_json(serial_receipt.get('execution_settings')) != exact_json(current.get('execution_settings')):
        raise ValueError('Serial execution settings differ')
    adjusted = deepcopy(reference)
    adjusted['harness_files'] = current['harness_files']
    adjusted['execution_settings'] = current['execution_settings']
    adjusted['harness_sha256'] = current['harness_sha256']
    harness.compare(adjusted, current)
    return dict(kind='old-serial-to-sharded-v1', source_commit=MIGRATION_COMMIT,
                old_harness_sha256=reference['harness_sha256'], new_harness_sha256=current['harness_sha256'],
                records=len(expected))


def verify_completions(paths, receipts):
    """Require launcher-observed zero exit codes bound to exact artifact bytes."""
    names = [str(Path(path).resolve()) for path in paths]
    if len(set(names)) != len(names) or set(receipts) != set(names):
        raise ValueError('Completion receipts must cover each shard path exactly once')
    snapshots = {}
    for path, name in zip(paths, names):
        receipt = receipts[name]
        if type(receipt.get('exit_code')) is not int or receipt['exit_code'] != 0:
            raise ValueError('Failed shard process')
        data = Path(path).read_bytes()
        if receipt.get('sha256') != hashlib.sha256(data).hexdigest():
            raise ValueError('Shard completion receipt hash mismatch')
        snapshots[name] = data
    return snapshots


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from examples import characterize_simulator as harness
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shards', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--completion-receipts', required=True, type=Path, help='JSON map of absolute artifact paths to launcher exit_code and sha256')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--compare', type=Path)
    group.add_argument('--migrate-serial-reference', type=Path)
    parser.add_argument('--verify-source')
    parser.add_argument('--serial-runtime-receipt', type=Path, help='Baseline-bound CPU/thread receipt required for migration')
    args = parser.parse_args()
    if bool(args.migrate_serial_reference) != bool(args.serial_runtime_receipt):
        parser.error('Migration requires --serial-runtime-receipt, and only migration accepts it')
    if args.output.exists():
        parser.error('Output exists; choose a new path')
    receipts = load_capture(args.completion_receipts)
    snapshots = verify_completions(args.shards, receipts)
    output = merge([json.loads(data, object_pairs_hook=reject_duplicate_pairs) for data in snapshots.values()])
    output['completion_receipts'] = receipts
    output['shard_artifacts'] = {name: hashlib.sha256(data).hexdigest() for name, data in snapshots.items()}
    if args.verify_source:
        harness.verify_source(output, args.verify_source)
    if args.compare:
        harness.compare(load_capture(args.compare), output)
    if args.migrate_serial_reference:
        reference_bytes = args.migrate_serial_reference.read_bytes()
        receipt_bytes = args.serial_runtime_receipt.read_bytes()
        reference = json.loads(reference_bytes, object_pairs_hook=reject_duplicate_pairs)
        serial_receipt = json.loads(receipt_bytes, object_pairs_hook=reject_duplicate_pairs)
        output['migration'] = verify_serial_migration(reference, output, serial_receipt)
        output['migration']['runtime_receipt_sha256'] = hashlib.sha256(receipt_bytes).hexdigest()
        output['migration']['reference_sha256'] = hashlib.sha256(reference_bytes).hexdigest()
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2)
    print(f'Saved {len(output["records"])} verified records to {args.output}')


if __name__ == '__main__':
    main()
