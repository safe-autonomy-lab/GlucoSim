"""One-time, pinned v3 -> v4 expected-rejection contract migration.

Run on the capture runtime after the launcher observes successful completion::

    python -m examples.characterization_migration --reference OLD --current NEW \
        --current-receipt RECEIPT --output NEW_REPORT

RECEIPT is {"exit_code": 0, "sha256": SHA256_OF_NEW_FILE_BYTES,
"capture_kind": "serial" or "merged"}. The trusted launcher records the actual
capture kind from its invoked command, never by inferring artifact markers,
alongside its observed exit and artifact hash. These are trusted launcher
assertions, not protection against forged receipts. The old
baseline is never rewritten. A merged current capture must retain readable raw
worker files at its recorded paths, with embedded successful completion receipts;
the verifier reconstructs the merge from those files. New successful records are checked by a bounded
construction/initialization replay, not described as old-baseline equality.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from examples import characterization_shards as shards

OLD_SOURCE = '0b8c087a27e40c5b179106458577527fa992f9a1'
NEW_SOURCE = '1130a478de832cee01452942d85dd1b375030b32'
OLD_BASELINE_SHA256 = 'b359401544c0f147cb17f8ca8b81ecdea3ac75f91c19a9021242bf3a58d9b323'
OLD_HARNESS = ('examples/characterize_simulator.py', 'examples/characterization_shards.py')
CASE = 'final_override'
VALID_CASE = 'final_override_no_pump_valid'
SPEC = {'types': ['t2d_no_pump'], 'exception': 'ValueError',
        'message': 'use_pump=False requires basal=0',
        'origin': 'build_patient_params.resolved_delivery_guard'}


def git_bytes(revision, path):
    return subprocess.check_output(['git', 'show', f'{revision}:{path}'], cwd=shards.ROOT)


def expected_manifest():
    old = json.loads(git_bytes(NEW_SOURCE, 'tests/characterization_manifest.json'))
    new = deepcopy(old)
    new['version'] = 4
    cases = new['nondefault']['factory_cases']
    cases[CASE]['expected_rejection'] = deepcopy(SPEC)
    cases[VALID_CASE] = {'types': ['t2d_no_pump'], 'kwargs': {
        'carb_absorption_scale': 1.4, 'kabs': 0.071, 'basal': 0.0}}
    return old, new


def runtime():
    import jax
    import jaxlib
    import numpy as np
    return dict(python=platform.python_version(), platform=platform.platform(),
                jax=jax.__version__, jaxlib=jaxlib.__version__, numpy=np.__version__,
                backend=jax.default_backend(), x64=bool(jax.config.x64_enabled))


def live_sources():
    return {str(path.relative_to(shards.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((shards.ROOT / 'glucosim').rglob('*.py'))}


def replay_counterpart(manifest):
    from examples import characterize_simulator as harness
    bounded = deepcopy(manifest)
    bounded['types'] = ['t2d_no_pump']
    bounded['nondefault']['factory_cases'] = {
        VALID_CASE: bounded['nondefault']['factory_cases'][VALID_CASE]}
    bounded['nondefault']['adapter_configs'] = {}
    return harness.capture_nondefault(bounded)


def verify_merged_evidence(capture, capture_kind):
    """Rebuild a merged artifact from successful, byte-bound worker evidence."""
    fields = ('merged_shards', 'shard_artifacts', 'completion_receipts', 'shard_resources')
    if capture_kind == 'serial':
        if any(field in capture for field in fields):
            raise ValueError('Serial capture contains merged worker evidence')
        return
    if capture_kind != 'merged':
        raise ValueError('Completion receipt requires serial or merged capture_kind')
    if not all(isinstance(capture.get(field), dict if field != 'merged_shards' else list)
               for field in fields):
        raise ValueError('Merged capture lacks complete worker evidence')
    artifacts = capture['shard_artifacts']
    snapshots = shards.verify_completions(list(artifacts), capture['completion_receipts'])
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in snapshots.items()}
    if shards.exact_json(hashes) != shards.exact_json(artifacts):
        raise ValueError('Merged shard artifact hashes differ from worker evidence')
    rebuilt = shards.merge([json.loads(data, object_pairs_hook=shards.reject_duplicate_pairs)
                            for data in snapshots.values()])
    for field, value in rebuilt.items():
        if field not in capture or shards.exact_json(capture[field]) != shards.exact_json(value):
            raise ValueError(f'Merged capture differs from reconstructed workers: {field}')


def verify_migration(reference_path, current_path, current_receipt):
    """Validate exact provenance and every record in this single approved bridge."""
    from examples import characterize_simulator as harness
    from examples.characterization_contract import rejection_digest

    reference_path, current_path = Path(reference_path), Path(current_path)
    old_bytes, new_bytes = reference_path.read_bytes(), current_path.read_bytes()
    if hashlib.sha256(old_bytes).hexdigest() != OLD_BASELINE_SHA256:
        raise ValueError('Migration requires the pinned untouched server baseline')
    if type(current_receipt.get('exit_code')) is not int or current_receipt['exit_code'] != 0:
        raise ValueError('Current capture process failed or lacks completion evidence')
    if current_receipt.get('sha256') != hashlib.sha256(new_bytes).hexdigest():
        raise ValueError('Completion receipt does not match current artifact')
    capture_kind = current_receipt.get('capture_kind')
    if capture_kind not in ('serial', 'merged'):
        raise ValueError('Completion receipt requires serial or merged capture_kind')
    old = json.loads(old_bytes, object_pairs_hook=shards.reject_duplicate_pairs)
    new = json.loads(new_bytes, object_pairs_hook=shards.reject_duplicate_pairs)
    old_manifest, new_manifest = expected_manifest()
    for capture, manifest, revision in ((old, old_manifest, OLD_SOURCE),
                                         (new, new_manifest, NEW_SOURCE)):
        if shards.exact_json(capture.get('manifest')) != shards.exact_json(manifest):
            raise ValueError('Manifest differs from the pinned migration contract')
        if capture.get('source_commit') != revision:
            raise ValueError('Migration requires the pinned source commit')
        if capture.get('smoke') is not False or 'shard' in capture:
            raise ValueError('Migration requires completed full captures, not workers')
        harness.verify_source(capture, revision)
    old_files = {name: hashlib.sha256(git_bytes(NEW_SOURCE, name)).hexdigest()
                 for name in OLD_HARNESS}
    files_before = shards.harness_files()
    for capture, files in ((old, old_files), (new, files_before)):
        if capture.get('harness_files') != files or capture.get('harness_sha256') != shards.harness_digest(files):
            raise ValueError('Harness provenance differs from the pinned migration')
    for field in ('runtime', 'execution_settings', 'input_hashes'):
        if field not in old or field not in new or shards.exact_json(old[field]) != shards.exact_json(new[field]):
            raise ValueError(f'Migration requires identical {field}')
    if (new['runtime'].get('backend') != 'cpu' or
            shards.exact_json(new['runtime']) != shards.exact_json(runtime()) or
            shards.exact_json(new['execution_settings']) != shards.exact_json(shards.execution_settings())):
        raise ValueError('Bounded replay requires the capture CPU runtime and execution settings')
    if new['source_hashes'] != live_sources() or new['input_hashes'] != harness.input_hashes(new_manifest):
        raise ValueError('Bounded replay requires the frozen capture source and inputs')
    verify_merged_evidence(new, capture_kind)
    prefix = f'nondefault/{old_manifest["nondefault"]["patient"]}/t2d_no_pump/factory/'
    removed = {prefix + CASE + '/' + stage for stage in ('patient', 'created', 'tuned')}
    rejections = {prefix + CASE + '/' + stage + '_rejection': rejection_digest(stage + '_rejection', SPEC)
                  for stage in ('patient', 'created')}
    added = {prefix + VALID_CASE + '/' + stage for stage in ('patient', 'created', 'tuned')}
    old_expected = [key for keys in shards.job_records(old_manifest).values() for key in keys]
    new_expected = [key for keys in shards.job_records(new_manifest).values() for key in keys]
    shards.validate_records(old.get('records'), old_expected)
    shards.validate_records(new.get('records'), new_expected)
    common = set(old_expected) & set(new_expected)
    if (len(common) != 6268 or set(old_expected) - set(new_expected) != removed or
            set(new_expected) - set(old_expected) != added | rejections.keys()):
        raise ValueError('Migration record accounting differs from the approved delta')
    changed = sorted(key for key in common if old['records'][key] != new['records'][key])
    if changed:
        raise AssertionError(f'Unaffected records changed: {changed[:8]}')
    if any(new['records'][key] != value for key, value in rejections.items()):
        raise AssertionError('Expected rejection digest differs')
    replay = replay_counterpart(new_manifest)
    if set(replay) != added or any(new['records'][key] != replay[key] for key in added):
        raise AssertionError('Added valid counterpart differs from bounded replay')
    if (files_before != shards.harness_files() or new['source_hashes'] != live_sources() or
            new['input_hashes'] != harness.input_hashes(new_manifest)):
        raise ValueError('Frozen replay source, harness, or inputs changed during verification')
    return dict(kind='pinned-v3-to-v4-delivery-rejection',
                reference_sha256=OLD_BASELINE_SHA256,
                current_sha256=hashlib.sha256(new_bytes).hexdigest(),
                capture_kind=capture_kind,
                old_source=OLD_SOURCE, new_source=NEW_SOURCE,
                old_harness=old['harness_sha256'], new_harness=new['harness_sha256'],
                unaffected_exact_records=len(common), removed_records=sorted(removed),
                expected_rejection_records=rejections, added_bounded_replay_records=replay,
                note='Added records pass bounded replay; they have no old-baseline oracle.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'current', 'current-receipt', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path')
    report = verify_migration(args.reference, args.current, shards.load_capture(args.current_receipt))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print('6268 unchanged records exact; 3 removed, 2 expected rejections, 3 valid additions verified')


if __name__ == '__main__':
    main()
