"""Field-level evidence for the bounded stored-volume -> computed-volume transition.

Run this frozen script with --source CHECKOUT (the checkout is never modified):
  python -m examples.characterization_volumes capture --expected-source-json HASHES \
      --expected-verifier-sha256 SHA --evidence NEW_SIDECAR -- --output NEW_CAPTURE [--smoke | --shard-index I --shard-count N]
Merge sidecars with launcher receipts (path -> {exit_code: 0, sha256: file hash, capture_kind: worker}):
  python -m examples.characterization_volumes merge --receipts RECEIPTS --output NEW SIDECARS...
Compare complete sidecars against the untouched pre-transition baseline:
  python -m examples.characterization_volumes compare --before OLD --after NEW \
      --baseline BASELINE --before-source HASHES --after-source HASHES \
      --baseline-sha256 SHA --receipts RECEIPTS --output REPORT
Comparison receipts require capture_kind serial or merged; merged receipts also
require all original worker paths, bytes, and worker receipts for reconstruction.
Source maps are reviewed, frozen path -> SHA256 maps, not inferred from outputs.
Raw field arrays are represented by exact dtype/shape/byte hashes; nested digest
calls (Gym frames) are expanded, never treated as exempt mixed records.
"""
import argparse
from copy import deepcopy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

# An external script can inspect a frozen source without writing into it.
if '--source' in sys.argv:
    index = sys.argv.index('--source')
    source_root = Path(sys.argv[index + 1]).resolve()
    del sys.argv[index:index + 2]
else:
    source_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source_root))
from examples import characterization_shards as shards

REMOVED = frozenset(('V_G', 'V_I', 'V_G_L', 'V_I_L'))
APPROVED = frozenset('glucosim/simglucose/core/' + n + '.py' for n in
                     ('params', 'conversion', 'configuration', 'parameter_builder'))
PATIENT = 'glucosim.simglucose.core.params.PatientParams'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value, previous):
    import numpy as np
    if dataclasses.is_dataclass(value):
        name = type(value).__module__ + '.' + type(value).__qualname__
        fields = {f.name: encode(getattr(value, f.name), previous) for f in dataclasses.fields(value)}
        result = {'dataclass': name, 'fields': fields}
        if name == PATIENT:
            # Independent arithmetic: do not call production conversion helpers.
            result['volumes'] = {name: encode(getattr(value, name), previous) for name in ('V_G_L', 'V_I_L')}
            result['expected_volumes'] = {'V_G_L': encode(value.Vg * value.BW / 10.0, previous),
                                          'V_I_L': encode(value.Vi * value.BW, previous)}
        return result
    if isinstance(value, str):
        return {'nested_digest': previous[value]} if value in previous else {'str': value}
    if value is None:
        return {'none': True}
    if isinstance(value, dict):
        return {'dict': [[encode(k, previous), encode(v, previous)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {'sequence_type': type(value).__module__ + '.' + type(value).__qualname__,
                'items': [encode(v, previous) for v in value]}
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError('Unsupported object leaf: ' + repr(type(value)))
    return {'dtype': array.dtype.str, 'shape': list(array.shape),
            'bytes_sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, separators=(',', ':'))


def authenticated_json(path, expected_sha256):
    """Authenticate and parse one immutable byte snapshot, never reopen the path."""
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Hash does not bind artifact: ' + str(path))
    return json.loads(data, object_pairs_hook=shards.reject_duplicate_pairs)


def require_receipt(path, receipts):
    receipt = receipts.get(str(Path(path).resolve()))
    if not isinstance(receipt, dict) or type(receipt.get('exit_code')) is not int or receipt['exit_code'] != 0:
        raise ValueError('Failed or missing worker receipt: ' + str(path))
    return authenticated_json(path, receipt.get('sha256'))


def validate(evidence, source=None):
    capture = evidence['capture']
    if evidence.get('schema') != 1 or evidence.get('verifier_sha256') != file_hash(__file__):
        raise ValueError('Incompatible field verifier')
    if source is not None and capture['source_hashes'] != source:
        raise ValueError('Unexpected frozen source')
    if capture['harness_files'] != shards.harness_files() or capture['harness_sha256'] != shards.harness_digest(shards.harness_files()):
        raise ValueError('Unexpected frozen harness')
    if type(capture.get('smoke')) is not bool:
        raise ValueError('Invalid smoke mode')
    if shards.exact_json(capture['manifest']) != shards.exact_json(shards.load_capture(shards.ROOT / 'tests/characterization_manifest.json')):
        raise ValueError('Unexpected manifest')
    jobs = shards.job_records(capture['manifest'], capture['smoke'])
    selected = capture.get('shard', {}).get('jobs', list(jobs))
    shards.validate_records(capture['records'], [k for j in selected for k in jobs[j]], capture['manifest'])
    if set(evidence['fields']) != set(capture['records']):
        raise ValueError('Missing field records')
    for key, value in evidence['fields'].items():
        if value.get('fingerprint') != capture['records'][key]:
            raise ValueError('Field/fingerprint binding mismatch: ' + key)


def capture(args):
    from examples import characterize_simulator as harness
    verifier_before = file_hash(__file__)
    if args.expected_verifier_sha256 != verifier_before:
        raise ValueError('Unexpected verifier before capture')
    if Path(args.evidence).exists():
        raise ValueError('Evidence output already exists')
    source = shards.load_capture(args.expected_source_json)
    root = harness.MANIFEST.parent.parent
    hashes = {str(p.relative_to(root)): file_hash(p) for p in sorted((root / 'glucosim').rglob('*.py'))}
    if hashes != source:
        raise ValueError('Source differs before capture')
    argv = args.harness_args[1:] if args.harness_args[:1] == ['--'] else args.harness_args
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument('--output', type=Path, required=True)
    forwarded, _ = output_parser.parse_known_args(argv)
    output = forwarded.output
    previous = {}
    original = harness.digest
    def record(value):
        digest = original(value)
        encoded = encode(value, previous)
        if digest in previous and previous[digest] != encoded:
            raise ValueError('Ambiguous fingerprint field evidence')
        previous[digest] = encoded
        return digest
    harness.digest = record
    old_argv = sys.argv
    try:
        sys.argv = ['characterize_simulator.py'] + argv
        harness.main()
    finally:
        sys.argv = old_argv
        harness.digest = original
    standard = shards.load_capture(output)
    fields = {}
    for key, digest in standard['records'].items():
        if digest in previous:
            payload = previous[digest]
        elif key.endswith(('/patient_rejection', '/created_rejection')):
            payload = {'declared_rejection': digest}
        else:
            raise ValueError('No field evidence for ' + key)
        fields[key] = {'fingerprint': digest, 'payload': payload}
    result = {'schema': 1, 'verifier_sha256': file_hash(__file__), 'capture': standard,
              'capture_file_sha256': file_hash(output), 'fields': fields}
    validate(result, source)
    if file_hash(__file__) != verifier_before:
        raise ValueError('Field verifier changed during capture')
    publish(args.evidence, result)


def merge(paths, receipts):
    evidence = []
    for path in paths:
        item = require_receipt(path, receipts)
        if receipts[str(Path(path).resolve())].get('capture_kind') != 'worker':
            raise ValueError('Expected a launcher-observed worker receipt')
        validate(item)
        evidence.append(item)
    combined = shards.merge([item['capture'] for item in evidence])
    fields = {}
    for item in evidence:
        if fields.keys() & item['fields'].keys():
            raise ValueError('Duplicate field records')
        fields.update(item['fields'])
    result = {'schema': 1, 'verifier_sha256': file_hash(__file__), 'capture': combined, 'fields': fields,
              'merged_evidence_paths': [str(Path(path).resolve()) for path in paths]}
    validate(result)
    return result


def completed_evidence(path, receipts):
    """Use launcher-declared execution kind, never removable artifact markers."""
    item = require_receipt(path, receipts)
    kind = receipts[str(Path(path).resolve())].get('capture_kind')
    if kind == 'serial':
        if 'shard' in item['capture'] or 'merged_shards' in item['capture'] or 'merged_evidence_paths' in item:
            raise ValueError('Serial receipt cannot authenticate worker/merged evidence')
        validate(item)
    elif kind == 'merged':
        paths = item.get('merged_evidence_paths')
        if not isinstance(paths, list) or not paths:
            raise ValueError('Missing merged worker evidence')
        rebuilt = merge(paths, receipts)
        if shards.exact_json(item) != shards.exact_json(rebuilt):
            raise ValueError('Merged evidence differs from receipt-bound workers')
    else:
        raise ValueError('Missing or invalid launcher capture kind')
    return item


def compare_fields(before, after, path='$', counts=None):
    if counts is None:
        counts = {'patients': 0, 'removed_fields': 0, 'verified_properties': 0}
    if isinstance(before, dict) and before.get('dataclass') == PATIENT:
        if not isinstance(after, dict) or after.get('dataclass') != PATIENT:
            raise AssertionError('Patient representation changed at ' + path)
        for node in (before, after):
            if set(node) != {'dataclass', 'fields', 'volumes', 'expected_volumes'}:
                raise AssertionError('Unexpected patient metadata at ' + path)
            for key in ('volumes', 'expected_volumes'):
                if not isinstance(node[key], dict) or set(node[key]) != {'V_G_L', 'V_I_L'}:
                    raise AssertionError('Incomplete computed volume evidence at ' + path)
        old = deepcopy(before)
        new = deepcopy(after)
        if not REMOVED <= old['fields'].keys() or REMOVED & new['fields'].keys():
            raise AssertionError('Unexpected stored volume fields at ' + path)
        for field in REMOVED:
            del old['fields'][field]
        if new['volumes'] != new['expected_volumes']:
            raise AssertionError('Incorrect computed volume at ' + path)
        for obj in (old, new):
            del obj['volumes']
            del obj['expected_volumes']
        counts['patients'] += 1
        counts['removed_fields'] += 4
        counts['verified_properties'] += 2
        compare_fields(old['fields'], new['fields'], path + '.fields', counts)
        if old.keys() != new.keys():
            raise AssertionError('Unexpected patient metadata at ' + path)
        return counts
    if type(before) is not type(after):
        raise AssertionError('Type changed at ' + path)
    if isinstance(before, dict):
        if before.keys() != after.keys():
            raise AssertionError('Fields changed at ' + path)
        for key in before:
            compare_fields(before[key], after[key], path + '.' + key, counts)
    elif isinstance(before, list):
        if len(before) != len(after):
            raise AssertionError('Length changed at ' + path)
        for index, (old, new) in enumerate(zip(before, after)):
            compare_fields(old, new, path + '[' + str(index) + ']', counts)
    elif before != after:
        raise AssertionError('Retained value changed at ' + path)
    return counts


def compare(before, after, baseline, before_source, after_source):
    validate(before, before_source)
    validate(after, after_source)
    old, new = before['capture'], after['capture']
    if 'shard' in old or 'shard' in new:
        raise ValueError('Comparison requires complete captures')
    if set(before_source) != set(after_source) or not {k for k in before_source if before_source[k] != after_source[k]} <= APPROVED:
        raise ValueError('Source changes exceed the approved volume repair')
    for field in shards.PROVENANCE:
        if field != 'source_hashes' and shards.exact_json(old[field]) != shards.exact_json(new[field]):
            raise ValueError('Incompatible provenance: ' + field)
    for field in shards.PROVENANCE:
        if field != 'smoke' and shards.exact_json(old[field]) != shards.exact_json(baseline[field]):
            raise ValueError('Before capture differs from baseline provenance: ' + field)
    for key, value in old['records'].items():
        if baseline['records'].get(key) != value:
            raise AssertionError('Before fingerprint differs from baseline: ' + key)
    if before['fields'].keys() != after['fields'].keys():
        raise AssertionError('Record coverage changed')
    counts = {'patients': 0, 'removed_fields': 0, 'verified_properties': 0}
    for key in before['fields']:
        compare_fields(before['fields'][key]['payload'], after['fields'][key]['payload'], key, counts)
    if not counts['patients']:
        raise ValueError('No patient volume evidence')
    return {'status': 'passed', 'records': len(old['records']), **counts,
            'unchanged_fingerprints': sum(old['records'][k] == new['records'][k] for k in old['records']),
            'before_source': before_source, 'after_source': after_source,
            'verifier_sha256': file_hash(__file__)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    cap = sub.add_parser('capture')
    cap.add_argument('--expected-source-json', required=True)
    cap.add_argument('--expected-verifier-sha256', required=True)
    cap.add_argument('--evidence', required=True)
    cap.add_argument('harness_args', nargs=argparse.REMAINDER)
    mer = sub.add_parser('merge')
    mer.add_argument('--receipts', required=True)
    mer.add_argument('--output', required=True)
    mer.add_argument('paths', nargs='+')
    cmp = sub.add_parser('compare')
    for name in ('before', 'after', 'baseline', 'before-source', 'after-source', 'baseline-sha256', 'receipts', 'output'):
        cmp.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.command == 'capture':
        capture(args)
    elif args.command == 'merge':
        publish(args.output, merge(args.paths, shards.load_capture(args.receipts)))
    else:
        receipts = shards.load_capture(args.receipts)
        before = completed_evidence(args.before, receipts)
        after = completed_evidence(args.after, receipts)
        baseline = authenticated_json(args.baseline, args.baseline_sha256)
        result = compare(before, after, baseline, *(shards.load_capture(getattr(args, name)) for name in
                           ('before_source', 'after_source')))
        publish(args.output, result)
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
