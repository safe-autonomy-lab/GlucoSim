"""Capture exact simulator fingerprints without writing into the source tree.

    python examples/characterize_simulator.py --output /tmp/legacy.json
    python examples/characterize_simulator.py --output /tmp/postfix.json --compare /tmp/legacy.json --allow-reset-fixes

Default is the full manifest; --smoke selects its explicitly defined subset.
Both modes include the nondefault factory/adapter cases. Captures record the
bundled and generated CSV hashes. For strict refactors, use --verify-source REV
and --verify-reference-source REV to bind each side to its intended Git source.
Version 4 records declared factory rejections and a valid no-pump late override.
For the pinned v3 -> v4 bridge, run python -m examples.characterization_migration
--reference OLD --current NEW --current-receipt RECEIPT --output NEW_REPORT.
The receipt binds successful process exit to the new artifact bytes; see that
module for its schema. Normal exact comparisons require the same harness on both sides. CSV provenance predating
version 3 cannot be reconstructed retrospectively.
Output paths must not exist. Digests cover full arrays, keys and public outputs,
not just plasma glucose. This is characterization, not clinical validation.
Capture releases JAX compilation caches between groups to bound memory; run
legacy and repaired captures sequentially. New reset keys require independent
72-hour stochastic warmups; repeated identical keys can reuse cached states.
"""
import argparse
import csv
import dataclasses
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
from functools import lru_cache
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from examples import characterization_shards as sharding

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

import glucosim
from glucosim import gym_env as gym
from glucosim.simglucose.core import params as patient_factory
from glucosim.simglucose.core.params import create_env_params
from glucosim.simglucose.core.types import PatientType
from glucosim.simglucose.physiology.initialization import tune_initial_state
from glucosim.simglucose.physiology.glucose_dynamics import t1d_rk4_step, t2d_rk4_step
from glucosim.simglucose.sim.reset import WARMUP_MINUTES

MANIFEST = Path(__file__).resolve().parents[1] / 'tests' / 'characterization_manifest.json'
PATIENT_CSV = 'glucosim/simglucose/params/vpatient_params.csv'


def patient_inputs(manifest):
    """Hash the exact bundled and generated CSV bytes used by this harness."""
    bundled = (MANIFEST.parent.parent / PATIENT_CSV).read_bytes()
    reader = csv.DictReader(io.StringIO(bundled.decode()))
    rows = list(reader)
    matches = [row for row in rows if row['Name'] == manifest['nondefault']['patient']]
    if len(matches) != 1:
        raise ValueError('Alternative CSV requires exactly one matching patient')
    for field, value in manifest['nondefault']['csv_changes'].items():
        if field not in reader.fieldnames:
            raise ValueError(f'Unknown CSV field: {field}')
        matches[0][field] = str(value)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=reader.fieldnames, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return {PATIENT_CSV: bundled, 'generated/alternative_patient.csv': stream.getvalue().encode()}


def input_hashes(manifest):
    return {name: hashlib.sha256(data).hexdigest() for name, data in patient_inputs(manifest).items()}


def capture_nondefault(manifest):
    """Capture successful factories or declared, origin-checked rejections."""
    from examples.characterization_contract import expected_rejection, rejection_digest, require_factory_rejection
    from glucosim.simglucose.core.parameter_builder import build_patient_params
    spec = manifest['nondefault']
    records = {}
    acceptance = patient_factory.ACCEPTANCE_PROB_DEFAULT
    try:
        with tempfile.TemporaryDirectory(prefix='glucosim-characterization-') as directory:
            alternative = Path(directory) / 'patients.csv'
            alternative.write_bytes(patient_inputs(manifest)['generated/alternative_patient.csv'])
            for kind in manifest['types']:
                for name, case in spec['factory_cases'].items():
                    if kind not in case.get('types', manifest['types']):
                        continue
                    patient_factory.ACCEPTANCE_PROB_DEFAULT = case.get('acceptance', acceptance)
                    kwargs = dict(case.get('kwargs', {}))
                    if case.get('alternative_csv'):
                        kwargs['csv_path'] = str(alternative)
                    prefix = f'nondefault/{spec["patient"]}/{kind}/factory/{name}'
                    rejection = expected_rejection(case, kind)
                    if rejection is not None:
                        for stage, factory in (('patient_rejection', patient_factory.create_patient_params),
                                               ('created_rejection', create_env_params)):
                            require_factory_rejection(
                                lambda: factory(spec['patient'], diabetes_type=kind, **kwargs),
                                build_patient_params, rejection)
                            records[prefix + '/' + stage] = rejection_digest(stage, rejection)
                        continue
                    patient = patient_factory.create_patient_params(spec['patient'], diabetes_type=kind, **kwargs)
                    env = create_env_params(spec['patient'], diabetes_type=kind, **kwargs)
                    records[prefix + '/patient'] = digest(patient)
                    records[prefix + '/created'] = digest(env)
                    records[prefix + '/tuned'] = digest(tune_initial_state(env))
                patient_factory.ACCEPTANCE_PROB_DEFAULT = acceptance
                if kind != 't1d':
                    base = patient_factory.create_patient_params(spec['patient'], diabetes_type='t1d')
                    adapt = getattr(patient_factory, f'adapt_params_for_{kind}')
                    for name, config in spec['adapter_configs'].items():
                        records[f'nondefault/{spec["patient"]}/{kind}/adapter/{name}'] = digest(adapt(base, config=dict(config)))
    finally:
        patient_factory.ACCEPTANCE_PROB_DEFAULT = acceptance
    return records


def digest(tree):
    leaves, structure = jax.tree.flatten(tree)
    h = hashlib.sha256(str(structure).encode())
    for value in leaves:
        array = np.asarray(value)
        h.update(str((array.dtype.str, array.shape)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def noise_config(cfg, mode):
    return dataclasses.replace(cfg, enable=mode != 'ode_noise_off', use_ou=mode == 'ou')


@jax.jit
def ode_rollout(params, initial, cfg, key, actions):
    step = t1d_rk4_step if params.diabetes_type == PatientType.t1d else t2d_rk4_step
    def advance(carry, item):
        x, key, ou, previous_carb, stomach, food = carry
        minute, action = item
        carb = action[0]
        new_meal = (carb > 0) & (previous_carb == 0)
        stomach = jnp.where(new_meal, x[0] + x[1], stomach)
        food = jnp.where(new_meal, 0., food) + carb  # g/min * 1 min
        key, subkey = jax.random.split(key)
        x, key, ou = step(x, 1., action, params, stomach, food, minute, subkey, cfg, ou)
        return (x, key, ou, carb, stomach, food), (x, key, ou, stomach, food)
    zero = jnp.array(0., dtype=initial.dtype)
    return jax.lax.scan(advance, (initial, key, zero, zero, zero, zero),
                        (jnp.arange(len(actions), dtype=initial.dtype), actions))


def action_trace(case, horizon):
    actions = np.zeros((horizon, 3), dtype=np.float32)
    for amount, start, duration, column in [
        ('meal_g', 'meal_start_min', 'meal_duration_min', 0),
        ('bolus_u', 'bolus_start_min', 'bolus_duration_min', 1),
    ]:
        if amount in case:
            actions[case[start]:case[start] + case[duration], column] = case[amount] / case[duration]
    if 'hr_reserve' in case:
        start = case['exercise_start_min']
        actions[start:start + case['exercise_duration_min'], 2] = case['hr_reserve']
    return jnp.asarray(actions)


def capture(manifest, smoke=False, jobs=None):
    if manifest['ode']['dt_min'] != 1.0 or manifest['gym']['warmup_minutes'] != WARMUP_MINUTES:
        raise ValueError('Manifest timestep/warmup does not match the characterization implementation')
    selected = set(sharding.job_records(manifest, smoke) if jobs is None else jobs)
    if not selected <= sharding.job_records(manifest, smoke).keys():
        raise ValueError('Unknown capture jobs')
    records = capture_nondefault(manifest) if 'nondefault' in selected else {}
    subset = manifest['smoke'] if smoke else {}
    numbers = subset.get('patient_numbers', manifest['patients']['numbers'])
    seeds = subset.get('seeds', manifest['seeds'])
    modes = subset.get('noise_modes', manifest['noise_modes'])
    cases = subset.get('ode_cases', list(manifest['ode']['cases']))
    for cohort in manifest['patients']['cohorts']:
        for number in numbers:
            patient = f'{cohort}#{number:03d}'
            for kind in manifest['types']:
                if f'ode/{kind}' not in selected:
                    continue
                prefix = f'{patient}/{kind}'
                env = create_env_params(patient_name=patient, diabetes_type=kind)
                records[prefix + '/created'] = digest(env)
                tuned, initial = tune_initial_state(env)
                records[prefix + '/tuned'] = digest((tuned, initial))
                for mode in modes:
                    cfg = noise_config(tuned.noise_config, mode)
                    for seed in seeds:
                        for name in cases:
                            actions = action_trace(manifest['ode']['cases'][name], manifest['ode']['horizon_min'])
                            result = ode_rollout(tuned.patient_params, initial, cfg, jax.random.PRNGKey(seed), actions)
                            records[f'{prefix}/ode/{mode}/{seed}/{name}'] = digest(result)
                print('Captured', prefix, flush=True)
    release_compilations()
    spec = manifest['gym']
    for cohort in manifest['patients']['cohorts']:
        for number in spec['patient_numbers']:
            patient = f'{cohort}#{number:03d}'
            for kind in manifest['types']:
                if f'gym/{patient}/{kind}' not in selected:
                    continue
                for sample in subset.get('sample_time_min', spec['sample_time_min']):
                    for mode in modes:
                        instance = gym.make(f'{kind}-v0', patient_name=patient, sample_time=sample,
                                            simulation_minutes=spec['simulation_minutes'])
                        u = instance.unwrapped
                        u.env_params = dataclasses.replace(u.env_params, noise_config=noise_config(u.env_params.noise_config, mode))
                        try:
                            for seed in seeds:
                                type(u)._warmup_cache.clear()
                                for history in spec['cache_histories']:
                                    if history == 'after_other_seed':
                                        type(u)._warmup_cache.clear()
                                        instance.reset(seed=spec['other_seed'])
                                    prefix = f'{patient}/{kind}/gym/{sample}/{mode}/{seed}/{history}'
                                    output = instance.reset(seed=seed)
                                    records[prefix + '/exposed'] = digest(u.get_patient_params())
                                    records[prefix + '/effective'] = digest(u.env_params.patient_params)
                                    records[prefix + '/reset'] = digest((output, u._jax_state, u.key, u.env_params,
                                                                        u.action_space.nvec, u.observation_space.low,
                                                                        u.observation_space.high))
                                    frames = []
                                    for i in range(spec['steps']):
                                        action = np.array(spec['actions'][i % len(spec['actions'])])
                                        output = instance.step(action)
                                        frames.append(digest((output, u._jax_state, u.key)))
                                        if output[3] or output[4]:
                                            break
                                    records[prefix + '/steps'] = digest(frames)
                        finally:
                            type(u)._warmup_cache.clear()
                            instance.close()
                release_compilations()
                print('Captured Gym', patient, kind, flush=True)
    return records


def release_compilations():
    # All captured arrays have already been synchronized by digest(). This only
    # drops harness compilation caches, never changes the numerical kernels.
    jax.clear_caches()
    gc.collect()


@lru_cache(maxsize=4)
def legacy_source(revision):
    root = MANIFEST.parent.parent
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', revision + '^{commit}'], cwd=root).decode().strip()
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, '--', 'glucosim'], cwd=root).decode().splitlines()
    hashes = {path: hashlib.sha256(subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=root)).hexdigest()
              for path in paths if path.endswith('.py')}
    return commit, hashes


def verify_source(capture, revision):
    """Bind each artifact independently; refactors intentionally change sources."""
    commit, hashes = legacy_source(revision)
    if capture['source_hashes'] != hashes:
        raise ValueError(f'Capture source hashes do not match {commit}')
    bundled = subprocess.check_output(['git', 'show', f'{commit}:{PATIENT_CSV}'], cwd=MANIFEST.parent.parent)
    if capture['input_hashes'][PATIENT_CSV] != hashlib.sha256(bundled).hexdigest():
        raise ValueError(f'Capture CSV does not match {commit}')


def compare(reference, current, allow_reset_fixes=False):
    for field in ('runtime', 'manifest', 'smoke', 'harness_sha256', 'input_hashes'):
        if field not in reference or field not in current:
            raise ValueError(f'Missing {field}; recapture with the current harness')
        if sharding.exact_json(reference[field]) != sharding.exact_json(current[field]):
            raise ValueError(f'Cannot compare different {field}')
    for field in ('harness_files', 'execution_settings'):
        if field in reference or field in current:
            if sharding.exact_json(reference.get(field)) != sharding.exact_json(current.get(field)):
                raise ValueError(f'Cannot compare different {field}')
    if current['manifest'].get('version') == 4:
        # Equal partial results are not a successful full or smoke capture.
        if sharding.exact_json(current['manifest']) != sharding.exact_json(sharding.load_capture(MANIFEST)):
            raise ValueError('Manifest differs from the frozen comparison manifest')
        if (current.get('harness_files') != sharding.harness_files()
                or current['harness_sha256'] != sharding.harness_digest(sharding.harness_files())):
            raise ValueError('Comparison requires the same frozen harness')
        for artifact in (reference, current):
            if type(artifact['smoke']) is not bool:
                raise ValueError('Invalid smoke mode')
            jobs = sharding.job_records(artifact['manifest'], artifact['smoke'])
            sharding.validate_records(artifact['records'],
                                      [key for group in jobs.values() for key in group],
                                      artifact['manifest'])
    if allow_reset_fixes:
        commit, hashes = legacy_source(reference['manifest']['legacy_commit'])
        if reference['source_commit'] != commit or reference['source_hashes'] != hashes:
            raise ValueError('Legacy reference does not match the declared baseline source')
        verify_source(reference, commit)
    old, new = reference['records'], current['records']
    if old.keys() != new.keys():
        raise AssertionError('Characterization record keys changed')
    changed = [key for key in old if old[key] != new[key]]
    if not allow_reset_fixes:
        if changed:
            raise AssertionError(f'{len(changed)} records differ: {changed[:8]}')
        return changed
    for key, value in new.items():
        if '/gym/' in key and key.endswith('/exposed'):
            if new.get(key.removesuffix('/exposed') + '/effective') != value:
                raise AssertionError(f'Exposed parameters differ from effective parameters: {key}')
    # Intentional changes are restricted to exposed parameters and mixed-seed cache history.
    unexpected = [key for key in changed if '/gym/' not in key or
                  not (key.endswith('/exposed') or ('/after_other_seed/' in key and
                       key.rsplit('/', 1)[-1] in ('reset', 'steps')))]
    if unexpected:
        raise AssertionError(f'Unexpected changes: {unexpected[:8]}')
    for key, value in new.items():
        if '/gym/' in key and '/cold/' in key:
            for history in ('repeat_same_seed', 'after_other_seed'):
                if new[key.replace('/cold/', f'/{history}/')] != value:
                    raise AssertionError(f'Cache history changed post-fix record: {key}')
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--shard-index', type=int)
    parser.add_argument('--shard-count', type=int)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--allow-reset-fixes', action='store_true')
    parser.add_argument('--verify-source', help='Require current source/CSV hashes to match this Git revision')
    parser.add_argument('--verify-reference-source', help='Require compared artifact source/CSV hashes to match this Git revision')
    args = parser.parse_args()
    if (args.shard_index is None) != (args.shard_count is None):
        parser.error('--shard-index and --shard-count are required together')
    if args.shard_index is not None and args.compare:
        parser.error('Compare only complete merged captures')
    if args.allow_reset_fixes and not args.compare:
        parser.error('--allow-reset-fixes requires --compare')
    if args.verify_reference_source and not args.compare:
        parser.error('--verify-reference-source requires --compare')
    if args.output.exists():
        parser.error('Output exists; choose a new path')
    if jax.default_backend() != 'cpu':
        parser.error('Characterization requires CPU')
    manifest = json.loads(MANIFEST.read_text())
    runtime = dict(python=platform.python_version(), platform=platform.platform(), jax=jax.__version__,
                   jaxlib=jaxlib.__version__, numpy=np.__version__, backend=jax.default_backend(),
                   x64=bool(jax.config.x64_enabled))
    files = sharding.harness_files()
    settings = sharding.execution_settings()
    started = time.monotonic()
    jobs = None if args.shard_index is None else sharding.membership(manifest, args.smoke, args.shard_index, args.shard_count)
    output = dict(manifest=manifest, runtime=runtime, smoke=args.smoke,
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=MANIFEST.parent).decode().strip(),
                  source_hashes={str(path.relative_to(MANIFEST.parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted((MANIFEST.parent.parent / 'glucosim').rglob('*.py'))},
                  harness_sha256=sharding.harness_digest(files), harness_files=files, execution_settings=settings,
                  input_hashes=input_hashes(manifest))
    if args.verify_source:
        verify_source(output, args.verify_source)
    reference = sharding.load_capture(args.compare) if args.compare else None
    if args.verify_reference_source:
        verify_source(reference, args.verify_reference_source)
    output['records'] = capture(manifest, args.smoke, jobs)
    coverage = sharding.job_records(manifest, args.smoke)
    sharding.validate_records(output['records'], [key for job in (coverage if jobs is None else jobs) for key in coverage[job]], manifest)
    if sharding.harness_files() != files or sharding.execution_settings() != settings:
        raise ValueError('Harness or execution settings changed during capture')
    source_now = {str(path.relative_to(MANIFEST.parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted((MANIFEST.parent.parent / 'glucosim').rglob('*.py'))}
    if output['source_hashes'] != source_now:
        raise ValueError('Source changed during capture')
    if jobs is not None:
        output['shard'] = dict(schema=sharding.SCHEMA, index=args.shard_index, count=args.shard_count, jobs=jobs, status='complete')
    if output['input_hashes'] != input_hashes(manifest):
        raise ValueError('Patient CSV changed during capture')
    output['capture_resources'] = {'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                   'elapsed_seconds': time.monotonic() - started,
                                   'hostname': platform.node(),
                                   'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None}
    if args.compare:
        changed = compare(reference, output, args.allow_reset_fixes)
        print('Comparison passed; intentional changed records:', len(changed))
    with args.output.open('x') as file:
        json.dump(output, file, indent=2)
    print('Saved', len(output['records']), 'records to', args.output)


if __name__ == '__main__':
    main()
