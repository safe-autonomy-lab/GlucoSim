"""Generate the fixed 15-case noise-on README gallery (not a parameter sweep).

JAX_PLATFORMS=cpu python -m examples.generate_readme_traces --output-dir /new/traces
Use an allocated CPU job. Raw outputs belong outside the repository.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from examples.characterize_simulator import action_trace, ode_rollout
from glucosim.simglucose.core.params import create_env_params
from glucosim.simglucose.physiology.initialization import tune_initial_state
from glucosim.simglucose.sim.sensor import cgm_measurement

CONFIGS = [('t1d_balance', 't1d', dict(k1=.07, Km0=240., kp2=.003)),
           ('t2d_insulin', 't2d', dict(HEb=.5, Ib=120., m4=.2)),
           ('no_pump_insulin', 't2d_no_pump', dict(HEb=.5, m4=.2))]


@jax.jit
def sensor_trace(states, vg, cfg):
    # Separate observation stream: sampling CGM cannot perturb physiology.
    key = jax.random.fold_in(jax.random.PRNGKey(42), 1)
    def advance(carry, gp):
        key, scale = carry
        reading, scale, key = cgm_measurement(gp, vg, key, cfg, scale)
        return (key, scale), reading
    return jax.lax.scan(advance, (key, jnp.array(1.)), states[:, 3])[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / 'glucosim').rglob('*.py')) + [Path(__file__).resolve(),
        root / 'examples/characterize_simulator.py', root / 'tests/characterization_manifest.json']
    def hashes():
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    before = hashes()
    manifest = json.loads((root / 'tests/characterization_manifest.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for config, kind, overrides in CONFIGS:
        env, initial = tune_initial_state(create_env_params('adolescent#001', diabetes_type=kind, **overrides))
        cfg = env.noise_config
        assert cfg.enable and not cfg.use_ou
        for case, spec in manifest['ode']['cases'].items():
            actions = action_trace(spec, 180)
            result = ode_rollout(env.patient_params, initial, cfg, jax.random.PRNGKey(42), actions)
            states = jnp.concatenate([initial[None, :], result[1][0]])
            glucose = np.asarray(states[:, 3] / env.patient_params.Vg)
            cgm = np.asarray(sensor_trace(states, env.patient_params.Vg, cfg))
            assert states.shape == (181, 18) and np.isfinite(states).all()
            assert np.isfinite(cgm).any() and not np.isinf(cgm).any()
            target = args.output_dir / f'{config}_{case}.npz'
            np.savez_compressed(target, states=np.asarray(states), actions=np.asarray(actions),
                                glucose=glucose, cgm=cgm, random_keys=np.asarray(result[1][1]))
            rows.append(dict(config=config, kind=kind, requested=overrides, pattern=case,
                artifact=target.name, artifact_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                glucose_min=float(glucose.min()), glucose_max=float(glucose.max()),
                below_70=int((glucose < 70).sum()), below_54=int((glucose < 54).sum()),
                cgm_dropouts=int(np.isnan(cgm).sum()), completed=True,
                termination='fixed-horizon ODE; no episode termination policy applied'))
            print(config, case, rows[-1]['glucose_min'], rows[-1]['glucose_max'], flush=True)
    assert hashes() == before and len(rows) == 15
    report = dict(source_hashes=before, seed=42, noise='default enabled',
        noise_config=dataclasses.asdict(cfg), patient='adolescent#001',
        sensor_key='fold_in(PRNGKey(42), 1); separate stream; scale starts at 1',
        initialization='direct tuning; no Gym 72-hour warmup',
        runtime=dict(python=platform.python_version(), jax=jax.__version__, jaxlib=jaxlib.__version__,
                     numpy=np.__version__, backend=jax.default_backend(), x64=jax.config.jax_enable_x64),
        cases=manifest['ode']['cases'], rows=rows)
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
