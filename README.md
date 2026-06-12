# GlucoSim

JAX-based blood glucose simulation environments for studying safe generalization
across safe reinforcement learning algorithms.

GlucoSim provides Gymnasium-style environments for Type 1 and Type 2 diabetes
virtual patients. Each step returns both a **reward** and a **safety cost**, making
the environments directly usable as constrained MDPs (CMDPs).

> **Research use only.** This simulator is a research tool for studying
> reinforcement learning algorithms. It is **not** a medical device, has not been
> validated for clinical use, and must never be used to make treatment decisions
> for real patients.

This repository contains the simulator only. For the safe RL algorithms evaluated
against it, see [GlucoAlg](https://github.com/safe-autonomy-lab/GlucoAlg).

## Installation

Requires Python >= 3.10.

```bash
git clone https://github.com/safe-autonomy-lab/GlucoSim.git
cd GlucoSim
python -m pip install -e .
```

Or install just the dependencies and run from source:

```bash
python -m pip install -r requirements.txt
```

The core package depends on `jax`, `gymnasium`, `numpy`, `pandas`, and
`matplotlib`. PyTorch is **optional** and only needed for the OmniSafe CMDP
wrapper (see below). JAX runs on CPU by default; install a CUDA-enabled `jaxlib`
if you want GPU execution.

## Quickstart

The package registers `t1d-v0`, `t2d-v0`, and `t2d_no_pump-v0` in a local
registry on import (separate from Gymnasium's global registry):

```python
import numpy as np
import glucosim  # registers t1d-v0 / t2d-v0 / t2d_no_pump-v0
from glucosim import gym_env as gym

env = gym.make(
    "t1d-v0",
    simulation_minutes=24 * 60,   # episode length; minimum is one day
    sample_time=5,                # controller interval (minutes per step)
    patient_name="adolescent#001",
)

obs, info = env.reset(seed=42)
for _ in range(10):
    action = env.action_space.sample()  # [bolus_level, meal_level], 5 levels each
    obs, reward, cost, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

Note the **six-tuple** step return: `(obs, reward, cost, terminated, truncated,
info)`. This is why GlucoSim ships its own `gym_env` registry instead of using
Gymnasium's `make` directly.

A complete runnable example that simulates 24 hours and prints time-in-range and
other glycemic metrics lives in [`examples/basic_rollout.py`](examples/basic_rollout.py):

```bash
python examples/basic_rollout.py --seed 42 --plot cgm.png
```

### Key environment knobs

- `simulation_minutes`: total episode length in minutes (rounded up to >= 1 day).
- `sample_time`: controller interval in minutes per step (default 5).
- `patient_name`: a row from `glucosim/simglucose/params/vpatient_params.csv`
  (cohorts: `child#0XX`, `adolescent#0XX`, `adult#0XX`).
- `patient_overrides`: physiological overrides, see below.

### Observation and action spaces

- **Observation** (14-D `Box`): CGM (mg/dL), bolus insulin-on-board (U),
  carbs-on-board (g), CGM trend, time-of-day (sin/cos), normalized
  time-since-meal/bolus, pending meal buffer, daily meal/bolus counts, time
  until the next scheduled meal, its size, and a pre-bolus-window flag.
- **Action** (`MultiDiscrete([5, 5])`): `[bolus_level, meal_level]` where level
  0 is "do nothing" and levels 1-4 map linearly to the patient's maximum bolus
  (default 10 U) and maximum meal (default 80 g). Action acceptance is gated by
  behavioral rules (safety windows, daily limits, hypo/hyper overrides); the
  `info` dict reports `bolus_accepted` / `meal_accepted` and block reasons.
- **Cost**: a graded danger signal derived from a short glucose forecast; it is
  zero in the 70-180 mg/dL range and grows for hypoglycemia (<70, severe <54)
  and hyperglycemia (>180, severe >250), with hypoglycemia weighted more
  heavily. Episodes terminate early if blood glucose leaves (10, 600) mg/dL.

## Reproducibility

`env.reset(seed=...)` re-creates the simulator's master JAX PRNG key, so a fixed
seed reproduces the full episode (meal scenario, sensor noise, and behavioral
noise) exactly:

```python
obs_a, _ = env.reset(seed=42)   # episode A
obs_b, _ = env.reset(seed=42)   # identical to episode A
```

For CPU/GPU-independent results, force CPU execution with
`JAX_PLATFORMS=cpu` (the test suite does this automatically).

## Patient randomization / generalization

You can generate patient variation **without adding new CSVs** by sampling
`patient_overrides` at env creation time:

- `carb_absorption_scale` (scales `kmax`, `kabs`)
- `insulin_sensitivity_scale` (scales `Vmx`)
- `autobalance_basal_scale` / `autobalance_hepatic_scale` (shift basal steady state)
- `eat_rate_scale` (meal intake rate)

```python
import numpy as np
import glucosim
from glucosim import gym_env as gym

rng = np.random.default_rng(0)
overrides = {
    "carb_absorption_scale": rng.uniform(0.8, 1.2),
    "insulin_sensitivity_scale": rng.uniform(0.8, 1.2),
    "autobalance_basal_scale": rng.uniform(0.85, 1.15),
    "eat_rate_scale": rng.uniform(0.85, 1.15),
}

env = gym.make("t1d-v0", patient_name="adult#001", patient_overrides=overrides)
```

## OmniSafe / CMDP usage (optional)

`glucosim/diabetes_cmdp.py` wraps the simulator as an OmniSafe `CMDP`. It
requires extra dependencies that are **not** installed by default:

```bash
pip install torch omnisafe stable-baselines3
```

```python
from glucosim.diabetes_cmdp import DiabetesEnvs

cmdp = DiabetesEnvs(
    env_id="t1d-v0",
    device="cpu",
    num_envs=1,
    simulation_minutes=24 * 60,
    sample_time=5,
    patient_name="adolescent#001",
)
obs, info = cmdp.reset()
```

Note: OmniSafe's own Python-version support may lag behind this package; the
bundled `safety_gymnasium` compatibility layer is kept in-tree for that reason.

## Why both `gym_env` and `safety_gymnasium`?

Both folders provide Gymnasium-style APIs with costs, but for different consumers:

- `glucosim/gym_env`: a local registry whose `make(...)` returns environments
  with the six-tuple `(obs, reward, cost, terminated, truncated, info)` step API.
- `glucosim/safety_gymnasium`: a Safety-Gymnasium compatibility layer (vector
  envs and wrappers that preserve cost signals) used by OmniSafe-style workflows,
  updated to work with Python >= 3.10.

## Running the tests

```bash
python -m pytest
```

The suite (smoke tests plus blood glucose sanity checks) runs on CPU in well
under a minute. The first reset of a patient configuration JIT-compiles the
simulator and runs a 72-hour basal-only warmup; the result is cached per
configuration within a process.

## Known limitations

- The exercise action channel is currently disabled (the action space is
  `[bolus_level, meal_level]`); exercise dynamics exist in the code but are not
  exposed.
- Episode lengths shorter than one day are rounded up to one day.
- Sensor/behavioral noise is always enabled and is controlled only through the
  random seed; there is no public switch to disable it.
- The physiological model is a research adaptation of the UVA/Padova-style
  ODE model with behavioral layers on top; it has not been re-validated against
  clinical data in this repository.

## Troubleshooting

- **`TypeError: Parameters to Generic[...]`** on import: fixed for
  gymnasium >= 1.3 on the current branch; upgrade GlucoSim.
- **First reset is slow**: JIT compilation plus a 72h warmup run once per
  patient configuration. Subsequent resets reuse a cached warm state.
- **Slow startup on GPU**: cuDNN autotuning can take minutes; set
  `JAX_PLATFORMS=cpu` unless you need GPU throughput.
- **`ImportError` from `glucosim.diabetes_cmdp`**: install the optional
  `torch`, `omnisafe`, and `stable-baselines3` dependencies.

## Models and datasets

Trained models and datasets are available on Hugging Face:
[safe-diabetes-benchmark](https://huggingface.co/safe-diabetes-benchmark).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please run `python -m pytest` before
opening a pull request.

## License

This project is released under the [MIT License](LICENSE). Vendored
Gymnasium / Safety-Gymnasium / OmniSafe code in `glucosim/gym_env` and
`glucosim/safety_gymnasium` retains its original Apache-2.0 notices.

## Citation

If this repository was helpful to your research, please consider citing our work:

```bibtex
@inproceedings{kwon2026safetygeneralizationdistributionshift,
  title={Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed},
  author={Minjae Kwon and Josephine Lamp and Lu Feng},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026},
  url={https://openreview.net/forum?id=kSUGLBHd0T}
}
```
