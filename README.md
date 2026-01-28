# GlucoSim

JAX-based diabetes simulation environments with cost-aware wrappers for safe RL.

## Dependencies

Install the pinned minimums:

```bash
python -m pip install -r requirements.txt
```

Tested Python version: 3.10

## Why both `gym_env` and `safety_gymnasium`?

Both folders provide Gymnasium-style APIs with costs, but for different consumers:

- `envs/gym_env`: local, lightweight registry + wrappers with a Gym-like API. Use this for day-to-day runs or when you just want a `make(...)` call that returns `(obs, reward, cost, terminated, truncated, info)`.
- `envs/safety_gymnasium`: Safety-Gymnasium compatibility layer used by OmniSafe-style workflows. It is updated to work with Python >= 3.10, but OmniSafe itself is not compatible with Python >= 3.10 as of today.

When OmniSafe catches up to Python >= 3.10, use `envs/diabetes_cmdp.py` as the entry point for OmniSafe algorithms.


## Package install

This repo is laid out as a Python package (`envs/` as the top-level module). To use it as an installed package:

1) Add a minimal `pyproject.toml` (setuptools) at the repo root.  
2) Run:

```bash
python -m pip install -e .
```

If you don't want to package it, you can also run from source by adding the repo root to `PYTHONPATH`.

## Notes

- The `safety_gymnasium` package mirrors Gymnasium wrappers but preserves cost signals.
- The local `gym_env` registry is separate from Gymnasium's global registry; use `from glucobench import gym_env as gym` to get `gym.make(...)` behavior.


## Quickstart (Gym-like usage)

The repo registers `t1d-v0`, `t2d-v0`, and `t2d_no_pump-v0` on import. Use the local gym-like registry:

```python
import glucobench  # registers t1d-v0 / t2d-v0 / t2d_no_pump-v0
from glucobench import gym_env as gym

# Minimum episode length is one day! so even if you set up 12 * 60 (half day) as an episode length, it would convert to one day.
env = gym.make(
    "t1d-v0",
    simulation_minutes=24 * 60,
    sample_time=5,
    patient_name="adolescent#001",
)

obs, info = env.reset()
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, cost, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

Key knobs:
- `simulation_minutes`: total episode length in minutes.
- `sample_time`: controller interval (minutes per step).
- `patient_name`: from `envs/simglucose/params/vpatient_params.csv`.

## OmniSafe / CMDP usage

Use the CMDP wrapper in `envs/diabetes_cmdp.py`:

```python
from glucobench.diabetes_cmdp import DiabetesEnvs

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

This wrapper returns costs in the step API and is the intended entry point for OmniSafe algorithms.


## Patient randomization / generalization

You can generate more patient variation **without adding new CSVs** by sampling `patient_overrides` at env creation time.
These overrides are already supported by `JaxSimEnv` and `create_patient_params`.

Built-in scaling knobs you can randomize:
- `carb_absorption_scale` (scales `kmax`, `kabs`)
- `insulin_sensitivity_scale` (scales `Vmx`)
- `autobalance_basal_scale` / `autobalance_hepatic_scale` (shift basal steady-state)
- `eat_rate_scale` (meal intake rate)

Example:

```python
import numpy as np
import glucobench
from glucobench import gym_env as gym

rng = np.random.default_rng(0)
overrides = {
    "carb_absorption_scale": rng.uniform(0.8, 1.2),
    "insulin_sensitivity_scale": rng.uniform(0.8, 1.2),
    "autobalance_basal_scale": rng.uniform(0.85, 1.15),
    "eat_rate_scale": rng.uniform(0.85, 1.15),
}

env = gym.make(
    "t1d-v0",
    patient_name="adult#001",
    patient_overrides=overrides,
)
```
