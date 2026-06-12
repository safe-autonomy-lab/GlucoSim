# Contributing to GlucoSim

Thanks for your interest in improving GlucoSim. This simulator backs published
research results, so the bar for changes that affect simulation behavior is
deliberately high.

## Ground rules

- **Do not change physiological equations, constants, or reward/cost
  definitions** unless you are fixing a demonstrable bug. If a change alters
  simulated trajectories, say so explicitly in the pull request, explain why,
  and add or update a test that pins the corrected behavior.
- Keep changes small and reviewable. Separate refactoring from behavior changes.
- New features should come with tests and, where user-facing, README updates.

## Development setup

```bash
git clone https://github.com/safe-autonomy-lab/GlucoSim.git
cd GlucoSim
python -m pip install -e .
python -m pip install pytest
```

## Running the tests

```bash
python -m pytest
```

The suite runs on CPU in under a minute (`JAX_PLATFORMS=cpu` is set by
`tests/conftest.py`). All tests must pass before a pull request is merged.

## Reporting issues

Please include:

- the environment id, patient name, and any `patient_overrides`,
- the seed and a minimal script that reproduces the problem,
- your Python, `jax`, and `gymnasium` versions.

## Safety disclaimer

GlucoSim is a research tool. Contributions must not present it as suitable for
clinical decision-making.
