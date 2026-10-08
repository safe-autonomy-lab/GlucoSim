"""Every glucosim module must import, apart from missing optional extras.

Catches stale imports (e.g. the old ``glucobench`` package name) in modules
that no other test happens to import.
"""
import importlib
import pkgutil

import pytest

import glucosim

# Top-level packages that only optional extras provide (see pyproject.toml).
OPTIONAL_PACKAGES = {"torch", "omnisafe", "stable_baselines3"}
# Unused MuJoCo utilities vendored from Safety-Gymnasium need these; glucosim
# does not declare them.
VENDORED_MUJOCO_PACKAGES = {"mujoco", "glfw", "xmltodict"}
SKIPPABLE = OPTIONAL_PACKAGES | VENDORED_MUJOCO_PACKAGES

MODULES = sorted(
    info.name
    for info in pkgutil.walk_packages(glucosim.__path__, prefix="glucosim.")
)


def _missing_package(exc):
    """Top-level package a ModuleNotFoundError in ``exc``'s chain names."""
    while exc is not None:
        if isinstance(exc, ModuleNotFoundError):
            return (exc.name or "").split(".")[0]
        exc = exc.__cause__
    return None


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        missing = _missing_package(exc)
        if missing in SKIPPABLE:
            pytest.skip(f"dependency {missing!r} not installed")
        raise
