"""JAX-based glucose-insulin physiology simulator.

The user-facing environments (``t1d-v0``, ``t2d-v0``, ``t2d_no_pump-v0``) are
registered in :mod:`glucosim.__init__`; import :mod:`glucosim` and use
``glucosim.gym_env.make(...)`` to create them.
"""
