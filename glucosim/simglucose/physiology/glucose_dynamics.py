"""Compatibility imports for the former combined dynamics module.

Vector fields and deterministic integration live in physiology; stochastic step
orchestration lives in sim. Re-export the original functions themselves so
call signatures, JIT interfaces and legacy imports remain available.
"""
from .vector_fields import _nn, hovorka_t1d, hybrid_t2d
from .integration import _exercise_e2_step
from ..sim.physiology_step import t1d_rk4_step, t2d_rk4_step
