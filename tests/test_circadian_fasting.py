"""Regression test: the circadian EGP swing must stay physiologically bounded.

The fasting glucose equilibrium is highly sensitive to the EGP intercept kp1
(kp2 is small), so the relative circadian modulation of kp1 maps to a much
larger absolute glucose swing. With the original amplitude (0.10) the
circadian component alone pulled the fasting start from the ~130 mg/dL tuned
equilibrium down to ~80 mg/dL. After reducing the amplitude to 0.03 the
circadian-only start should sit much closer to the equilibrium.

This test isolates circadian by disabling the other noise sources, so it pins
the circadian amplitude specifically and is independent of the separate
process-noise contribution to the fasting level.
"""
import dataclasses

import numpy as np

import glucosim
from glucosim import gym_env as gym


def _circadian_only_config(base):
    """A NoiseConfig with every stochastic source off, circadian left on."""
    return dataclasses.replace(
        base,
        meal_logn_sigma=0.0, bolus_logn_sigma=0.0, missed_bolus_prob=0.0,
        pump_basal_rel_sigma=0.0,
        sigma_gpgt_dL=0.0, sigma_ip=0.0,
        cgm_bias_mgdl=0.0, cgm_scale_bias=0.0, cgm_rw_sigma_bias=0.0,
        cgm_obs_sigma_mgdl=0.0, cgm_dropout_prob=0.0,
        # circadian_egp_amp_rel left at its default
    )


def test_circadian_swing_is_bounded():
    env = gym.make("t1d-v0", simulation_minutes=24 * 60, sample_time=5,
                   patient_name="adult#001")
    u = env.unwrapped
    u.env_params = dataclasses.replace(
        u.env_params, noise_config=_circadian_only_config(u.env_params.noise_config)
    )
    type(u)._warmup_cache.clear()  # force a fresh warmup for this noise config
    obs, _ = env.reset(seed=0)
    cgm0 = float(obs[0])
    env.close()
    # No-circadian fasting equilibrium for adult#001 is ~130 mg/dL. With the
    # reduced amplitude the circadian-only start must stay within ~25 mg/dL of
    # it (was ~80 mg/dL, i.e. ~50 mg/dL low, at the original 0.10 amplitude).
    assert cgm0 >= 105.0, (
        f"circadian-only fasting start {cgm0:.1f} mg/dL is too far below the "
        "~130 mg/dL equilibrium; circadian amplitude may be too large again"
    )
