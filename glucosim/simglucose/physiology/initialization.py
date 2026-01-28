from typing import Tuple
import dataclasses
import jax.numpy as jnp
from ..core.params import _mgdl_to_mM, EnvParams
from ..core.types import PatientType
from ..physiology.steady_state_solvers import tune_kp1_to_EGPb, tune_Vm0_to_basal, egp0_for_target
from ..physiology.state_builders import init_state_t1d, init_state_t2d


def initialize_patient_state_history(basal_rate_U_hr: float, dt_mins: float, kernel_len: int):
    basal_per_step = (basal_rate_U_hr / 60.0) * dt_mins
    return jnp.full((kernel_len,), basal_per_step, dtype=jnp.float32)

def tune_initial_state(env_params: EnvParams) -> Tuple[EnvParams, jnp.ndarray]:
    """
    Tune the initial state of the patient to the basal state.
    """
    patient_params = env_params.patient_params
    diabetes_type = patient_params.diabetes_type
    if diabetes_type == PatientType.t1d:
        x0 = init_state_t1d(patient_params)
        kp1_new = tune_kp1_to_EGPb(patient_params, x0)
        Vm0_new = tune_Vm0_to_basal(patient_params, x0)
        patient_params = dataclasses.replace(patient_params, Vm0=Vm0_new, kp1=kp1_new)
    elif diabetes_type in [PatientType.t2d, PatientType.t2d_no_pump]:
        # Align the T2D fasting setpoint with the CSV basal glucose by retuning
        # the base EGP term before the burn-in. The 72h warmup otherwise drifts
        # to a hypo equilibrium (~60 mg/dL) because EGP_0 + endogenous secretion
        # slightly undershoot the combined CNS + exchange losses. Use the insulin
        # concentration implied by the pump/endogenous steady state (temp_x0)
        # rather than Ipb alone so we balance against the true basal I level.
        G_target_mM = _mgdl_to_mM(patient_params.Gpb / jnp.maximum(patient_params.Vg, 1e-6))
        temp_x0 = init_state_t2d(patient_params)
        I_target_mU_L = float(temp_x0[5]) / jnp.maximum(patient_params.Vi, 1e-6) / 6.0
        egp0_new = egp0_for_target(patient_params, G_target_mM, I_target_mU_L)
        patient_params = dataclasses.replace(patient_params, EGP_0=egp0_new)
        x0 = init_state_t2d(patient_params)
    else:
        raise ValueError(f"Invalid diabetes type: {diabetes_type}, expected 't1d', 't2d', or 't2d_no_pump'")
    
    env_params = dataclasses.replace(env_params, patient_params=patient_params)
    return env_params, x0
