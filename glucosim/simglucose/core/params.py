from typing import Dict, Optional, Literal, Iterable, Tuple, ClassVar

import dataclasses
import numpy as np
import jax.numpy as jnp
import logging
import jax


from ..core.types import PatientType
# Re-export the existing loader API for callers importing core.params.
from .patient_loader import load_patient_parameters_from_csv
from .conversion import _as_liters_from_Vg, _as_liters_from_Vi
from ..physiology import calibration
# Keep existing import paths available; construction uses the owners directly.
from ..sim.scenario_gen import get_meal_profile_for_cohort
from ..physiology.kernels import create_insulin_kernel

logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL + 1)

# Global toggle for behavioral acceptance (can be tuned centrally)
# Set <1.0 to reintroduce stochastic acceptance
# It's usually hard with varying acceptance probabilities for each patient
# Controllers struggle to learn a good timing
ACCEPTANCE_PROB_DEFAULT = 1.0  

Unit = str
Desc = str
Category = str

# Define frozen dataclasses for nested params
@dataclasses.dataclass(frozen=True)
class PatientParams:
    """
    Patient and model parameters.

    Notes on a few important unit choices:
    - Glucose 'mass' states Gp,Gt are mg/kg (NOT mg/dL). Concentration is Gb = Gp / Vg with Vg in dL/kg.
    - Insulin 'mass' states Ip,Il are pmol/kg. Concentration is Ib = Ip / Vi with Vi in L/kg.
    - Insulin-effect gains S_I* expect insulin in mU/L and produce a rate in 1/min:
        dx/dt = -k_a * x + S_I * I_mU_per_L
      => S_I units are (1/min) per (mU/L) == L/(mU·min).
    - beta_ex is a direct glucose flow term in mg/kg/min.
    """

    # -------------------- Core Physiological --------------------
    diabetes_type: int
    BW:   float  # kg
    EGPb: float  # mg/kg/min
    Gb:   float  # mg/dL
    Ib:   float  # pmol/L
    u2ss: float  # pmol/kg/min (raw CSV basal infusion)
    Vg:   float  # dL/kg
    Vi:   float  # L/kg
    Ipb:  float  # pmol/kg
    Ilb:  float  # pmol/kg
    Gpb:  float  # mg/kg
    Gtb:  float  # mg/kg
    Fsnc: float  # mg/kg/min (legacy CNS usage if not using F_cns0)

    # -------------------- Meal Absorption --------------------
    kmax: float  # 1/min
    kmin: float  # 1/min
    kabs: float  # 1/min
    b:    float  # -
    d:    float  # -
    f:    float  # -

    # -------------------- Insulin Kinetics --------------------
    ka1: float   # 1/min
    ka2: float   # 1/min
    kd:  float   # 1/min
    ksc: float   # 1/min (CGM filter)
    m1:  float   # 1/min
    m2:  float   # 1/min
    m30: float   # 1/min
    m4:  float   # 1/min
    m5:  float   # min·kg/pmol (used only in some HE dynamics forms)
    CL:  float   # TBD (not used)
    HEb: float   # - (hepatic extraction at basal)

    # -------------------- Insulin Action --------------------
    Vmx: float   # (mg/kg/min) per (pmol/L)
    Vm0: float   # mg/kg/min
    Km0: float   # mg/kg
    p2u: float   # 1/min (legacy, if using Ki/p2u style effects)
    ki:  float   # 1/min (insulin-effect filter rate)

    # -------------------- Glucose Kinetics --------------------
    kp1: float   # mg/kg/min (EGP at zero G and I) — only if using linear EGP form
    kp2: float   # 1/min (hepatic glucose effectiveness) — linear EGP form
    kp3: float   # (mg/kg/min) per (pmol/L) — hepatic insulin action — linear EGP form
    k1:  float   # 1/min
    k2:  float   # 1/min
    ke1: float   # 1/min
    ke2: float   # mg/kg
    Rdb: float   # TBD (not used)
    PCRb: float  # TBD (not used)

    # -------------------- Exercise --------------------
    age: float       # years
    HR0: float       # bpm
    tau_HR: float    # min
    alpha_HR: float  # -
    n_power: float   # -
    c1: float        # min
    c2: float        # min
    tau_ex: float    # min
    tau_in: float    # min

    # -------------------- Behaviors / policy scaffolding --------------------
    eat_rate: float
    meal_acceptance_prob: float
    use_pump: bool
    bolus_acceptance_prob: float
    exercise_acceptance_prob: float
    beta_cell_function: float
    insulin_resistance_factor: float
    meal_safe_window: float
    bolus_safe_window: float
    exercise_safe_window: float
    max_bolus_U: float
    max_meal_g: float
    max_exercise_min: float
    basal: float  # U/hr typically outside; in code you convert to U/min then pmol/kg/min

    # -------------------- T2D hybrid-specific --------------------
    A_G: float = 0.9                      # -
    tau_D: float = 40.0                   # min
    MwG_mg_per_mmol: float = 180.0        # mg/mmol
    tau_S: float = 55.0                   # min
    gamma: float = 0.2                    # 1/min (secretion dyn if used)
    K_deriv: float = 0.0                 # mU/(min·mM)
    alpha_s: float = 0.05                 # 1/min
    beta_s: float = 1.0                   # mU/(min²·mM); forcing gain, not steady secretion gain
    h: float = 6.0                        # mM
    Sb_per_kg: float = 0.02               # mU/(kg·min)
    k_a1: float = 0.006                   # 1/min
    k_a2: float = 0.06                    # 1/min
    k_a3: float = 0.12                    # 1/min
    # IMPORTANT: S_I* units INCLUDE per-minute:
    S_I1: float = 0.00051                 # (1/min) per (mU/L)  == L/(mU·min)
    S_I2: float = 0.0081                  # (1/min) per (mU/L)
    S_I3: float = 0.00520                 # (1/min) per (mU/L)
    beta_ex: float = 0.2                 # mg/kg/min
    alpha_QE: float = 0.004                 # mg/(kg·min³), with E2 measured in minutes
    m6: float = 0.0                       # - (HE dynamics intercept if used)
    EGP_0: float = 0.0                    # mmol/min (base EGP if using x3-suppressed EGP path)
    F_cns0: float = 0.0                   # mmol/min (brain usage; preferred over Fsnc)
    k12: float = 0.0                      # 1/min (optional intercompartment)
    F01: float = 0.0                      # mmol/min (non–insulin-dependent uptake)

    @property
    def V_G_L(self):
        """Total glucose distribution volume [L], derived from effective inputs."""
        return _as_liters_from_Vg(self.Vg, self.BW)

    @property
    def V_I_L(self):
        """Total insulin distribution volume [L], derived from effective inputs."""
        return _as_liters_from_Vi(self.Vi, self.BW)

    def __setstate__(self, state):
        if any(name in state for name in ('V_G', 'V_I', 'V_G_L', 'V_I_L')):
            raise ValueError('Obsolete PatientParams pickle contains stored derived volumes; '
                             'reconstruct from authoritative BW, Vg, and Vi inputs')
        self.__dict__.update(state)

    # -------------------- Units/Descriptions registry --------------------
    UNITS: ClassVar[Dict[str, Unit]] = {
        # Core
        "BW":"kg","EGPb":"mg/kg/min","Gb":"mg/dL","Ib":"pmol/L","u2ss":"pmol/kg/min",
        "Vg":"dL/kg","Vi":"L/kg","V_G_L":"L","V_I_L":"L","Ipb":"pmol/kg","Ilb":"pmol/kg",
        "Gpb":"mg/kg","Gtb":"mg/kg","Fsnc":"mg/kg/min",
        # Meal
        "kmax":"1/min","kmin":"1/min","kabs":"1/min","b":"-","d":"-","f":"-",
        # Insulin kin
        "ka1":"1/min","ka2":"1/min","kd":"1/min","ksc":"1/min","m1":"1/min","m2":"1/min",
        "m30":"1/min","m4":"1/min","m5":"min·kg/pmol","CL":"(TBD)","HEb":"-",
        # Insulin action
        "Vmx":"(mg/kg/min)/(pmol/L)","Vm0":"mg/kg/min","Km0":"mg/kg","p2u":"1/min","ki":"1/min",
        # Glucose kin
        "kp1":"mg/kg/min","kp2":"1/min","kp3":"(mg/kg/min)/(pmol/L)",
        "k1":"1/min","k2":"1/min","ke1":"1/min","ke2":"mg/kg","Rdb":"(TBD)","PCRb":"(TBD)",
        # Exercise
        "age":"y","HR0":"bpm","tau_HR":"min","alpha_HR":"-","n_power":"-","c1":"min","c2":"min",
        "tau_ex":"min","tau_in":"min",
        # Behavior
        "eat_rate":"g/min","meal_acceptance_prob":"-","use_pump":"bool","bolus_acceptance_prob":"-",
        "beta_cell_function":"-","insulin_resistance_factor":"-","meal_safe_window":"min",
        "bolus_safe_window":"min","max_bolus_U":"U","max_meal_g":"g","max_exercise_min":"min","basal":"U/hr",
        # T2D hybrid
        "A_G":"-","tau_D":"min","MwG_mg_per_mmol":"mg/mmol","tau_S":"min","gamma":"1/min",
        "K_deriv":"mU/(min·mM)","alpha_s":"1/min","beta_s":"mU/(min²·mM)","h":"mM",
        "Sb_per_kg":"mU/(kg·min)","k_a1":"1/min","k_a2":"1/min","k_a3":"1/min",
        "S_I1":"(1/min)/(mU/L)","S_I2":"(1/min)/(mU/L)","S_I3":"(1/min)/(mU/L)",
        "beta_ex":"mg/kg/min","alpha_QE":"mg/(kg·min³)","m6":"-","EGP_0":"mmol/min",
        "F_cns0":"mmol/min","k12":"1/min","F01":"mmol/min",
        "patient_name":"-"
    }

    DESCRIPTIONS: ClassVar[Dict[str, Desc]] = {
        # Only the most critical ones are filled verbosely; extend as you like.
        "EGPb":"Basal endogenous glucose production; if using x3-suppressed EGP, prefer EGP_0 instead.",
        "Gb":"Basal plasma glucose concentration; Gb = Gp/Vg.",
        "Ib":"Basal plasma insulin concentration; Ib = Ip/Vi.",
        "Vg":"Glucose distribution volume per kg.",
        "Vi":"Insulin distribution volume per kg.",
        "V_G_L":"Total glucose distribution volume: Vg * BW / 10 [L].",
        "V_I_L":"Total insulin distribution volume: Vi * BW [L].",
        "Ipb":"Basal plasma insulin mass.",
        "Ilb":"Basal liver insulin mass.",
        "Gpb":"Basal plasma+tightly equilibrated glucose mass.",
        "Gtb":"Basal tissue glucose mass (slow).",
        "Fsnc":"CNS/erythrocyte glucose usage (legacy). Prefer F_cns0 path.",
        "Vmx":"Peripheral insulin action slope vs plasma insulin conc.",
        "Vm0":"Basal peripheral max utilization.",
        "Km0":"Glucose MM half-saturation (mass space).",
        "kp1":"Linear EGP offset; remove if using nonlinear EGP with x3.",
        "kp2":"Hepatic glucose effectiveness (linear EGP form).",
        "kp3":"Hepatic insulin action slope (linear EGP form).",
        "ke2":"Renal threshold (mass space).",
        "HEb":"Basal hepatic extraction fraction.",
        "S_I1":"Insulin-effect gain (1/min)/(mU/L) for x1.",
        "S_I2":"Insulin-effect gain (1/min)/(mU/L) for x2.",
        "S_I3":"Insulin-effect gain (1/min)/(mU/L) for x3 (EGP suppression).",
        "beta_ex":"Exercise-driven extra uptake term added to flows.",
        "EGP_0":"Base EGP (mmol/min) used by x3-suppressed EGP path.",
        "F_cns0":"Brain/CNS glucose usage (mmol/min) converted to mg/kg/min by 180/BW."
    }

    CATEGORIES: ClassVar[Dict[str, Category]] = {
        **{k:"Core" for k in ["patient_name","BW","EGPb","Gb","Ib","u2ss","Vg","Vi","V_G_L","V_I_L","Ipb","Ilb","Gpb","Gtb","Fsnc"]},
        **{k:"Meal" for k in ["kmax","kmin","kabs","b","d","f"]},
        **{k:"InsulinKinetics" for k in ["ka1","ka2","kd","ksc","m1","m2","m30","m4","m5","CL","HEb"]},
        **{k:"InsulinAction" for k in ["Vmx","Vm0","Km0","p2u","ki"]},
        **{k:"GlucoseKinetics" for k in ["kp1","kp2","kp3","k1","k2","ke1","ke2","Rdb","PCRb"]},
        **{k:"Exercise" for k in ["age","HR0","tau_HR","alpha_HR","n_power","c1","c2","tau_ex","tau_in"]},
        **{k:"Behavior" for k in ["eat_rate","meal_acceptance_prob","use_pump","bolus_acceptance_prob","exercise_acceptance_prob",
                                  "beta_cell_function","insulin_resistance_factor","meal_safe_window",
                                  "bolus_safe_window","exercise_safe_window","max_bolus_U","max_meal_g","max_exercise_min","basal"]},
        **{k:"T2DHybrid" for k in ["A_G","tau_D","MwG_mg_per_mmol","tau_S","gamma","K_deriv","alpha_s",
                                   "beta_s","h","Sb_per_kg","k_a1","k_a2","k_a3",
                                   "S_I1","S_I2","S_I3","beta_ex","alpha_QE","m6",
                                   "EGP_0","F_cns0","k12","F01"]},
    }

    # -------------------- Pretty-print helpers --------------------
    def _iter_params(self) -> Iterable[Tuple[str, float]]:
        for f in dataclasses.fields(self):
            if f.name in ("UNITS","DESCRIPTIONS","CATEGORIES"):
                continue
            yield f.name, getattr(self, f.name)
        for name in ("V_G_L", "V_I_L"):
            yield name, getattr(self, name)

    def to_table(self, markdown: bool = False, only_category: Optional[str] = None) -> str:
        rows = []
        header = ("Name","Value","Units","Category","Description")
        rows.append(header)
        for name, val in self._iter_params():
            cat = self.CATEGORIES.get(name,"Other")
            if only_category and cat != only_category:
                continue
            unit = self.UNITS.get(name,"(TBD)")
            desc = self.DESCRIPTIONS.get(name,"")
            rows.append((name, repr(val), unit, cat, desc))

        # Column widths
        widths = [max(len(r[i]) for r in rows) for i in range(5)]

        def fmt(r):
            return "  ".join(s.ljust(w) for s, w in zip(r, widths))

        if not markdown:
            out = [fmt(rows[0]), "-" * (sum(widths) + 8)]
            out += [fmt(r) for r in rows[1:]]
            return "\n".join(out)

        # Markdown table
        md = []
        md.append("| " + " | ".join(rows[0]) + " |")
        md.append("| " + " | ".join("-" * w for w in widths) + " |")
        for r in rows[1:]:
            md.append("| " + " | ".join(r) + " |")
        return "\n".join(md)

    def describe(self, markdown: bool = False) -> str:
        """Return a full parameter table with units and brief descriptions."""
        return self.to_table(markdown=markdown)

    def list_probably_unused(self) -> str:
        """Heuristic list of params not wired in the current hybrid T2D code path."""
        unused = ["CL","Rdb","PCRb","Fsnc","m5"]  # update as wiring changes
        return ", ".join(p for p in unused if hasattr(self, p))

# Convenience free functions
def print_params_with_units(p: PatientParams, markdown: bool = False, only_category: Optional[str]=None) -> None:
    print(p.to_table(markdown=markdown, only_category=only_category))

def params_to_markdown(p: PatientParams, only_category: Optional[str]=None) -> str:
    return p.to_table(markdown=True, only_category=only_category)

@dataclasses.dataclass(frozen=True)
class PumpParams:
    U2PMOL: int = 6000
    inc_bolus: float = 0.1   # Smallest bolus increment (pmol/min)
    max_bolus: float = 10.0  # Max bolus (U)
    min_bolus: float = 0.0   # Min bolus (U)
    inc_basal: float = 0.1   # Smallest basal increment (pmol/min)
    max_basal: float = 5.0   # Max basal rate (U/hr)
    min_basal: float = 0.0   # Min basal rate (U/hr)

@dataclasses.dataclass(frozen=True)
class NoiseConfig:
    # Action-level adherence
    meal_logn_sigma: float = 0.20          # log-normal sd for meal stream (g/min)
    bolus_logn_sigma: float = 0.12         # log-normal sd for bolus (U/min)
    missed_bolus_prob: float = 0.02        # Bernoulli miss
    pump_basal_rel_sigma: float = 0.03     # multiplicative noise on pump basal (as action delta)

    # Circadian modulation (applied as additive delta on Gp post-step).
    # NOTE: the fasting glucose equilibrium is highly sensitive to the EGP
    # intercept kp1 (kp2 is small), so this relative modulation of kp1 maps to a
    # much larger absolute glucose swing. At 0.10 the daily swing was ~50 mg/dL
    # and episodes started near 80 mg/dL (circadian trough at t=0). 0.03 keeps a
    # realistic dawn-phenomenon swing (~15 mg/dL). See tests/test_circadian_fasting.py.
    circadian_egp_amp_rel: float = 0.03    # ±relative modulation of hepatic drive
    circadian_phase_min: float = 4 * 60.0
    circadian_period_min: float = 24 * 60.0

    # Fast process noise (state), per sqrt(min)
    sigma_gpgt_dL: float = 1.0             # mg/dL / sqrt(min) — turned into mg/kg via Vg
    sigma_ip: float = 0.8                  # pmol/kg / sqrt(min)

    # OU option (for smoother exchange noise)
    use_ou: bool = False
    ou_theta: float = 1 / 15.0             # 1/min
    ou_sigma_dL: float = 1.0               # stationary sd in mg/dL

    # CGM observation (used in rollouts)
    cgm_bias_mgdl: float = 0.0
    cgm_scale_bias: float = 0.0
    cgm_rw_sigma_bias: float = 0.002       # random-walk on multiplicative scale
    cgm_obs_sigma_mgdl: float = 1.5
    cgm_dropout_prob: float = 0.01

    enable: bool = True


@dataclasses.dataclass(frozen=True, eq=False)
class EnvParams:
    patient_params: PatientParams
    sample_time: int
    simulation_minutes: int
    dia_steps: int
    insulin_kernel: jnp.ndarray
    insulin_kernel_5: jnp.ndarray
    iob_kernel: jnp.ndarray
    noise_config: NoiseConfig
    patient_name: str
    meal_amount_mu: jnp.ndarray
    meal_amount_sigma: jnp.ndarray


def create_patient_params(patient_name: str, 
                         csv_path: Optional[str] = None,
                         diabetes_type: Optional[Literal["t1d", "t2d", "t2d_no_pump"]] = None,
                         **override_params) -> PatientParams:
    """
    Create PatientParams instance for a specific patient from CSV data.
    
    Args:
        patient_name: Name of the patient (e.g., 'adolescent#001', 'adult#005')
        csv_path: Path to CSV file. If None, uses default path
        diabetes_type: Type of diabetes adaptation ("t1d", "t2d", "t2d_no_pump").
            Omitted or None selects "t1d".
        **override_params: BW (final kg), Vg (dL/kg), Vi (L/kg), Gpb (mg/kg),
            Fsnc and EGPb (mg/kg/min), and insulin_resistance_factor describe
            effective physiology before conversion/calibration. T1D resistance
            must be 1; T2D resistance must be finite and positive. T1D kp1/Vm0
            and T2D h/F_cns0/Sb_per_kg/S_I1/S_I2/S_I3/EGP_0 are owned outputs
            and cannot be overridden here. k1/k2/Km0/ke1/ke2 also resolve before
            calibration for all types; Gtb/kp2/kp3 do so only for T1D;
            HEb/m1/m2/m30/m4/k_a3 do so for both T2D variants, and Ib only for
            pump T2D. Their existing units and numerical handling are unchanged.
            V_G/V_I/V_G_L/V_I_L are derived volumes and cannot be overridden;
            set BW, Vg, and Vi instead. Other fields retain late replacement, including T2D Gtb and
            no-pump Ib. The resolved use_pump=False requires basal=0.
            autobalance_enabled controls only T1D factory calibration; reset
            initialization still tunes its type-specific outputs.
        
    Returns:
        PatientParams instance configured for the specified patient
        
    Raises:
        ValueError: If patient not found in CSV or invalid parameters
        
    Examples:
        # Standard T1D patient (default behavior)
        params = create_patient_params("adult#005")
        
        # T1D patient
        params = create_patient_params("adolescent#001", diabetes_type="t1d")
        
        # T2D patient with pump
        params = create_patient_params("adult#003", diabetes_type="t2d")
        
        # T2D patient without pump
        params = create_patient_params("adult#007", diabetes_type="t2d_no_pump")
    """
    from .parameter_builder import _build_patient_from_overrides
    return _build_patient_from_overrides(
        patient_name, csv_path, diabetes_type, ACCEPTANCE_PROB_DEFAULT, override_params
    )


def autobalance_basal_t1d(
    p: PatientParams,
    basal_scale: float = 1.0,
    hepatic_scale: float = 1.0,
) -> PatientParams:
    """Compatibility entry point for the T1D factory calibration."""
    return calibration.autobalance_basal_t1d(p, basal_scale, hepatic_scale)


def autobalance_basal_t2d(params: PatientParams) -> PatientParams:
    """Compatibility entry point for the T2D factory calibration."""
    return calibration.autobalance_basal_t2d(params)


def adapt_params_for_t1d(
    base_params: PatientParams,
    autobalance_enabled: bool = True,
    autobalance_basal_scale: float = 1.0,
    autobalance_hepatic_scale: float = 1.0,
    carb_absorption_scale: float = 1.0,
    insulin_sensitivity_scale: float = 1.0,
    eat_rate_scale: float = 1.0,
) -> PatientParams:
    """
    Adapt patient parameters for Type 1 Diabetes.
    
    T1D characteristics:
    - No beta-cell function (no endogenous insulin)
    - Normal insulin sensitivity (no resistance)
    - Requires external insulin for all needs
    - Typically uses insulin pump or multiple daily injections
    
    Args:
        base_params: Base patient parameters dictionary
        
    Returns:
        Adapted parameters for T1D patient
    """
    from .configuration import BuildOptions
    from .parameter_builder import build_t1d

    options = BuildOptions(
        acceptance_probability=ACCEPTANCE_PROB_DEFAULT,
        autobalance_enabled=autobalance_enabled,
        autobalance_basal_scale=autobalance_basal_scale,
        autobalance_hepatic_scale=autobalance_hepatic_scale,
        carb_absorption_scale=carb_absorption_scale,
        insulin_sensitivity_scale=insulin_sensitivity_scale,
        eat_rate_scale=eat_rate_scale,
    )
    return build_t1d(base_params, options)

def adapt_params_for_t2d(
    base_params: PatientParams,
    config: Optional[Dict] = None,
    carb_absorption_scale: float = 1.0,
    insulin_sensitivity_scale: float = 1.0,
    eat_rate_scale: float = 1.0,
) -> PatientParams:
    """
    Adapt patient parameters for Type 2 Diabetes with insulin pump.

    This function uses the comprehensive parameter conversion from test_sim2.py
    to properly adapt T1D parameters for use with the T2D hybrid_2d model.

    T2D characteristics:
    - Residual beta-cell function (20-30% remaining)
    - Insulin resistance (2.5-2.8x normal)
    - Higher body weight
    - Proper T2D parameter conversion for hybrid_2d model

    Args:
        base_params: PatientParams instance to adapt
        config: Optional configuration for T2D adjustments

    Returns:
        Adapted PatientParams for T2D patient with pump
    """
    from .configuration import BuildOptions
    from .parameter_builder import build_t2d

    options = BuildOptions(
        acceptance_probability=ACCEPTANCE_PROB_DEFAULT,
        carb_absorption_scale=carb_absorption_scale,
        insulin_sensitivity_scale=insulin_sensitivity_scale,
        eat_rate_scale=eat_rate_scale,
    )
    return build_t2d(base_params, config, options)


def adapt_params_for_t2d_no_pump(
    base_params: PatientParams,
    config: Optional[Dict] = None,
    carb_absorption_scale: float = 1.0,
    insulin_sensitivity_scale: float = 1.0,
    eat_rate_scale: float = 1.0,
) -> PatientParams:
    """
    Adapt patient parameters for Type 2 Diabetes without insulin pump.

    T2D No-Pump characteristics:
    - Same physiological changes as T2D with pump
    - Manual insulin injections only
    - Even lower adherence rates
    - Less precise insulin delivery
    - Higher insulin resistance due to injection site issues

    Args:
        base_params: PatientParams instance to adapt
        config: Optional configuration for T2D adjustments

    Returns:
        Adapted PatientParams for T2D patient without pump
    """
    from .configuration import BuildOptions
    from .parameter_builder import build_t2d_no_pump

    options = BuildOptions(
        acceptance_probability=ACCEPTANCE_PROB_DEFAULT,
        carb_absorption_scale=carb_absorption_scale,
        insulin_sensitivity_scale=insulin_sensitivity_scale,
        eat_rate_scale=eat_rate_scale,
    )
    return build_t2d_no_pump(base_params, config, options)


# Preserve established helper imports; implementation ownership is conversion.
from .conversion import (
    _mgdl_to_mM, _mM_to_mgdl, _mmolmin_from_mgkgmin,
    _mu_per_l,
    _almost_equal, _units_ok,
)


def _steady_state_insulin_from_Sb(params: PatientParams) -> tuple[float, float]:
    """Compatibility entry point for the legacy factory insulin calibration."""
    return calibration._steady_state_insulin_from_Sb(params)



def patient_to_t2d_params(base_params: PatientParams, use_dynamic_HE: bool = False) -> PatientParams:
    """Compatibility entry point for T2D unit conversion."""
    from . import conversion
    return conversion.patient_to_t2d_params(base_params, use_dynamic_HE)


def _assert_csv_contract_preserved(original: PatientParams, updated: PatientParams) -> None:
    from . import conversion
    return conversion._assert_csv_contract_preserved(original, updated)


def _assert_t2d_units(params: PatientParams) -> None:
    from . import conversion
    return conversion._assert_t2d_units(params)


def create_env_params(patient_name: str = "adolescent#001",
                     csv_path: Optional[str] = None,
                     diabetes_type: Optional[Literal["t1d", "t2d", "t2d_no_pump"]] = None,
                     simulation_minutes: int = 24 * 60,
                     sample_time: int = 5,
                     **patient_overrides) -> EnvParams:
    """
    Create environment parameters with patient-specific configuration.
    
    Args:
        patient_name: Name of patient from CSV (e.g., 'adolescent#001', 'adult#005')
        csv_path: Path to patient parameters CSV file
        diabetes_type: Type of diabetes adaptation ("t1d", "t2d", "t2d_no_pump").
            Omitted or None selects "t1d".
        simulation_minutes: Length of simulation in minutes
        sample_time: Sampling time in minutes
        **patient_overrides: Additional patient parameter overrides
        
    Returns:
        EnvParams configured for the specified patient
        
    Examples:
        # Use default patient (adolescent#001)
        env_params = create_env_params()
        
        # Use specific patient with T1D adaptations
        env_params = create_env_params("adolescent#002", diabetes_type="t1d")
        
        # Use T2D patient with pump
        env_params = create_env_params("adult#005", diabetes_type="t2d")
        
        # Use T2D patient without pump
        env_params = create_env_params("adult#007", diabetes_type="t2d_no_pump")
        
        # Use specific patient with custom overrides
        env_params = create_env_params("child#003", diabetes_type="t1d", 
                                      max_bolus_U=5.0, meal_acceptance_prob=0.95)
    """
    from .parameter_builder import build_env_params
    return build_env_params(
        patient_name, csv_path, diabetes_type, simulation_minutes, sample_time,
        ACCEPTANCE_PROB_DEFAULT, patient_overrides
    )


def _register_dataclass_pytree(cls, static_field_names):
    def flatten(obj):
        children = []
        aux = {}
        for f in dataclasses.fields(obj):
            val = getattr(obj, f.name)
            if f.name in static_field_names:
                aux[f.name] = val
            else:
                children.append(val)
        return children, aux

    def unflatten(aux, children):
        # We need to map children back to their fields in the correct order
        field_names = [f.name for f in dataclasses.fields(cls)]
        dynamic_names = [n for n in field_names if n not in static_field_names]
        
        # Combine aux and children
        kwargs = aux.copy()
        for name, child in zip(dynamic_names, children):
            kwargs[name] = child
        return cls(**kwargs)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)

# Register the classes
_register_dataclass_pytree(PatientParams, {'diabetes_type', 'use_pump'})
_register_dataclass_pytree(NoiseConfig, {'use_ou', 'enable'})
_register_dataclass_pytree(EnvParams, {'sample_time', 'simulation_minutes', 'dia_steps', 'patient_name'})
