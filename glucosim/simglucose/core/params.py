from typing import Dict, Optional, Literal, Iterable, Tuple, ClassVar

import dataclasses
import numpy as np
import jax.numpy as jnp
import pandas as pd
import os
import hashlib
import logging
import jax


from ..core.types import PatientType
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
    u2ss: float  # U/min (per-patient steady pump basal proxy)
    Vg:   float  # dL/kg
    Vi:   float  # L/kg
    V_G_L: float # L (absolute glucose distribution vol; may be derived)
    V_I_L: float # L (absolute insulin distribution vol; may be derived)
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
    ki:  float   # mg/kg/min (legacy delay parameter for EGP/uptake couplings)

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
    c1: float        # -
    c2: float        # -
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
    K_deriv: float = 0.0                 # mU/mM
    alpha_s: float = 0.05                 # 1/min
    beta_s: float = 1.0                   # mU/(min·mM)
    h: float = 6.0                        # mM
    Sb_per_kg: float = 0.02               # mU/(kg·min)
    V_I: float = 0.05                     # L (will be scaled by BW externally if used)
    k_a1: float = 0.006                   # 1/min
    k_a2: float = 0.06                    # 1/min
    k_a3: float = 0.12                    # 1/min
    # IMPORTANT: S_I* units INCLUDE per-minute:
    S_I1: float = 0.00051                 # (1/min) per (mU/L)  == L/(mU·min)
    S_I2: float = 0.0081                  # (1/min) per (mU/L)
    S_I3: float = 0.00520                 # (1/min) per (mU/L)
    beta_ex: float = 0.2                 # mg/kg/min
    alpha_QE: float = 0.004                 # 1/min
    m6: float = 0.0                       # - (HE dynamics intercept if used)
    V_G: float = 0.0                      # L (absolute glucose distribution vol)
    EGP_0: float = 0.0                    # mmol/min (base EGP if using x3-suppressed EGP path)
    F_cns0: float = 0.0                   # mmol/min (brain usage; preferred over Fsnc)
    k12: float = 0.0                      # 1/min (optional intercompartment)
    F01: float = 0.0                      # mmol/min (non–insulin-dependent uptake)

    # -------------------- Units/Descriptions registry --------------------
    UNITS: ClassVar[Dict[str, Unit]] = {
        # Core
        "BW":"kg","EGPb":"mg/kg/min","Gb":"mg/dL","Ib":"pmol/L","u2ss":"U/min",
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
        "age":"y","HR0":"bpm","tau_HR":"min","alpha_HR":"-","n_power":"-","c1":"-","c2":"-",
        "tau_ex":"min","tau_in":"min",
        # Behavior
        "eat_rate":"g/min","meal_acceptance_prob":"-","use_pump":"bool","bolus_acceptance_prob":"-",
        "beta_cell_function":"-","insulin_resistance_factor":"-","meal_safe_window":"min",
        "bolus_safe_window":"min","max_bolus_U":"U","max_meal_g":"g","max_exercise_min":"min","basal":"U/hr",
        # T2D hybrid
        "A_G":"-","tau_D":"min","MwG_mg_per_mmol":"mg/mmol","tau_S":"min","gamma":"1/min",
        "K_deriv":"mU/mM","alpha_s":"1/min","beta_s":"mU/(min·mM)","h":"mM",
        "Sb_per_kg":"mU/(kg·min)","V_I":"L","k_a1":"1/min","k_a2":"1/min","k_a3":"1/min",
        "S_I1":"(1/min)/(mU/L)","S_I2":"(1/min)/(mU/L)","S_I3":"(1/min)/(mU/L)",
        "beta_ex":"mg/kg/min","alpha_QE":"1/min","m6":"-","V_G":"L","EGP_0":"mmol/min",
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
                                   "beta_s","h","Sb_per_kg","V_I","k_a1","k_a2","k_a3",
                                   "S_I1","S_I2","S_I3","beta_ex","alpha_QE","m6","V_G",
                                   "EGP_0","F_cns0","k12","F01"]},
    }

    # -------------------- Pretty-print helpers --------------------
    def _iter_params(self) -> Iterable[Tuple[str, float]]:
        for f in dataclasses.fields(self):
            if f.name in ("UNITS","DESCRIPTIONS","CATEGORIES"):
                continue
            yield f.name, getattr(self, f.name)

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
        unused = ["CL","Rdb","PCRb","Fsnc","m5","V_G_L","V_I_L"]  # update as wiring changes
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

    # Circadian modulation (applied as additive delta on Gp post-step)
    circadian_egp_amp_rel: float = 0.10    # ±% modulation of hepatic drive
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


def load_patient_parameters_from_csv(csv_path: str) -> Dict[str, Dict]:
    """
    Load patient parameters from CSV file and return as dictionary.
    
    Args:
        csv_path: Path to the vpatient_params.csv file
        
    Returns:
        Dictionary mapping patient names to their parameter dictionaries
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Patient parameters CSV file not found: {csv_path}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded patient data for {len(df)} patients from {csv_path}")
        
        # Validate required columns
        required_columns = ['Name', 'BW', 'EGPb', 'Gb', 'Ib', 'u2ss', 'Vg', 'Vi', 'Ipb', 'Ilb', 'Gpb', 'Gtb']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # Convert to dictionary format
        patient_data = {}
        for _, row in df.iterrows():
            patient_name = row['Name']
            if pd.isna(patient_name) or patient_name == '':
                continue  # Skip empty rows
                
            # Extract parameters from CSV row
            params = {
                # Core Physiological
                'BW': float(row['BW']),
                'EGPb': float(row['EGPb']), # mg/kg/min
                'Gb': float(row['Gb']), # mg/dL
                'Ib': float(row['Ib']),
                'u2ss': float(row['u2ss']),
                'Vg': float(row['Vg']), # dL/kg
                'Vi': float(row['Vi']), # L/kg
                'Ipb': float(row['Ipb']),
                'Ilb': float(row['Ilb']),
                'Gpb': float(row['Gpb']),
                'Gtb': float(row['Gtb']),
                'Fsnc': float(row.get('Fsnc', 1.0)),  # mg/kg/min
                
                # Meal Absorption
                'kmax': float(row['kmax']),
                'kmin': float(row['kmin']),
                'kabs': float(row['kabs']),
                'b': float(row['b']),
                'd': float(row['d']),
                'f': float(row['f']),
                
                # Insulin Kinetics
                'ka1': float(row['ka1']),
                'ka2': float(row['ka2']),
                'kd': float(row['kd']),
                'ksc': float(row['ksc']),
                'm1': float(row['m1']),
                'm2': float(row['m2']),
                'm30': float(row['m30']),
                'm4': float(row['m4']),
                'm5': float(row['m5']),
                'CL': float(row['CL']),
                'HEb': float(row['HEb']),
                
                # Insulin Action
                'Vmx': float(row['Vmx']),
                'Vm0': float(row['Vm0']),
                'Km0': float(row['Km0']),
                'p2u': float(row['p2u']),
                'ki': float(row['ki']),
                
                # Glucose Kinetics
                'kp1': float(row['kp1']),
                'kp2': float(row['kp2']),
                'kp3': float(row['kp3']),
                'k1': float(row['k1']),
                'k2': float(row['k2']),
                'ke1': float(row['ke1']),
                'ke2': float(row['ke2']),
                'Rdb': float(row['Rdb']),
                'PCRb': float(row['PCRb']),
            }
            
            patient_data[patient_name] = params
            logger.debug(f"Loaded parameters for patient: {patient_name}")
            
        return patient_data
        
    except Exception as e:
        logger.error(f"Error loading patient parameters from CSV: {e}")
        raise ValueError(f"Failed to parse patient parameters CSV: {e}")


def create_patient_params(patient_name: str, 
                         csv_path: Optional[str] = None,
                         diabetes_type: Optional[Literal["t1d", "t2d", "t2d_no_pump"]] = None,
                         **override_params) -> PatientParams:
    """
    Create PatientParams instance for a specific patient from CSV data.
    
    Args:
        patient_name: Name of the patient (e.g., 'adolescent#001', 'adult#005')
        csv_path: Path to CSV file. If None, uses default path
        diabetes_type: Type of diabetes adaptation ("t1d", "t2d", "t2d_no_pump")
        **override_params: Additional parameters to override defaults
        
    Returns:
        PatientParams instance configured for the specified patient
        
    Raises:
        ValueError: If patient not found in CSV or invalid parameters
        
    Examples:
        # Standard patient (default behavior)
        params = create_patient_params("adult#005")
        
        # T1D patient
        params = create_patient_params("adolescent#001", diabetes_type="t1d")
        
        # T2D patient with pump
        params = create_patient_params("adult#003", diabetes_type="t2d")
        
        # T2D patient without pump
        params = create_patient_params("adult#007", diabetes_type="t2d_no_pump")
    """
    # Extract calibration-only options that are not PatientParams fields
    autobalance_enabled = override_params.pop("autobalance_enabled", True)
    autobalance_basal_scale = override_params.pop("autobalance_basal_scale", 1.0)
    autobalance_hepatic_scale = override_params.pop("autobalance_hepatic_scale", 1.0)
    carb_absorption_scale = override_params.pop("carb_absorption_scale", 1.0)
    insulin_sensitivity_scale = override_params.pop("insulin_sensitivity_scale", 1.0)
    eat_rate_scale = override_params.pop("eat_rate_scale", 1.0)

    # Default CSV path if not provided
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, '..', 'params', 'vpatient_params.csv')
        
    # Load patient data from CSV
    patient_data = load_patient_parameters_from_csv(csv_path)
    
    if patient_name not in patient_data:
        available_patients = list(patient_data.keys())
        raise ValueError(f"Patient '{patient_name}' not found in CSV. Available patients: {available_patients}")
    
    # Get base parameters for the patient
    base_params = patient_data[patient_name].copy()
    # PD tuning: shrink glucose distribution volume and conserved masses
    if 'Vg' in base_params and np.isfinite(base_params['Vg']):
        base_params['Vg'] = float(base_params['Vg'])
    if 'Gpb' in base_params and np.isfinite(base_params['Gpb']):
        base_params['Gpb'] = float(base_params['Gpb'])
    if 'Gtb' in base_params and np.isfinite(base_params['Gtb']):
        base_params['Gtb'] = float(base_params['Gtb'])
    age_ranges = {
        'child': (0, 12),      # 0 to 12 years
        'adolescent': (13, 19), # 13 to 19 years
        'adult': (20, 80)      # 20 to 80 years (reasonable upper limit for adults)
    }
    age = 35
    HR0 = 70.0
    tau = 10.0
    # Customize parameters based on age
    if patient_name.split('#')[0] in age_ranges:
        min_age, max_age = age_ranges[patient_name.split('#')[0]]
        stable_seed = int(hashlib.md5(patient_name.encode("ascii", "ignore")).hexdigest()[:8], 16)
        age = min_age + (stable_seed % (max_age - min_age + 1))
        # HR0: resting heart rate, higher for younger ages
        HR0 = 70 + 50 * max(0, (18 - age) / 18)
        # tau_HR, tau_ex, tau_in: time constants, smaller (faster recovery) for younger ages
        tau = 10 - 0.2 * (20 - age) if age < 20 else 10 + 0.1 * (age - 20)
        tau_HR = tau
        tau_ex = tau
        tau_in = tau
        
    # Add default values for parameters not in CSV
    defaults = {
        # Eating behavior
        'eat_rate': 10.0,
        'meal_acceptance_prob': ACCEPTANCE_PROB_DEFAULT,
        'bolus_acceptance_prob': ACCEPTANCE_PROB_DEFAULT,
        'exercise_acceptance_prob': ACCEPTANCE_PROB_DEFAULT,
        'V_G_L': (base_params['Vg'] * base_params['BW']) / 10.0,  # dL/kg to L
        'V_I_L': base_params['Vi'] * base_params['BW'],           #
        
        # Pump settings
        'use_pump': False,
        
        # Endogenous insulin secretion
        'beta_cell_function': 0.3,
        'insulin_resistance_factor': 2.5,
        
        # Safety windows
        'meal_safe_window': 60.0,  # minutes
        'bolus_safe_window': 60.0,  # minutes
        'exercise_safe_window': 1440.0,  # minutes (long enough to ignore)
        
        # Limits
        'max_meal_g': 80.0,
        'max_bolus_U': 80.0 / 10.0, # we maintain 1 U/h insulin covers 10g of carbs
        'max_exercise_min': 90.0,
        
        # Basal rate
        'basal': 0.0,  # U/h

        # Exercise
        'age': age,
        'HR0': HR0,
        'alpha_HR': 0.01,
        'n_power': 1.0,
        'c1': 0.01,
        'c2': 0.01,
        'tau_HR': tau_HR,
        'tau_ex': tau_ex,
        'tau_in': tau_in,
        'beta_ex': 0.2,
        'alpha_QE': 0.004,
    }
    
    # Merge defaults with CSV data
    for key, default_value in defaults.items():
        if key not in base_params:
            base_params[key] = default_value

    # Ensure patient identification is present before creating dataclass
    base_params['diabetes_type'] = getattr(PatientType, diabetes_type)
    # Convert base parameters to PatientParams for proper T2D adaptation
    temp_params = PatientParams(**base_params)
    if diabetes_type == "t1d":
        patient_params = adapt_params_for_t1d(
            temp_params,
            autobalance_enabled=autobalance_enabled,
            autobalance_basal_scale=autobalance_basal_scale,
            autobalance_hepatic_scale=autobalance_hepatic_scale,
            carb_absorption_scale=carb_absorption_scale,
            insulin_sensitivity_scale=insulin_sensitivity_scale,
            eat_rate_scale=eat_rate_scale,
        )
    elif diabetes_type == "t2d":
        patient_params = adapt_params_for_t2d(
            temp_params,
            carb_absorption_scale=carb_absorption_scale,
            insulin_sensitivity_scale=insulin_sensitivity_scale,
            eat_rate_scale=eat_rate_scale,
        )
    elif diabetes_type == "t2d_no_pump":
        patient_params = adapt_params_for_t2d_no_pump(
            temp_params,
            carb_absorption_scale=carb_absorption_scale,
            insulin_sensitivity_scale=insulin_sensitivity_scale,
            eat_rate_scale=eat_rate_scale,
        )
    else:
        raise ValueError(f"Invalid diabetes_type: {diabetes_type}. Must be 't1d', 't2d', or 't2d_no_pump'")

    # Apply overrides at the very end to ensure they take precedence
    if override_params:
        patient_params = dataclasses.replace(patient_params, **override_params)
        logger.info(f"Applied {len(override_params)} parameter overrides: {list(override_params.keys())}")

    logger.info(f"Applied {diabetes_type} adaptations to patient {patient_name}")

    return patient_params


def autobalance_basal_t1d(
    p: PatientParams,
    basal_scale: float = 1.0,
    hepatic_scale: float = 1.0,
) -> PatientParams:
    """
    Make the T1D default a true steady state for the provided (Gpb,Gtb,Ipb) and current p.
    Solves:
      1) peripheral utilization  U_id(Gtb, Ipb) = k1*Gpb - k2*Gtb  (=> Vmx)
      2) hepatic balance         EGP(Gpb,Ipb)   = Fsnc + E_renal + k1*Gpb - k2*Gtb (=> kp1)
    """
    # ---- Targets from basal masses ----
    U_star = p.k1 * p.Gpb - p.k2 * p.Gtb                      # mg/kg/min
    frac   = p.Gtb / (p.Km0 + p.Gtb)                          # dimensionless MM fraction

    # ---- Basal insulin concentration (pmol/L) ----
    I_p_pmol_L = p.Ipb / p.Vi

    # ---- (1) Peripheral: choose Vm0 (keep) and solve Vmx to hit U_star ----
    # Ensure Vmx is positive by capping Vm0 at a fraction of total uptake
    # If Vm0 from CSV is too high, insulin has no room to act.
    # Standard basal insulin-independent uptake is usually 1.0-2.5 mg/kg/min.
    Vm0_target = min(p.Vm0, U_star / frac * 0.75)
    
    # If Vm0_target is still too high or close to total, force it down.
    # We want Vmx * I_p > 0. So Vm0 + Vmx*I = U_star/frac.
    # Vmx = (U_star/frac - Vm0) / I_p.
    # Let's just use the adjusted Vm0.
    Vm0_new = Vm0_target * basal_scale
    Vmx_new = max((U_star/frac - Vm0_new) / max(I_p_pmol_L, 1e-6), 1e-5)

    # ---- (2) Hepatic: pick kp1 so dGp=0 at basal (no meal, no exercise) ----
    # Renal loss at basal:
    E_renal = p.ke1 * (p.Gpb - p.ke2) if p.Gpb > p.ke2 else 0.0
    # CNS usage (legacy Fsnc, mg/kg/min):
    F_cns = p.Fsnc

    # We want: 0 = EGP + Ra - F_cns - E_renal - k1*Gp + k2*Gt  (Ra=0 at basal)
    # With EGP = kp1 - kp2*Gp - kp3*I_conc (I_conc=Ipb/Vi):
    I_conc = I_p_pmol_L
    kp1_new = (p.kp2 * p.Gpb + p.kp3 * I_conc
               + F_cns + E_renal + p.k1 * p.Gpb - p.k2 * p.Gtb)
    kp1_new = kp1_new * hepatic_scale

    return dataclasses.replace(p, Vm0=Vm0_new, Vmx=Vmx_new, kp1=kp1_new)


def autobalance_basal_t2d(params: PatientParams) -> PatientParams:
    U_star = params.k1 * params.Gpb - params.k2 * params.Gtb
    frac = params.Gtb / max(params.Km0 + params.Gtb, 1e-6)

    I_p_pmol_L = params.Ipb / params.Vi
    I_p_mU_L   = I_p_pmol_L / 6.0

    Vm0_new = 1.0
    Vmx_new = max((U_star / max(frac, 1e-6) - Vm0_new) / max(I_p_pmol_L, 1e-6), 0.0)

    x3_basal = (params.S_I3 / max(params.k_a3, 1e-6)) * I_p_mU_L
    F_cns_mgkgmin = (params.F_cns0 * 180.0) / params.BW
    E_renal = params.ke1 * max(params.Gpb - params.ke2, 0.0)

    one_minus = max(1.0 - x3_basal, 1e-6)
    EGP0_mgkgmin = (U_star + F_cns_mgkgmin + E_renal) / one_minus
    EGP0_mmolmin = EGP0_mgkgmin * params.BW / 180.0

    return dataclasses.replace(params, Vm0=Vm0_new, Vmx=Vmx_new, EGP_0=EGP0_mmolmin)


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
    params = base_params
    
    logger.info("Adapting parameters for Type 1 Diabetes")
    
    # T1D-specific adjustments
    params = dataclasses.replace(params,
        # No endogenous insulin production
        beta_cell_function=0.0,

        # Normal insulin sensitivity
        insulin_resistance_factor=1.0,
        
        # Typically use pump for better control
        use_pump=True,
        
        # Higher adherence to insulin therapy (survival depends on it)
        bolus_acceptance_prob=ACCEPTANCE_PROB_DEFAULT,
        
        # More careful meal planning
        meal_acceptance_prob=ACCEPTANCE_PROB_DEFAULT,
        
        # Tighter safety margins due to no endogenous backup
        meal_safe_window=60.0,  # minutes
        bolus_safe_window=60.0,  # minutes

        # Proxy basal rate from t1dpatient.py
        basal=params.BW * 0.011,  # U/h
    )
    # Autobalance the basal rate, with optional weakening to expose harsher dynamics
    if autobalance_enabled:
        params = autobalance_basal_t1d(
            params,
            basal_scale=autobalance_basal_scale,
            hepatic_scale=autobalance_hepatic_scale,
        )

    # Gut appearance tweaks for calibration (stronger absorption => sharper spikes)
    if carb_absorption_scale != 1.0:
        params = dataclasses.replace(
            params,
            kmax=params.kmax * carb_absorption_scale,
            kabs=params.kabs * carb_absorption_scale,
        )
    if eat_rate_scale != 1.0:
        params = dataclasses.replace(
            params,
            eat_rate=params.eat_rate * eat_rate_scale,
        )

    # Insulin action scaling for harsher “no bolus” behaviour
    if insulin_sensitivity_scale != 1.0:
        params = dataclasses.replace(
            params,
            Vmx=params.Vmx * insulin_sensitivity_scale,
        )

    return params

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
    logger.info("Adapting parameters for Type 2 Diabetes (with pump)")

    # Default T2D scaling factors
    default_config = {
        'Ib_factor': 1.25,      # Residual beta-cell function (25% increase)
        'Ipb_factor': 1.25,     # Plasma insulin at basal state
        'HEb_factor': 0.85,     # Reduced hepatic insulin clearance
        'BW_factor': 1.15,      # Higher body weight for T2D
    }
    config = config or default_config

    # Start with original T1D parameters for proper conversion
    temp_params = base_params

    # Apply physiological scaling before T2D conversion
    temp_params = dataclasses.replace(
        temp_params,
        BW=temp_params.BW * config['BW_factor'],
        Ib=temp_params.Ib * config['Ib_factor'],
        Ipb=temp_params.Ipb * config['Ipb_factor'],
        Ilb=temp_params.Ilb * config['Ipb_factor'],  # Scale liver insulin to maintain equilibrium ratio
        HEb=temp_params.HEb * config['HEb_factor'],
    )

    # Convert to T2D format using comprehensive conversion logic
    t2d_params = patient_to_t2d_params(temp_params, use_dynamic_HE=False)
    clearance_mU_per_min = (t2d_params.m2 + t2d_params.m4) * (t2d_params.Ib * t2d_params.Vi * t2d_params.BW / 6.0)
    total_insulin_need_U_per_hr = (clearance_mU_per_min * 60) / 1000

    # 2. Calculate the insulin the patient produces naturally (in U/hr)
    # This is the systemic portion of the basal endogenous secretion.
    portal_secretion_U_per_hr = (t2d_params.Sb_per_kg * t2d_params.BW * 60) / 1000
    endogenous_supply_U_per_hr = portal_secretion_U_per_hr * (1 - t2d_params.HEb)

    # 3. The pump's basal rate is the difference
    basal_rate = max(0.0, total_insulin_need_U_per_hr - endogenous_supply_U_per_hr)

    # Apply T2D-specific behavioral and physiological parameters
    t2d_params = dataclasses.replace(
        t2d_params,
        # Residual beta-cell function (25-30% remaining)
        beta_cell_function=0.25,

        # Moderate insulin resistance
        insulin_resistance_factor=2.5,

        # Use pump for better glucose control
        use_pump=True,

        # For now, full acceptance to simplify learning
        bolus_acceptance_prob=ACCEPTANCE_PROB_DEFAULT,

        # For now, full acceptance to simplify learning
        meal_acceptance_prob=ACCEPTANCE_PROB_DEFAULT,

        # Longer safety windows due to residual insulin production
        meal_safe_window=60.0,  # minutes
        bolus_safe_window=60.0,  # minutes

        # Higher limits due to insulin resistance
        max_bolus_U=10.0,
        max_meal_g=100.0,

        # Proxy basal rate
        basal=basal_rate,  # U/h
    )

    # Re-balance basal fluxes once insulin sensitivity scaling and pump basal are final.
    t2d_params = autobalance_basal_t2d(t2d_params)

    # Final steady-state sync: recompute Ipb/Ilb after all scaling so residuals vanish
    Ip_ss, Il_ss = _steady_state_insulin_from_Sb(t2d_params)
    t2d_params = dataclasses.replace(t2d_params, Ipb=Ip_ss, Ilb=Il_ss)

    # 4. Final steady-state sync
    # We have changed Sb_per_kg, HEb, and potentially other factors.
    # The scaled Ipb/Ilb we started with are just estimates.
    # We must solve for the TRUE steady state implied by these new parameters to avoid residuals.
    Ip_ss, Il_ss = _steady_state_insulin_from_Sb(t2d_params)
    t2d_params = dataclasses.replace(t2d_params, Ipb=Ip_ss, Ilb=Il_ss)

    # Safety validation for T2D
    assert 0.2 <= t2d_params.beta_cell_function <= 0.3, f"Invalid T2D beta-cell function: {t2d_params.beta_cell_function}"
    assert 2.0 <= t2d_params.insulin_resistance_factor <= 3.0, f"Invalid T2D insulin resistance: {t2d_params.insulin_resistance_factor}"
    assert t2d_params.BW > 0, f"Invalid body weight after T2D adjustment: {t2d_params.BW}"

    logger.debug(f"T2D adaptations: beta_cell={t2d_params.beta_cell_function:.2f}, "
                f"IR_factor={t2d_params.insulin_resistance_factor:.1f}, BW={t2d_params.BW:.1f}")

    # Optional scaling knobs for generalization stress tests
    if carb_absorption_scale != 1.0:
        t2d_params = dataclasses.replace(
            t2d_params,
            kmax=t2d_params.kmax * carb_absorption_scale,
            kabs=t2d_params.kabs * carb_absorption_scale,
        )
    if eat_rate_scale != 1.0:
        t2d_params = dataclasses.replace(
            t2d_params,
            eat_rate=t2d_params.eat_rate * eat_rate_scale,
        )
    if insulin_sensitivity_scale != 1.0:
        t2d_params = dataclasses.replace(
            t2d_params,
            Vmx=t2d_params.Vmx * insulin_sensitivity_scale,
        )

    return t2d_params


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
    # Start with T2D pump adaptations
    params = adapt_params_for_t2d(
        base_params,
        config,
        carb_absorption_scale=carb_absorption_scale,
        insulin_sensitivity_scale=insulin_sensitivity_scale,
        eat_rate_scale=eat_rate_scale,
    )

    logger.info("Adapting parameters for Type 2 Diabetes (no pump)")

    # No-pump specific adjustments
    params = dataclasses.replace(
        params,
        # No insulin pump
        use_pump=False,
        beta_cell_function=0.3,

        # Higher insulin resistance (injection site variability)
        insulin_resistance_factor=2.8,

        # For now, full acceptance to simplify learning
        bolus_acceptance_prob=ACCEPTANCE_PROB_DEFAULT,

        # For now, full acceptance to simplify learning
        meal_acceptance_prob=ACCEPTANCE_PROB_DEFAULT,

        # Longer safety windows due to manual injections
        meal_safe_window=60.0,  # minutes
        bolus_safe_window=60.0,  # minutes

        # Higher limits to account for less precise delivery
        max_bolus_U=10.0,
        max_meal_g=100.0,
        K_deriv=30.0,

        # No pump, no basal
        basal=0.0,  # U/h baseline
    )

    Ip_ss, Il_ss = _steady_state_insulin_from_Sb(params)
    params = dataclasses.replace(params, Ipb=Ip_ss, Ilb=Il_ss)
    params = autobalance_basal_t2d(params)

    # Safety validation for T2D no-pump
    assert not params.use_pump, "SAFETY: T2D no-pump must not use pump"
    assert params.insulin_resistance_factor >= 2.5, f"T2D no-pump IR factor too low: {params.insulin_resistance_factor}"
    assert params.max_bolus_U <= 25.0, f"Unsafe max bolus for T2D no-pump: {params.max_bolus_U}"

    logger.debug(f"T2D no-pump adaptations: IR_factor={params.insulin_resistance_factor:.1f}, "
                f"max_bolus={params.max_bolus_U:.1f}U, adherence={params.bolus_acceptance_prob:.2f}")

    # Re-apply scaling after the no-pump autobalance step
    if carb_absorption_scale != 1.0:
        params = dataclasses.replace(
            params,
            kmax=params.kmax * carb_absorption_scale,
            kabs=params.kabs * carb_absorption_scale,
        )
    if eat_rate_scale != 1.0:
        params = dataclasses.replace(
            params,
            eat_rate=params.eat_rate * eat_rate_scale,
        )
    if insulin_sensitivity_scale != 1.0:
        params = dataclasses.replace(
            params,
            Vmx=params.Vmx * insulin_sensitivity_scale,
        )

    return params


def _mgdl_to_mM(g_mgdl: float) -> float:
    """Convert glucose from mg/dL to mM."""
    return g_mgdl * 0.0555

def _mM_to_mgdl(G_mM: float) -> float:
    """Convert glucose from mM to mg/dL."""
    return G_mM * 18.0

def _mmolmin_from_mgkgmin(val_mgkgmin: float, BW: float) -> float:
    """Convert mass rate per kg to molar rate for patient."""
    return (val_mgkgmin * BW) / 180.0

def _as_liters_from_Vg(Vg_raw: float, BW: float) -> float:
    """Convert glucose distribution volume to liters."""
    return float(Vg_raw * BW / 10.0)

def _as_liters_from_Vi(Vi_raw: float, BW: float) -> float:
    """Convert insulin distribution volume to liters."""
    return float(Vi_raw * BW)

def _mu_per_l(Ib_raw: float) -> float:
    """Convert basal insulin concentration."""
    return float(Ib_raw / 6.0)


def _steady_state_insulin_from_Sb(params: PatientParams) -> tuple[float, float]:
    """
    Compute fasting plasma and liver insulin masses implied by the current basal secretion.
    Mirrors the linear two-compartment equilibrium used for Hovorka/UVA calibration.
    """
    S_endog = 6.0 * float(params.Sb_per_kg)
    if S_endog <= 0.0:
        return 0.0, 0.0

    m1 = float(params.m1)
    m2 = float(params.m2)
    m4 = float(params.m4)
    m30 = float(params.m30)

    denom_sec = (m1 + m30) - (m2 * m1) / max(m2 + m4, 1e-8)
    denom_sec = max(denom_sec, 1e-8)

    Il_ss = S_endog / denom_sec
    Ip_ss = (m1 / max(m2 + m4, 1e-8)) * Il_ss
    return float(Ip_ss), float(Il_ss)


def _almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    """Helper for floating-point comparisons that respects exact CSV values."""
    return float(np.abs(a - b)) <= tol


def _units_ok(value: float) -> bool:
    """Simple finiteness/positivity guard for derived parameters."""
    return np.isfinite(value) and value >= 0.0


def patient_to_t2d_params(base_params: PatientParams, use_dynamic_HE: bool = False) -> PatientParams:
    """
    Derive the additional fields the hybrid T2D ODE expects without mutating the CSV
    contract (glucose in mg/kg, insulin in pmol/kg, volumes in dL/kg or L/kg).

    Args:
        base_params: PatientParams populated directly from the CSV (plus behavior fields).
        use_dynamic_HE: Whether to model time-varying hepatic extraction.

    Returns:
        A new PatientParams with the original base fields intact and the T2D-only
        derived quantities expressed in the units required by the hybrid ODE.
    """
    BW = float(base_params.BW)
    Vg_dL_per_kg = float(base_params.Vg)
    Vi_L_per_kg = float(base_params.Vi)

    # These conversions only create model-side conveniences; base CSV entries stay mg/kg & pmol/kg.
    V_G = _as_liters_from_Vg(Vg_dL_per_kg, BW)   # L
    V_I = _as_liters_from_Vi(Vi_L_per_kg, BW)    # L

    # Basal glucose concentration in mg/dL is still (mg/kg)/(dL/kg); convert to mM for hybrid secretion.
    Gpb_mg_per_kg = float(base_params.Gpb)
    G_conc_mg_per_dL = Gpb_mg_per_kg / max(Vg_dL_per_kg, 1e-6)
    h_mM = _mgdl_to_mM(G_conc_mg_per_dL)

    # Gut absorption dynamics remain in 1/min; we only surface tau_D for the hybrid model.
    kgut_avg = 0.5 * (float(base_params.kmax) + float(base_params.kmin))
    k_eff = max(1e-6, min(kgut_avg, float(base_params.kabs)))
    tau_D = 0.5 / k_eff

    # SC insulin absorption time constant derived from existing rate constants.
    ka1 = float(base_params.ka1)
    ka2 = float(base_params.ka2)
    kd = float(base_params.kd)
    tau_S = 0.5 * (1.0 / max(ka1 + kd, 1e-6) + 1.0 / max(ka2, 1e-6))

    # Hepatic extraction handling mirrors the original conversion but never rewrites CSV fields.
    m1 = float(base_params.m1)
    m2 = float(base_params.m2)
    m4 = float(base_params.m4)
    HEb = float(jnp.clip(base_params.HEb, 0.0, 0.95))
    m3_b = (HEb * m1) / (1.0 - HEb + 1e-9)
    m5 = float(base_params.m5) if use_dynamic_HE else 0.0
    m6 = HEb

    # Map basal insulin from pmol/L-equivalent (CSV) into mU/L for secretion bookkeeping.
    Ib_mU_L = _mu_per_l(float(base_params.Ib))
    I_p_b = Ib_mU_L * V_I
    K_b = max(0.0, ((m3_b + m1) * (m2 + m4) / m1) - m2)
    S_t_b = max(0.0, K_b * I_p_b)
    # The result cannot be negative.
    S_sys_target_U_per_hr = 0.4

    # 5. Use the derived S_sys_target to calculate the final Sb_per_kg.
    # This is the formula from your discussion, which converts the systemic target
    # back into a per-kilogram portal secretion rate.
    Sb_per_kg = (S_sys_target_U_per_hr * 1000 / 60) / ((1 - HEb) * BW)

    # Insulin sensitivity terms can be scaled by the IR factor without breaking dimensionality.
    ir_factor = max(float(base_params.insulin_resistance_factor), 1e-6)
    S_I1 = base_params.S_I1 / ir_factor
    S_I2 = base_params.S_I2 / ir_factor
    S_I3 = base_params.S_I3 / ir_factor

    # k12 mirrors k2 but lives in the hybrid state space.
    k12 = float(base_params.k2)

    # Endogenous glucose production and CNS uptake remain mg/kg/min in the CSV.
    # We convert to mmol/min for the hybrid equations here.
    x3_b = (S_I3 / max(float(base_params.k_a3), 1e-8)) * Ib_mU_L
    EGP_b_mmol_per_min = _mmolmin_from_mgkgmin(float(base_params.EGPb), BW)
    EGP_0 = float(EGP_b_mmol_per_min / max(1.0 - x3_b, 1e-6))
    F_cns0 = _mmolmin_from_mgkgmin(float(base_params.Fsnc), BW)

    updated = dataclasses.replace(
        base_params,
        # Hybrid model expects these derived fields.
        A_G=0.9,
        tau_D=tau_D,
        MwG_mg_per_mmol=180.0,
        tau_S=tau_S,
        gamma=base_params.gamma,
        K_deriv=base_params.K_deriv,
        alpha_s=base_params.alpha_s,
        beta_s=base_params.beta_s,
        h=h_mM,
        Sb_per_kg=Sb_per_kg,
        m5=m5,
        m6=m6,
        V_I=V_I,
        k_a1=base_params.k_a1,
        k_a2=base_params.k_a2,
        k_a3=base_params.k_a3,
        S_I1=S_I1,
        S_I2=S_I2,
        S_I3=S_I3,
        V_G=V_G,
        EGP_0=EGP_0,
        F_cns0=F_cns0,
        k12=k12,
        beta_ex=base_params.beta_ex,
        alpha_QE=base_params.alpha_QE,
    )

    _assert_csv_contract_preserved(base_params, updated)
    _assert_t2d_units(updated)

    return updated


def _assert_csv_contract_preserved(original: PatientParams, updated: PatientParams) -> None:
    """
    Make sure the CSV-derived quantities stay identical so unit conversions happen at
    the ODE boundary rather than in parameter packing.
    """
    base_fields = (
        'BW', 'EGPb', 'Gb', 'Ib', 'u2ss',
        'Vg', 'Vi', 'V_G_L', 'V_I_L',
        'Ipb', 'Ilb', 'Gpb', 'Gtb',
        'Fsnc', 'ke1', 'ke2',
        'kp1', 'kp2', 'kp3',
        'k1', 'k2',
        'Vm0', 'Km0', 'Vmx',
    )
    for field in base_fields:
        original_val = getattr(original, field)
        updated_val = getattr(updated, field)
        if isinstance(original_val, (float, int)):
            assert _almost_equal(float(original_val), float(updated_val)), (
                f"CSV base field '{field}' was altered during T2D conversion "
                f"({original_val} -> {updated_val})."
            )
        else:
            assert original_val == updated_val, (
                f"CSV base field '{field}' was altered during T2D conversion."
            )


def _assert_t2d_units(params: PatientParams) -> None:
    """Run quick sanity checks on the hybrid T2D parameters."""
    assert _units_ok(params.V_G), "Expected V_G in liters and non-negative."
    assert _units_ok(params.V_I), "Expected V_I in liters and non-negative."
    assert _units_ok(params.EGP_0), "EGP_0 must be finite mmol/min."
    assert _units_ok(params.F_cns0), "F_cns0 must be finite mmol/min."
    assert np.isfinite(params.h) and params.h >= 0.0, "Setpoint h must be in mM."
    assert np.isfinite(params.Sb_per_kg) and params.Sb_per_kg >= 0.0, "Basal secretion must be >= 0."


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
        diabetes_type: Type of diabetes adaptation ("t1d", "t2d", "t2d_no_pump")
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
    logger.info(f"Creating environment parameters for patient: {patient_name}")
    
    cohort_key = patient_name.split('#')[0].lower()
    meal_mu, meal_sigma = get_meal_profile_for_cohort(cohort_key)

    # Create patient-specific parameters
    patient_params = create_patient_params(
        patient_name=patient_name,
        csv_path=csv_path,
        diabetes_type=diabetes_type,
        **patient_overrides
    )
    
    # Create other parameter structures
    dia_steps = 360 # 6 hours * 60 min/hr
    noise_config = NoiseConfig()
    
    # dt_mins=1 because the simulation kernel resolution is 1 minute
    # duration_hours=6 matches dia_steps=360
    iob_decay_kernel, insulin_act_kernel = create_insulin_kernel(dt_mins=1, duration_hours=6)
    
    # Convert to numpy for storage in EnvParams (JAX arrays are also fine, but casting ensures consistency)
    iob_decay_kernel = np.array(iob_decay_kernel)
    insulin_act_kernel = np.array(insulin_act_kernel)

    insulin_kernel_5 = np.array(insulin_act_kernel).reshape((-1, 5)).sum(axis=1)

    # Create environment parameters
    env_params = EnvParams(
        patient_params=patient_params,
        sample_time=sample_time,
        simulation_minutes=simulation_minutes,
        dia_steps=dia_steps,
        insulin_kernel=tuple(insulin_act_kernel),
        insulin_kernel_5=tuple(insulin_kernel_5),
        iob_kernel=tuple(iob_decay_kernel),
        noise_config=noise_config,
        patient_name=patient_name,
        meal_amount_mu=jnp.asarray(meal_mu, dtype=jnp.float32),
        meal_amount_sigma=jnp.asarray(meal_sigma, dtype=jnp.float32),
    )
    
    logger.info(f"Environment parameters created successfully for {patient_name}")
    return env_params


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
