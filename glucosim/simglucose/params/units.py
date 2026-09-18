"""Units of the implemented simulator (time in minutes).

Gut states 0:3: whole-patient carbohydrate mass, mg.
Glucose states Gp, Gt, Gsc: mg/kg; Gp/Vg is mg/dL.
Insulin states Ip, Il, Isc1, Isc2: pmol/kg; Ip/Vi is pmol/L.
T1D remote states: pmol/L (x1 is an offset); T2D remote states: dimensionless.
Exercise E1: bpm; T_E and E2: min. Secretion Y: mU/min; filtered Gf: mM.

BW: kg. Vg: dL/kg. Vi: L/kg. Gb: mg/dL. Gpb/Gtb: mg/kg.
Ipb/Ilb: pmol/kg. Ib: pmol/L. ke2 and Km0: mg/kg.
Vm0, Fsnc, EGPb, kp1: mg/(kg*min).
Vmx, kp3: (mg/(kg*min))/(pmol/L). kp2: 1/min.
ki, p2u and kinetic rate constants: 1/min.
Raw CSV u2ss: pmol/(kg*min); pump basal: U/hr.
ODE carbohydrate input: g/min; additional insulin input: U/min.
Glucose uses 180 mg/mmol (18 mg/dL per mM); insulin uses 6000 pmol/U.

Empirical gain definitions for the existing equations:
c1, c2: min; alpha_QE: mg/(kg*min^3); beta_ex: mg/(kg*min).
beta_s: mU/(min^2*mM); K_deriv: mU/(min*mM); alpha_s: 1/min.
S_I1/S_I2/S_I3: L/(mU*min); Sb_per_kg: mU/(kg*min).
These conventions do not assert clinical calibration or published-model equivalence.
See PatientParams.UNITS for the parameter registry.
"""
