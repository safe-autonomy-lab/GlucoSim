"""
x0_1, x0_2, x0_3	mg	Carbohydrate amounts in the gut/stomach compartments.
x0_4, x0_5	mg/dL	Glucose concentrations (Plasma, Tissue). Values like 250 are typical for mg/dL.
x0_6, x0_10	pmol	Insulin amounts in plasma and liver compartments.
x0_7, x0_8, x0_9	pmol/L	Insulin concentrations (remote action signals), matching the Ib value.
x0_11, x0_12	pmol	Insulin amounts in the subcutaneous depots.
x0_13	mg/dL	CGM glucose concentration, matching the plasma glucose value.
BW	kg	Body Weight.
EGPb	mg/kg/min	Basal Endogenous Glucose Production rate.
Gb, Gpb, Gtb	mg/dL	Basal glucose concentrations (Blood, Plasma, Tissue).
Ib	pmol/L	Basal Insulin concentration.
Vg	dL/kg	Volume of glucose distribution (deciliters per kilogram).
Vi	L/kg	Volume of insulin distribution (liters per kilogram).
kabs, kmax, kmin	1/min	Rate constants for meal absorption.
k1, k2	1/min	Rate constants for glucose transport between compartments.
ka1, ka2, kd, ksc	1/min	Rate constants for subcutaneous insulin absorption dynamics.
m1, m2, m30, m4,	1/min	Rate constants for insulin kinetics.
m5  min*kg/pmol	Rate constants for insulin kinetics.
ki, p2u	1/min	Rate constants for insulin action.
ke1	1/min	Rate constant for renal glucose excretion.
ke2	mg/dL	Renal threshold for glucose excretion.
Km0	mg/kg	Michaelis-Menten constant (a concentration).
Ipb, Ilb	pmol/kg 	Basal insulin amounts in plasma and liver.
Vmx, Vm0	mg/kg/min per pmol/l	Michaelis-Menten maximum velocity rates for glucose uptake.
Fsnc, Rdb, PCRb, kp1	mg/kg/min	Various rates of glucose production or disposal.
b, d, f, HEb	dimensionless	Fractions or bioavailability parameters.
CL	L/min	Insulin clearance rate.
kp2, kp3	Model Coefficients	Units depend on the exact EGP equation formulation (e.g., (mg/kg/min)/(mg/dL)).
u2ss	U/min	Steady-state basal insulin infusion rate.
isc1ss, isc2ss	pmol	Steady-state subcutaneous insulin amounts.
dosekempt	pmol	A total insulin dose amount (90,000 pmol = 15 U).
"""