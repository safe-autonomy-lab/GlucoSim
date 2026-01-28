from typing import NamedTuple

# --------------------------
# State vector (16 states)
# 0 D1 1 D2 (Meal absorption)
# 2 S1 3 S2 (SC insulin)
# 4 I_po 5 Y (Insulin secretion)
# 6 I_l 7 I_p (Insulin kinetics)
# 8 x1 9 x2 10 x3 (Insulin effects)
# 11 E1 12 T_E 13 E2 (Exercise)
# 14 Q1 15 Q2 (Glucose)
# --------------------------

# ControllerAction, this is normalized action space, degree of action recommendations
class ControllerAction(NamedTuple):
    bolus: float
    meal: float
    exercise: float

# This is actualy patient action space, which is not normalized, meal (g) and bolus (U)
class Action(NamedTuple):
    bolus: float
    meal: float
    exercise: float

class PatientType:
    t1d = 0
    t2d = 1
    t2d_no_pump = 2


