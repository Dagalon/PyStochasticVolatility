import numpy as np
from Tools.Types import EULER_SCHEME_TYPE
from MC_Engines.GenericSDE.EuropeanGreeksSimulation import get_malliavin_greeks_bs_flat

t = 10.0
s0 = 100.0
k = 90.0
r = 0.02
q = 0.0
sigma = 0.35

no_paths = 10000
no_steps = 2

get_malliavin_greeks_bs_flat(s0, t, no_steps, no_paths, r, q, sigma,
                             lambda x: np.maximum((x - k), 0), EULER_SCHEME_TYPE.LOG_NORMAL)
