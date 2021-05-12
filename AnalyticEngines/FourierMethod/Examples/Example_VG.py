import numpy as np
from AnalyticEngines.FourierMethod.CharesticFunctions import JumpDiffusionCharesticFunction
from functools import partial
from AnalyticEngines.FourierMethod.COSMethod import COSRepresentation
from Instruments.EuropeanInstruments import TypeEuropeanOption

import time

# European option price
k_s = np.array([70, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
f0 = 100
x0 = np.log(f0)
T = 1

# VG parameters
r = 0.1
sigma = 0.18
theta = -0.13
beta = 0.25

# Upper and lower bound for cos integral
a = -8 * np.sqrt(T)
b = 8 * np.sqrt(T)

cf_VG = partial(JumpDiffusionCharesticFunction.get_VG_cf, t=T, x=x0, r=r, sigma=sigma, beta=beta, theta=theta)
start_time = time.time()
cos_price = np.exp(-r * T) * COSRepresentation.get_european_option_price(TypeEuropeanOption.CALL, a, b, 2 ** 14, k_s,
                                                                         cf_VG)
end_time = time.time()
diff_time = end_time - start_time
