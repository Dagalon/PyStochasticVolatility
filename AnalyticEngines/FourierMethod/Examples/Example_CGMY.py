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

# CGMY parameters
r = 0.1
sigma = 0.2
C = 1.0
G = 5.0
M = 5.0
Y = 0.5

# Upper and lower bound for cos integral
a = -8 * np.sqrt(T)
b = 8 * np.sqrt(T)

cf_CGMY = partial(JumpDiffusionCharesticFunction.get_CGMYB_cf, t=T, x=x0, r=r, sigma=sigma, C=C, G=G, M=M, Y=Y)
start_time = time.time()
cos_price = np.exp(-r * T) * COSRepresentation.get_european_option_price(TypeEuropeanOption.CALL, a, b, 2 ** 14, k_s,
                                                                         cf_CGMY)
end_time = time.time()
diff_time = end_time - start_time
