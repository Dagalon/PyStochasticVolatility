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

# NIG parameters
r = 0.01
sigma = 0
alpha = 8.9932
beta = 0
delta = 1.1528

# Upper and lower bound for cos integral
a = -8 * np.sqrt(T)
b = 8 * np.sqrt(T)

cf_NIG = partial(JumpDiffusionCharesticFunction.get_NIGB_cf, t=T, x=x0, r=r, sigma=sigma, alpha=alpha, beta=beta,
                 delta=delta)
start_time = time.time()
cos_price = np.exp(-r * T) * COSRepresentation.get_european_option_price(TypeEuropeanOption.CALL, a, b, 2 ** 14, k_s,
                                                                         cf_NIG)
end_time = time.time()
diff_time = end_time - start_time
