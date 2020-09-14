import numpy as np
from AnalyticEngines.FourierMethod import HestonCharesticFunction
from functools import partial
from Tools import Types
from Instruments.EuropeanInstruments import TypeEuropeanOption
from AnalyticEngines.COSMethod import COSRepresentation
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from scipy.stats import norm

import time

# European option price
k_s = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
f0 = 100.0
x0 = np.log(f0)
T = 2.0

# Heston parameters
epsilon = 1.1
k = 0.5
rho = -0.9
v0 = 0.05
theta = 0.05
b2 = k
u2 = -0.5

# Upper and lower bound for cos integral
a = -2.0
b = 0.7

cf_heston = partial(HestonCharesticFunction.get_trap_cf, t=T, r_t=0.0, x=x0, v=v0, theta=theta, rho=rho, k=k, epsilon=epsilon, b=b2, u=u2)
start_time = time.time()
cos_price = COSRepresentation.get_european_option_price(TypeEuropeanOption.CALL, a, b, 64, k_s, cf_heston)
end_time = time.time()

# Integration Heston´s charestic function
price_cf_integration = []
no_strikes = len(k_s)

for i in range(0, no_strikes):
    european_option = EuropeanOption(k_s[i], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)
    price_cf_integration.append(european_option.get_analytic_value(0.0, theta, rho, k, epsilon, v0, 0.0,
                                                                   model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_REGULAR,
                                                                   compute_greek=False))


