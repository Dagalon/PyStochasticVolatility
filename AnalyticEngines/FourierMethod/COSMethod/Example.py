import numpy as np
import matplotlib.pylab as plt
from AnalyticEngines.FourierMethod.CharesticFunctions import HestonCharesticFunction
from functools import partial
from Tools import Types
from AnalyticEngines.FourierMethod.COSMethod import COSRepresentation
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption

import time

# European option price
k_s = np.array([60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0, 160.0, 170.0])
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
b = 2.0

cf_heston = partial(HestonCharesticFunction.get_trap_cf, t=T, r_t=0.0, x=x0, v=v0, theta=theta, rho=rho, k=k, epsilon=epsilon, b=b2, u=u2)
start_time = time.time()
cos_price = COSRepresentation.get_european_option_price(TypeEuropeanOption.CALL, a, b, 64, k_s, cf_heston)
end_time = time.time()
diff_time = end_time - start_time

# Integration Heston´s charestic function
price_cf_integration = []
no_strikes = len(k_s)

start_time = time.time()
for i in range(0, no_strikes):
    european_option = EuropeanOption(k_s[i], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)
    price_cf_integration.append(european_option.get_analytic_value(0.0, theta, rho, k, epsilon, v0, 0.0,
                                                                   model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_REGULAR,
                                                                   compute_greek=False))
end_time = time.time()
diff_time_cf = end_time - start_time

print(diff_time)
print(diff_time_cf)

plt.plot(k_s, cos_price, label="cos method")
plt.plot(k_s, price_cf_integration, label="integration cf")

plt.legend()
plt.show()



