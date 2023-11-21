__author__ = 'David Garcia Lorite'

#
# Copyright 2020 David Garcia Lorite
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and limitations under the License.
#

import numpy as np
import matplotlib.pylab as plt
from AnalyticEngines.FourierMethod.CharesticFunctions import HestonCharesticFunction
from functools import partial
from Tools import Types
from AnalyticEngines.FourierMethod.COSMethod import COSRepresentation
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption

import time

# European option price
k_s = np.array([50.0, 60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 140.0, 160.0, 170.0, 180.0])
f0 = 100.0
x0 = np.log(f0)
T = 0.25

# Heston parameters
r = 0.00
v0 = 0.25
k = 1.5
theta = 0.2
epsilon = 0.05
rho = -0.8
b2 = k
u2 = -0.5

# Upper and lower bound for cos integral
a = -6.0
b = 6.0

cf_heston = partial(HestonCharesticFunction.get_trap_cf, t=T, r_t=0.0, x=x0, v=v0, theta=theta, rho=rho, k=k, epsilon=epsilon, b=b2, u=u2)
start_time = time.time()
cos_price = COSRepresentation.get_european_option_price(TypeEuropeanOption.CALL, a, b, 64, k_s, cf_heston)
end_time = time.time()
diff_time = end_time - start_time
print("COS method time ---- " + str(diff_time))

# Integration Heston´s charestic function
price_cf_integration = []
no_strikes = len(k_s)

start_time = time.time()
for i in range(0, no_strikes):
    european_option = EuropeanOption(k_s[i], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T)
    price_cf_integration.append(european_option.get_analytic_value(0.0, theta, rho, k, epsilon, v0, 0.0,
                                                                   model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_LEWIS,
                                                                   compute_greek=False))
end_time = time.time()
diff_time_cf = end_time - start_time
print("Lewis's formula integration method time ---- " + str(diff_time_cf))

plt.plot(k_s, cos_price, color="blue", linestyle="dotted", label="cos method")
plt.plot(k_s, price_cf_integration, color="green", label="integration cf")

plt.legend()
plt.show()



