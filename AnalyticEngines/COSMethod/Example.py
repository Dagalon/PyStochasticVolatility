import numpy as np
from AnalyticEngines.FourierMethod import HestonCharesticFunction
from functools import partial
from Instruments.EuropeanInstruments import TypeEuropeanOption
from AnalyticEngines.COSMethod import COSRepresentation
from scipy.stats import norm

import time


# def normal_phi(u: Types.ndarray):
#     return np.exp(-0.5 * u * u) + 0.0 * 1j
#
#
# x = np.arange(-6.0, 6.0, 0.1)
# normal_pdf = norm.pdf(x, 0.0, 1.0)
# start_time = time.time()
# cos_normal_pdf = COSRepresentation.get_cos_density(-10.0, 10.0, 32, normal_phi, x)
# end_time = time.time()
# diff_time = end_time - start_time

# plt.plot(x, normal_pdf, label="normal pdf")
# plt.plot(x, cos_normal_pdf, label="COS normal pdf")

# plt.legend()
# plt.show()

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
a = -10.0
b = 10.0

cf_heston = partial(HestonCharesticFunction.get_cf, t=T, x=x0, v=v0, theta=theta, rho=rho, epsilon=epsilon, b=b2, u=u2)
start_time = time.time()
cos_price = COSRepresentation.get_european_option_price(TypeEuropeanOption.CALL, a, b, 32, k_s, cf_heston)
end_time = time.time()

