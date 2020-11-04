import numpy as np

from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import Types
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
import matplotlib.pylab as plt

# Parameters
epsilon = 1.1
k = 0.5
rho = -0.9
v0 = 0.05
theta = 0.05

# options
strikes = np.arange(80.0, 120.0, 1.0)
no_strikes = len(strikes)
f0 = 100
T = 0.1
notional = 1.0
options = []
for k_i in strikes:
    options.append(EuropeanOption(k_i, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T))


iv_vol_rho = []
iv_rho_rho_zero = []

for i in range(0, no_strikes):
    price_rho_no_zero = options[i].get_analytic_value(0.0, theta, rho, k, epsilon, v0, 0.0,
                                                      model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_ATTARI,
                                                      compute_greek=False)

    price_rho_zero = options[i].get_analytic_value(0.0, theta, 0.0, k, epsilon, v0, 0.0,
                                                   model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_ATTARI,
                                                   compute_greek=False)

    iv_vol_rho.append(implied_volatility(price_rho_no_zero, f0, strikes[i], T, 0.0, 0.0, 'c'))
    iv_rho_rho_zero.append(implied_volatility(price_rho_zero, f0, strikes[i], T, 0.0, 0.0, 'c'))


plt.plot(strikes, iv_vol_rho, label="rho=%s" % rho, marker=".", linestyle="--", color="black")
plt.plot(strikes, iv_rho_rho_zero, label="rho=0.0", marker="+", linestyle="--", color="black")

plt.legend()
plt.show()


