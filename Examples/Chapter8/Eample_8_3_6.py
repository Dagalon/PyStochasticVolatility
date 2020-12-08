import numpy as np

from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import Types
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
import matplotlib.pylab as plt

# Parameters
epsilon = 2.5
k = 0.3
# rho = -0.9
rho = -0.2
v0 = 0.2
theta = 0.4

# options
strikes = np.linspace(70.0, 130.0, 30)
no_strikes = len(strikes)
f0 = 100
T = 0.1
notional = 1.0
options = []
for k_i in strikes:
    options.append(EuropeanOption(k_i, notional, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, T))


iv_vol = []

for i in range(0, no_strikes):
    price = options[i].get_analytic_value(0.0, theta, rho, k, epsilon, v0, 0.0,
                                          model_type=Types.ANALYTIC_MODEL.HESTON_MODEL_ATTARI,
                                          compute_greek=False)

    iv_vol.append(implied_volatility(price, f0, strikes[i], T, 0.0, 0.0, 'c'))


plt.plot(strikes, iv_vol, label="rho=%s" % rho, marker=".", linestyle="--", color="black")

plt.xlabel("K")
plt.legend()
plt.show()


