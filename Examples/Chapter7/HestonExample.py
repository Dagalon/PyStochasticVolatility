import numpy as np
import matplotlib.pylab as plt
from AnalyticEngines.FourierMethod.CharesticFunctions import JumpDiffusionCharesticFunction, HestonCharesticFunction
from functools import partial
from AnalyticEngines.FourierMethod.COSMethod import COSRepresentation
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from Tools import Types

# European option price
no_strikes = 22
k_s = np.linspace(70.0, 130.0, no_strikes)
f0 = 100.0
x0 = np.log(f0)

# Heston parameters
epsilon = 0.75
k = 0.6
rho = -0.5
v0 = 0.25
theta = 0.5
b2 = k
u2 = -0.5

# Upper and lower bound for cos integral
a = -2.0
b = 2.0

# maturities
T = [0.2, 0.4, 0.6, 1.0]
markers = ['.', '+', '*', '^']
no_maturities = len(T)

for i in range(0, no_maturities):
    cf_heston = partial(HestonCharesticFunction.get_trap_cf, t=T[i], r_t=0.0, x=x0, v=v0, theta=theta, rho=rho, k=k, epsilon=epsilon, b=b2, u=u2)

    cos_price_heston = COSRepresentation.get_european_option_price(TypeEuropeanOption.CALL, a, b, 128, k_s, cf_heston)

    iv_smile_heston = []
    for j in range(0, no_strikes):
        iv_smile_heston.append(implied_volatility(cos_price_heston[j], f0, f0, T[i], 0.0, 0.0, 'c'))

    plt.plot(k_s, iv_smile_heston, label='T=%s' % T[i], linestyle='--', color='black', marker=markers[i])

plt.ylim([0.0, 2.0])
plt.xlabel('K')
plt.legend()
plt.show()
