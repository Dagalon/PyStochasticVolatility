import matplotlib.pylab as plt
import numpy as np

from AnalyticEngines.BetaZeroSabr import ExpansionTools
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from VolatilitySurface.Tools import SABRTools

# option info
f0 = 0.02
strike = f0
ts = np.linspace(0.05, 5, 200)
strikes = []
options = []
for tsi in ts:
    options.append(EuropeanOption(strike, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, tsi))

# sabr parameters
alpha = 0.007
nu = 0.5
rho = 0.0
parameters = [alpha, nu, rho]

no_options = len(options)
iv_hagan = []
iv_watanabe = []

for i in range(0, no_options):
    iv_hagan.append(SABRTools.sabr_normal_jit(f0, strike, alpha, rho, nu, ts[i]))
    iv_watanabe.append(ExpansionTools.get_iv_normal_sabr_watanabe_expansion(f0, strike, ts[i], alpha, nu, rho))

plt.plot(ts, iv_hagan, label='iv hagan atm', linestyle='dotted')
plt.plot(ts, iv_watanabe, label='iv watanabe atm', linestyle='dashed')

plt.title("Implied Vol ATM by maturity")

plt.legend()
plt.show()



