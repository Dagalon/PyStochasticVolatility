import numpy as np
from py_vollib.black_scholes_merton import black_scholes_merton, implied_volatility
from Tools import AnalyticTools
import matplotlib.pylab as plt

# Underlying information
s0 = 100.0
r = 0.0
q = 0.0

# Options information
t = 1.0
strikes = [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160]
vols = [0.75, 0.7, 0.65, 0.6, 0.55, 0.50, 0.55, 0.6, 0.65, 0.7, 0.75]

# Compute BS distribution
no_strikes = len(strikes)
bs_distribution = []
bs_distr_approximation = []
for i in range(0, no_strikes):
    bs_distribution.append(1.0 - AnalyticTools.bs_distribution(r, q, t, vols[i], s0, strikes[i]))
    bs_distr_approximation.append(AnalyticTools.bs_approximation_distribution(r, q, t, vols[i], s0, strikes[i]))

plt.plot(strikes, bs_distribution, label='bs_distribution')
plt.plot(strikes, bs_distr_approximation, label='bs_distr_approximation')

plt.legend()
plt.show()
