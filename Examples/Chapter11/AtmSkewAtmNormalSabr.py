import matplotlib.pylab as plt
import numpy as np

from Tools import RNG, Types
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption
from scipy.optimize import curve_fit
from VolatilitySurface.Tools import SABRTools

dt = np.arange(7, 90, 2) * (1.0 / 365.0)
no_dt_s = len(dt)

# simulation info
alpha = 0.01
nu = 0.5
rho = 0.5
parameters = [alpha, nu, rho]
no_time_steps = 100

seed = 123456789
no_paths = 2000000
delta_time = 1.0 / 365.0

# random number generator
rnd_generator = RNG.RndGenerator(seed)

# option information
f0 = 0.03
options = []
options_shift_right = []
options_shift_left = []
shift_spot = 0.0002
for d_i in dt:
    options.append(EuropeanOption(f0, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))
    options_shift_right.append(
        EuropeanOption(f0 + shift_spot, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))
    options_shift_left.append(
        EuropeanOption(f0 - shift_spot, 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, d_i))

# outputs
skew_atm_hagan = []

for i in range(0, no_dt_s):
    skew_hagan = SABRTools.sabr_normal_partial_k_jit(f0, f0, alpha, rho, nu, dt[i])
    skew_atm_hagan.append(skew_hagan)


def f_law(x, a, b, c):
    return a + b * np.power(x, c)


popt, pcov = curve_fit(f_law, dt, skew_atm_hagan)
y_fit_values = f_law(dt, *popt)

plt.plot(dt, skew_atm_hagan, label='dIV(F0,T)/dF0', color='black', linestyle='--')
plt.scatter(dt, y_fit_values, label="%s + %s * t ^ %s" % (round(popt[0], 5), round(popt[1], 5), round(popt[2], 5)), marker='.',
         linestyle='--', color='black')

plt.xlabel('T')
plt.legend()
plt.show()
