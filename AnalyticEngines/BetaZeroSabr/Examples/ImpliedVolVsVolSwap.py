import numpy as np
import matplotlib.pylab as plt

from AnalyticEngines.BetaZeroSabr import ExpansionTools
from AnalyticEngines.MalliavinMethod.ExpansionTools import get_vol_swap_approximation_sabr
from scipy.optimize import curve_fit

# option info
f0 = 0.02
strike = f0
ts = np.linspace(0.05, 5, 200)

# sabr parameters
alpha = 0.007
nu = 0.5
rho = 0.0
parameters = [alpha, nu, rho]

iv_watanabe = []
vol_swap = []
diff = []

for i in range(0, len(ts)):
    iv_watanabe.append(ExpansionTools.get_iv_normal_sabr_watanabe_expansion(f0, strike, ts[i], alpha, nu, rho))
    vol_swap.append(get_vol_swap_approximation_sabr(parameters, 0.0, ts[i], alpha))
    diff.append(iv_watanabe[-1] - vol_swap[-1])


def f_law(x, a, b, c):
    return a + b * np.power(x, c)


popt, pcov = curve_fit(f_law, ts, diff)
y_fit_values = f_law(ts, *popt)

plt.plot(ts, diff, label='v_t - iv_t', linestyle='dotted')
plt.plot(ts, y_fit_values, label="%s + %s * t^%s" % (round(popt[0], 5), round(popt[1], 5), round(popt[2], 5)), linestyle='--')
# plt.plot(ts, iv_watanabe, label='iv_t', linestyle='--')

plt.title("v_t vs iv_t")

plt.legend()
plt.show()
