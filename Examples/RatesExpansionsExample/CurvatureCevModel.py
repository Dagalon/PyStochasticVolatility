import matplotlib.pylab as plt
import numpy as np

from scipy.optimize import curve_fit
from AnalyticEngines.LocalVolatility.Hagan import ExpansionLocVol


def curvature_limit_value(spot: float, sigma: float, gamma: float):
    m = sigma * np.power(gamma, 2.0) * np.power(spot, (gamma - 2))
    return m * (-7.0 / 9.0 + (1.0 + (3.0 * gamma - 1) * gamma * gamma) / 3.0)

# forward
f0 = 0.03

# CEV parameter
alpha = 0.3
nu = 0.01

tis = np.linspace(0.001, 0.5, 50)
atm_curvature = []

# Hagan approximation
expansion_hagan = ExpansionLocVol.hagan_loc_vol(lambda t: nu,
                                                lambda x: np.power(x, alpha),
                                                lambda x: alpha * np.power(x, alpha - 1.0),
                                                lambda x: alpha * (alpha - 1.0) * np.power(x, alpha - 2.0))

# Compute the curvature of CEV model using the Hagan's expansion
for t in tis:
    # hagan's curvature
    curvature = expansion_hagan.get_bachelier_curvature(t, f0, f0)
    atm_curvature.append(curvature)


limit_value = curvature_limit_value(f0, nu, alpha)


def f_law(x, a, b):
    return a  +  b * x

popt, pcov = curve_fit(f_law, tis, atm_curvature)
y_fit_values = f_law(tis, *popt)


plt.plot(tis, y_fit_values, label='%s + %s T' % (round(popt[0], 8), round(popt[1], 8)), color='orange',
          linestyle='--', marker='.')

plt.plot(tis, atm_curvature, label='atm curvature', linestyle='dotted')

plt.title("sigma=%s, gamma=%s F= %s" % (nu, alpha, f0))

plt.legend()
plt.show()








