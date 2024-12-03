import matplotlib.pylab as plt
import numpy as np

from scipy.optimize import curve_fit

from AnalyticEngines.LocalVolatility.Hagan import ExpansionLocVol
from VolatilitySurface.Tools import SABRTools


def curvature_limit_value(spot: float, sigma: float, gamma: float):
    sigma_t = sigma * np.power(spot, gamma)
    c1 = np.power(sigma * gamma, 2.0) * np.power(spot, 2.0 * (gamma - 1.0)) / (3.0 * sigma_t)
    c2 = -1.5 * np.power(sigma * gamma, 2.0) * np.power(spot, 2.0 * (gamma - 1.0)) / sigma_t
    c3 = (np.power(sigma * gamma, 2.0) * np.power(spot, 2.0 * (gamma - 1.0)) / sigma_t
          + (gamma - 1.0) * sigma * gamma * np.power(spot, 2.0 * (gamma - 1.0)) / 3.0)

    return c1 + c2 + c3


def curvature_limit_value_elisa(spot: float, sigma: float, gamma: float):
    sigma_t = sigma * np.power(spot, gamma)
    m = sigma * gamma * gamma * np.power(spot, gamma - 2.0)
    return m * (-7.0 / 6.0 + (1.0 + (3.0 * gamma - 1.0) * gamma * gamma) / 3.0)


def curvature_limit_value_analytic(spot: float, sigma: float, gamma: float):
    return np.power(spot, gamma) * (gamma * gamma * sigma - 2.0 * gamma * sigma) / (6 * spot * spot)


# forward
f0 = 0.03

# CEV parameter
alpha = 0.6
nu = 0.3

tis = np.linspace(0.0, 0.5, 50)
atm_curvature = []

# Hagan approximation
expansion_hagan = ExpansionLocVol.hagan_loc_vol(lambda t: nu,
                                                lambda x: np.power(x, alpha),
                                                lambda x: alpha * np.power(x, alpha - 1.0),
                                                lambda x: alpha * (alpha - 1.0) * np.power(x, alpha - 2.0))

# Compute the curvature of CEV model using the Hagan's expansion
shift = 0.0001
for t in tis:
    # hagan's curvature
    curvature = expansion_hagan.get_bachelier_curvature(t, f0, f0)
    # atm_curvature.append(curvature)

    cev_iv = SABRTools.cev_iv_normal_jit(f0, f0, nu,  alpha, t)
    cev_iv_upper = SABRTools.cev_iv_normal_jit(f0, f0 + shift, nu,  alpha, t)
    cev_iv_lower = SABRTools.cev_iv_normal_jit(f0, f0 - shift, nu, alpha, t)
    curvature_cev = (cev_iv_upper + cev_iv_lower - 2.0 * cev_iv) / (shift * shift)
    atm_curvature.append(curvature_cev)

limit_value = curvature_limit_value(f0, nu, alpha)
limit_value_at_short_term = curvature_limit_value_elisa(f0, nu, alpha)
limit_value_analytic_at_short_term = curvature_limit_value_analytic(f0, nu, alpha)


def f_law(x, a, b):
    return a + b * x


popt, pcov = curve_fit(f_law, tis, atm_curvature)
y_fit_values = f_law(tis, *popt)

plt.plot(tis, y_fit_values, label='%s + %s T' % (round(popt[0], 8), round(popt[1], 8)), color='orange',
         linestyle='--', marker='.')

plt.plot(tis, atm_curvature, label='atm curvature', linestyle='dotted')

plt.title("sigma=%s, gamma=%s S= %s" % (nu, alpha, f0))

plt.legend()
plt.show()
