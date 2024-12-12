import matplotlib.pylab as plt
import numpy as np

from scipy.optimize import curve_fit
from VolatilitySurface.Tools import SABRTools


# option info
f0 = 0.03
strikes = []
options = []

# sabr parameters
alpha = 0.007
nu = 0.4
rho = 0.5
parameters = [alpha, nu, rho]

# mc price
seed = 123456789
no_paths = 1000000

tis = np.linspace(0.0001, 0.25, 50)
shift = 0.00001
atm_curvature = []

for t in tis:
    # hagan's volatility
    partial_iv_hagan_base = SABRTools.sabr_normal_partial_k_jit(f0, f0, alpha, rho, nu, t)
    partial_iv_hagan_up = SABRTools.sabr_normal_partial_k_jit(f0, f0 + shift, alpha, rho, nu, t)
    partial_iv_hagan_down = SABRTools.sabr_normal_partial_k_jit(f0, f0 - shift, alpha, rho, nu, t)

    curvature = 0.5 * (partial_iv_hagan_up - partial_iv_hagan_down) / shift
    atm_curvature.append(curvature)


def f_law(x, a, b):
    return a  +  b * x

popt, pcov = curve_fit(f_law, tis, atm_curvature)
y_fit_values = f_law(tis, *popt)


plt.plot(tis, y_fit_values, label='%s + %s T' % (round(popt[0], 5), round(popt[1], 5)), color='orange',
          linestyle='--', marker='.')

plt.plot(tis, atm_curvature, label='atm curvature', linestyle='dotted')

plt.title("rho=%s, F= %s" % (rho, f0))

plt.legend()
plt.show()








