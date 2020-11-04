import numpy as np
import numba as nb
from scipy.special import hyp2f1
from scipy.integrate import quad_vec


def get_volterra_covariance(s: float, t: float, h: float):
    # We suppose that t > s
    if s > 0.0:
        # gamma = 0.5 - h
        x = s / t
        # alpha = ((1.0 - 2.0 * gamma) / (1.0 - gamma)) * np.power(x, gamma) * hyp2f1(1.0, gamma, 2.0 - gamma, x)
        # return np.power(s, 2.0 * h) * alpha
        alpha = 2.0 * np.power(s, h + 0.5) * np.power(t, h - 0.5) * h / (h + 0.5)
        return alpha * hyp2f1(0.5 - h, 1.0, h + 1.5, x)

    else:
        return 0.0


no_t_i_s = 200
h = 0.3
t_i_s = np.linspace(0.01, 2.0, 200)

output_analytic = []
output_numerical = []
for i in range(1, no_t_i_s):
    for j in range(0, i):
        output_analytic.append(get_volterra_covariance(t_i_s[j], t_i_s[i], h))
        mult = 2.0 * h * np.power(t_i_s[j], h + 0.5) * np.power(t_i_s[i], h - 0.5)
        z = (t_i_s[j] / t_i_s[i])
        f = lambda x: np.power(1 - z * x, h - 0.5) * np.power(1 - x, h - 0.5)
        integral_value = quad_vec(f, 0.0, 0.99999999999999)
        output_numerical.append(mult * integral_value[0])





