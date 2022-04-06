import numpy as np
import Tools.AnalyticTools
import numba as nb

from scipy.special import ndtr

CALL = 'c'
PUT = 'p'

binary_flag = {CALL: 1, PUT: -1}
phi_hat_c = -0.001882039271


@nb.jit("f8(f8)", nopython=True, nogil=True)
def gamma_inc_inv(x):

    return ndtr(x) + Tools.AnalyticTools.normal_pdf(0.0, 1.0, x) / x


def bachelier(f, k, t, sigma, flag):

    if f == k:
        return sigma * np.sqrt(t) * 1.0 / np.sqrt(2.0 * np.pi)
    else:
        x = - binary_flag[flag] * (k - f) / (sigma * np.sqrt(t))
        return - binary_flag[flag] * (k - f) * gamma_inc_inv(x)


def implied_volatility(price, f, k, t, flag):

    if f == k:
        return np.sqrt(2.0 * np.pi / t) * price

    phi_hat_target = - np.abs(price - np.maximum(binary_flag[flag] * (f - k), 0.0)) / np.abs(k - f)

    if phi_hat_target < phi_hat_c:
        g = 1.0 / (phi_hat_target - 0.5)
        psi_hat_numerator = 0.032114372355 - g * g * (
                0.016969777977 - g * g * (2.6207332461E-3 - 9.6066952861E-5 * g * g))
        psi_hat_denominator = 1.0 - g * g * (0.6635646938 - g * g * (0.14528712196 - 0.010472855461 * g * g))
        psi_hat = psi_hat_numerator / psi_hat_denominator
        x_hat = g * (1.0 / np.sqrt(2.0 * np.pi) + psi_hat * g * g)

    else:
        h = np.sqrt(-np.log(-phi_hat_target))
        x_hat_numerator = 9.4883409779 - h * (9.6320903635 - h * (0.58556997323 + 2.1464093351 * h))
        x_hat_denominator = 1.0 - h * (0.65174820867 + h * (1.5120247828 + 6.6437847132e-05 * h))
        x_hat = x_hat_numerator / x_hat_denominator

    q = (gamma_inc_inv(x_hat) - phi_hat_target) / Tools.AnalyticTools.normal_pdf(0.0, 1.0, x_hat)
    x_root_numerator = 3.0 * q * x_hat * x_hat * (2.0 - q * x_hat * (2.0 + x_hat * x_hat))
    x_root_denominator = 6.0 + q * x_hat * (
                -12.0 + x_hat * (6.0 * q + x_hat * (-6.0 * q * x_hat * (3.0 + x_hat * x_hat))))
    x_root = x_hat + x_root_numerator / x_root_denominator

    return np.abs(k - f) / np.abs(x_root * np.sqrt(t))
