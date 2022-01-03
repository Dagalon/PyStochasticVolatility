import numpy as np
from ncephes import ndtr

import Tools.AnalyticTools


def G(y):
    return Tools.AnalyticTools.normal_pdf(0.0, 1.0, y) + y * ndtr(-y)


def get_option_normal_sabr_watanabe_expansion(f0, k, t, alpha, nu, rho, option_type):
    y = (k - f0) / (alpha * np.sqrt(t))
    rho_inv = np.sqrt(1.0 - rho * rho)
    phi_y = Tools.AnalyticTools.normal_pdf(0.0, 1.0, y)
    g_y = G(y)

    a_t = 0.5 * rho * nu * y
    b_t = 0.5 * np.power(nu * rho, 2.0) * (np.power(y, 2.0) / 3.0 + 1.0 / 6.0) + \
          0.5 * np.power(nu * rho, 2.0) * np.power(0.25 * (y * y - 1.0), 2.0) + \
          0.5 * np.power(rho_inv * nu, 2.0) * ((np.power(y, 2.0) + 2.0) / 3.0) - 0.25 * nu * nu

    if option_type == 'c':
        return alpha * np.sqrt(t) * (g_y + phi_y * np.sqrt(t) * a_t + phi_y * t * b_t)
    else:
        return alpha * np.sqrt(t) * (g_y + phi_y * np.sqrt(t) * a_t + phi_y * t * b_t) - (f0 - k)


def get_iv_normal_sabr_watanabe_expansion(f0, k, t, alpha, nu, rho):
    y = (k - f0) / (alpha * np.sqrt(t))
    rho_inv = np.sqrt(1.0 - rho * rho)
    a_t = 0.5 * rho * nu * y
    b_t = 0.5 * np.power(nu * rho, 2.0) * (1.0 - y * y) / 6.0 + \
          0.5 * np.power(nu * rho, 2.0) * np.power(0.25 * (y * y - 1.0), 2.0) + \
          0.5 * np.power(rho_inv * nu, 2.0) * ((np.power(y, 2.0) + 2.0) / 3.0) - 0.25 * nu * nu

    return alpha * (1.0 + np.sqrt(t) * a_t + t * b_t)