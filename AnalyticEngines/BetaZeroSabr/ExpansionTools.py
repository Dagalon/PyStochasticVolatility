import numpy as np
import Tools.AnalyticTools

from scipy.special import ndtr
from MC_Engines.MC_LocalVol import LocalVolFunctionals


def G(y):
    return Tools.AnalyticTools.normal_pdf(0.0, 1.0, y) - y * ndtr(-y)


def get_option_normal_sabr_watanabe_expansion(f0, k, t, alpha, nu, rho, option_type):
    y = (k - f0) / (alpha * np.sqrt(t))
    rho_inv = np.sqrt(1.0 - rho * rho)
    phi_y = Tools.AnalyticTools.normal_pdf(0.0, 1.0, y)
    g_y = G(y)

    a_t = 0.5 * rho * nu * y
    b_t = 0.5 * np.power(nu * rho, 2.0) * (np.power(y, 2.0) / 3.0 + 1.0 / 6.0) + \
          0.5 * np.power(nu * rho, 2.0) * np.power(0.5 * (y * y - 1.0), 2.0) + \
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


def get_iv_normal_lv_sabr_watanabe_expansion(f0, k, t, alpha, nu, rho):
    y = (k - f0) / (alpha * np.sqrt(t))
    rho_inv = np.sqrt(1.0 - rho * rho)
    a_t = 0.5 * alpha * rho * nu * y
    b_t_0 = 0.25 * np.power(nu * rho_inv, 2.0) + 0.125 * alpha * np.power(nu * rho, 2.0) * np.power(y * y - 1.0, 2.0)
    - 0.125 * alpha * np.power(nu * rho, 2.0) * np.power(y, 2.0)
    b_t_1 = np.power(alpha * nu * rho_inv, 2.0) * (2.0 * y * y + 1.0) / 6.0
    return alpha + a_t * np.sqrt(t) + (b_t_0 + b_t_1) * t


def get_option_normal_sabr_loc_vol_expansion(f0, k, t, alpha, nu, rho, option_type):
    x = np.array([f0])
    sigma_0 = LocalVolFunctionals.local_vol_normal_sabr(t, x, f0, alpha, rho, nu)[0]
    y = (k - f0) / (sigma_0 * np.sqrt(t))
    phi_y = Tools.AnalyticTools.normal_pdf(0.0, 1.0, y)
    g_y = G(y)
    rho_inv = 1.0 - rho * rho

    a_t = 0.5 * rho * nu * y
    b_t = rho * nu * alpha * (2.0 * y * y + 1.0) / 6.0 + 0.25 * nu * nu * rho_inv + 0.125 * np.power(rho * nu, 2.0) * np.power(y * y - 1, 2.0)
    c_t = alpha * np.power(rho * nu, 2.0) * (0.25 * np.power(y, 5.0) - np.power(y, 3.0) / 3.0 + 0.25 * y)

    if option_type == 'c':
        return alpha * np.sqrt(t) * (g_y + phi_y * np.sqrt(t) * a_t + phi_y * t * b_t + phi_y * np.power(t, 1.5) * c_t)
    else:
        return alpha * np.sqrt(t) * (g_y + phi_y * np.sqrt(t) * a_t + phi_y * t * b_t + phi_y * np.power(t, 1.5) * c_t) - (f0 - k)





