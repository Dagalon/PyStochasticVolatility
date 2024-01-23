import numpy as np
import Tools.AnalyticTools

from scipy.special import ndtr
from MC_Engines.MC_LocalVol import LocalVolFunctionals


def G(y):
    return Tools.AnalyticTools.normal_pdf(0.0, 1.0, y) - y * ndtr(-y)


def Gq(y):
    return (1+y*y) * ndtr(-y) - 2.0 * Tools.AnalyticTools.normal_pdf(0.0, 1.0, y) * y


def get_quadratic_option_normal_sabr_watanabe_expansion(f0, k, t, alpha, nu, rho):
    y = (k - f0) / (alpha * np.sqrt(t))
    rho_inv = np.sqrt(1.0 - rho * rho)
    phi_y = Tools.AnalyticTools.normal_pdf(0.0, 1.0, y)
    cphi_y = ndtr(-y)
    g_y = Gq(y)

    a_t = rho * nu * (cphi_y * (1+y) - phi_y) * np.sqrt(t)
    b_t = np.power(nu * y, 2.0) / 3.0 + (2.0 * nu * nu - 3.0 * np.power(nu * rho, 2.0) / 12.0 + y * np.power(nu * rho, 2.0) / 8.0) * phi_y * t
    c_t = 0.25 * np.power(nu * rho_inv, 2.0) * cphi_y * t
    d_t = nu * nu * phi_y * np.power(t, 1.5)

    return alpha * alpha * t * (g_y + a_t + b_t + c_t - d_t)


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

    b_t = 0.5 * np.power(nu * rho, 2.0) * (1.0 / 6.0 + np.power(y, 2.0) / 3.0 - 0.25 * np.power(y, 4.0)) + \
          0.5 * np.power(nu * rho, 2.0) * np.power(0.5 * (y * y - 1.0), 2.0) + \
          0.5 * np.power(rho_inv * nu, 2.0) * ((np.power(y, 2.0) + 2.0) / 3.0) - 0.25 * nu * nu

    return alpha * (1.0 + np.sqrt(t) * a_t + t * b_t)


def get_iv_normal_lv_sabr_watanabe_expansion(f0, k, t, alpha, nu, rho):
    y = (k - f0) / (alpha * np.sqrt(t))

    a_t = 0.5 * rho * nu * y  # \sqrt{t}
    b_t = np.power(nu * y, 2.0) * (2.0 - 3.0 * rho * rho) / 12.0 + nu * nu * (2.0 - 3.0 * rho * rho) / 24.0

    return alpha * (1.0 + a_t * np.sqrt(t) + b_t * t)


def get_option_normal_sabr_loc_vol_expansion(f0, k, t, alpha, nu, rho, option_type):
    x = np.array([f0])
    sigma_0 = LocalVolFunctionals.local_vol_normal_sabr(t, x, f0, alpha, rho, nu)[0]
    y = (k - f0) / (sigma_0 * np.sqrt(t))
    phi_y = Tools.AnalyticTools.normal_pdf(0.0, 1.0, y)
    g_y = G(y)
    rho_inv = 1.0 - rho * rho

    a_t = 0.5 * rho * nu * y
    b_t = nu * nu * (y * y - 1.0) / 6.0
    c_t = 0.25 * np.power(nu * rho_inv, 2.0) + 0.125 * np.power(rho * nu, 2.0) * np.power(y*y - 1.0, 2.0)

    if option_type == 'c':
        return alpha * np.sqrt(t) * (g_y + phi_y * np.sqrt(t) * a_t + phi_y * t * (b_t + c_t))
    else:
        return alpha * np.sqrt(t) * (g_y + phi_y * np.sqrt(t) * a_t + phi_y * t * (b_t + c_t)) - (f0 - k)





