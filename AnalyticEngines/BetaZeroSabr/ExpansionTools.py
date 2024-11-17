import numpy as np
import scipy as scp
from sympy.stats.sampling.sample_scipy import scipy

import Tools.AnalyticTools

from scipy.special import ndtr
from MC_Engines.MC_LocalVol import LocalVolFunctionals


def G(y):
    return Tools.AnalyticTools.normal_pdf(0.0, 1.0, y) - y * ndtr(-y)


def Gq(y):
    return (1 + y * y) * ndtr(-y) - Tools.AnalyticTools.normal_pdf(0.0, 1.0, y) * y


def Gr(y):
    return (1.0 - y * y) * ndtr(-y) + y * Tools.AnalyticTools.normal_pdf(0.0, 1.0, y) * y


def get_quadratic_option_normal_sabr_watanabe_expansion_replication(f0, k, t, alpha, nu, rho):
    y = (k - f0) / (alpha * np.sqrt(t))
    rho_inv = np.sqrt(1.0 - rho * rho)
    phi_y = Tools.AnalyticTools.normal_pdf(0.0, 1.0, y)
    cphi_y_inv = ndtr(-y)
    gr_y = 0.5 * Gq(y)

    #------ T^{1/2}
    a_t = 0.5 * nu * rho * phi_y

    #------ T
    b_t = np.power(nu * rho, 2.0) * y * phi_y / 6.0

    c_t = (0.125 * np.power(nu * rho, 2.0) * (
            np.power(y, 3.0) * phi_y + 3.0 * y * phi_y + 4.0 * cphi_y_inv - 2.0 * phi_y)
           + np.power(nu * rho_inv, 2.0) * (2.0 * y * phi_y + 3.0 * cphi_y_inv) / 12.0)

    # d_t
    d_1_t = (np.power(nu * rho, 3.0) / 12.0) * (np.power(y, 3.0) - 3.0 * y) * phi_y

    md2a = 0.5 * np.power(nu * rho, 2.0) * (y * y + 1.0) * phi_y + np.power(nu * rho_inv, 2.0) * phi_y / 3.0
    m2b = (0.125 * np.power(nu * rho, 2.0) * (
            (np.power(y, 4.0) + 4.0 * y * y + 9.0 - 2.0 * y) * phi_y - 2.0 * cphi_y_inv)
           + np.power(nu * rho_inv, 2.0) * (2.0 * y * y + 5.0) * phi_y / 12.0)

    d_2_t = (- md2a + m2b) / 6.0

    #------ T^{3/2}
    e_1_t = np.power(nu * rho, 3.0) * (((np.power(y, 2.0) - 1.0) / 3.0) * phi_y / 24.0 - phi_y / 6.0)
    e_2_t = 0.25 * np.power(nu, 3.0) * rho * phi_y
    e_3_t = np.power(nu * rho, 3.0) * (np.power(y, 2.0) + 3.0) * phi_y / 8.0 - 0.5 * np.power(nu * rho, 3.0) * phi_y

    return 2.0 * alpha * alpha * t * (
            gr_y + a_t * np.sqrt(t) + b_t * t + c_t * t + (d_1_t + d_2_t) * t + (e_1_t + e_2_t + e_3_t) * np.power(
        t, 1.5))


def get_quadratic_option_normal_sabr_watanabe_expansion(f0, k, t, alpha, nu, rho):
    y = (k - f0) / (alpha * np.sqrt(t))
    rho_inv = np.sqrt(1.0 - rho * rho)
    phi_y = Tools.AnalyticTools.normal_pdf(0.0, 1.0, y)
    cphi_y_inv = ndtr(-y)
    g_y = Gq(y)

    # epsilon
    a_t = rho * nu * phi_y * np.sqrt(t)

    # epsilon ^2
    # b_t = nu * nu * t * (0.5 * rho_inv * rho_inv * cphi_y_inv + y * phi_y / 3.0)
    c_t = 0.25 * np.power(nu * rho, 2.0) * ((np.power(y, 3.0) + y) * phi_y + 2.0 * cphi_y_inv) * t
    b_t = y * (phi_y / 3.0) * nu * nu * t + 0.5 * np.power(nu * rho_inv, 2.0) * cphi_y_inv * t

    # epsilon^3
    # d_t = np.power(t, 1.5) * phi_y * (np.power(nu, 3.0) * rho * (np.power(y, 4.0) / 6.0 + 0.5 * y) +
    #                                   np.power(nu, 3.0) * rho * rho * np.power(y * y - 1.0, 3.0) / 24.0)
    # d_t = np.power(t, 1.5) * phi_y * (np.power(nu, 3.0) * rho * (np.power(y, 4.0) / 6.0 + 0.5 * y))

    return alpha * alpha * t * (g_y + a_t + b_t + c_t)


def get_option_normal_sabr_watanabe_expansion(f0, k, t, alpha, nu, rho, option_type):
    y = (k - f0) / (alpha * np.sqrt(t))
    rho_inv = np.sqrt(1.0 - rho * rho)
    phi_y = Tools.AnalyticTools.normal_pdf(0.0, 1.0, y)
    g_y = G(y)

    a_t = 0.5 * rho * nu * y

    # b_t
    b_t = np.power(nu * rho, 2.0) * (y * y - 1.0) / 6.0

    # c_t
    c_t = 0.125 * np.power(nu * rho * (y * y - 1.0), 2.0) + np.power(nu * rho_inv, 2.0) * (2.0 * y * y + 1.0) / 12.0

    # d_t
    p4y = np.power(y, 4.0) - 2.0 * y * y - 1.0

    d_1_t = np.power(nu * rho, 3.0) * (p4y / 12.0 - (y * y - 1.0) / 3.0) # + rho * np.power(nu, 3.0) * (y * y - 1.0) / 3.0
    md2 = np.power(nu * rho, 2.0) * (y * y - 1) * y + 2.0 * np.power(nu * rho_inv, 2.0) * y / 3.0 # partial_y g2
    d_2_t = - md2 / 6.0

    # extra term T^{3/2}
    e_1_t = np.power(nu * rho, 3.0) * ((np.power(y, 3.0) * phi_y + y * phi_y) / 24.0 - y * phi_y / 6.0)  # g4
    e_2_t = np.power(nu, 3.0) * rho * y * phi_y / 6.0  # g4
    e_2_t = 0.0
    e_3_t = np.power(nu * rho, 3.0) * (np.power(y, 3.0) + y) * phi_y / 8.0 + 0.5 * np.power(nu, 3.0) * rho_inv * rho_inv * rho * y * phi_y

    if option_type == 'c':
        return alpha * np.sqrt(t) * (g_y + phi_y * np.sqrt(t) * a_t + phi_y * t * (b_t + c_t) +
                                     t * phi_y * (d_1_t + d_2_t) + (e_1_t + e_2_t + e_3_t) * np.power(t, 1.5))
    else:
        call_price = alpha * np.sqrt(t) * (g_y + phi_y * np.sqrt(t) * a_t + phi_y * t * (b_t + c_t) +
                                           t * phi_y * (d_1_t + d_2_t) + (e_1_t + e_2_t + e_3_t) * np.power(t, 1.5))

        return call_price - (f0 - k)


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

    # T^1/2{}
    a_t = 0.5 * rho * nu * y

    # T
    b_t = 0.125 * np.power(nu * rho * (y * y - 1.0), 2.0)
    c_t = np.power(nu * rho, 2.0) * (y * y - 1.0) / 12.0 + np.power(nu * rho_inv, 2.0) * (2.0 * y * y + 1.0) / 12.0
    d_t = (1.0 / 24.0) * np.power(nu * rho * (y * y - 1.0), 2.0) * (np.power(y, 3.0) - 2.0 * y)

    # T^{3/2}
    # e_1_t = - 3.0 * alpha * rho * np.power(nu, 3.0) * np.power(rho_inv, 2.0) * (np.power(y, 3.0) - 3.0 * y)
    e_1_t = 0.0

    call_price = alpha * np.sqrt(t) * (g_y + phi_y * np.sqrt(t) * a_t + phi_y * t * (b_t + c_t + d_t) + phi_y * np.power(t, 1.5) * (e_1_t))

    if option_type == 'c':
        return call_price
    else:
        return call_price - (f0 - k)
