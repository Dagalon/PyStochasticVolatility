import numpy as np
import Tools.AnalyticTools

from scipy.special import ndtr
from MC_Engines.MC_LocalVol import LocalVolFunctionals


def G(y):
    return Tools.AnalyticTools.normal_pdf(0.0, 1.0, y) - y * ndtr(-y)


def Gq(y):
    return (1 + y * y) * ndtr(-y) - Tools.AnalyticTools.normal_pdf(0.0, 1.0, y) * y


def Gr(y):
    return (1.0 - y * y) * ndtr(-y) + y * Tools.AnalyticTools.normal_pdf(0.0, 1.0, y) * y


def get_quadratic_option_lv_normal_sabr_watanabe_expansion(f0, k, t, alpha, nu, rho):
    y = (k - f0) / (alpha * np.sqrt(t))
    rho_inv = np.sqrt(1.0 - rho * rho)
    phi_y = Tools.AnalyticTools.normal_pdf(0.0, 1.0, y)
    cphi_y_inv = ndtr(-y)
    g_y = Gq(y)

    # T^{1/2}
    a_t = rho * nu * phi_y

    # T
    # g^2_2
    b_t = 0.25 * np.power(nu * rho, 2.0) * (np.power(y, 3.0) * phi_y + y * phi_y + 2.0 * cphi_y_inv)

    # g_3
    c_t = (2.0 * np.power(nu * rho, 2.0) * (y*phi_y) + np.power(nu * rho_inv, 2.0) * (2.0 * y * phi_y + 3.0 * cphi_y_inv)) / 6.0

    # T^{3/2}
    # g_2 * g_3
    d_1_t = np.power(nu, 3.0) * np.power(rho_inv, 2.0) * rho * (2.0 * np.power(y, 4.0) + 3.0 * y * y + 9) * phi_y
    d_2_t = np.power(nu * rho, 3.0) * (np.power(y, 4.0) + 3.0) * phi_y
    d_t = (d_1_t + d_2_t) / 12.0

    # g^3_2
    e_t = np.power(nu * rho, 3.0) * np.power(y * y - 1.0, 3.0) * phi_y / 24.0

    # g_4
    f_1_t = - 3.0 * np.power(nu, 3.0) * np.power(rho_inv, 2.0) * rho * alpha * (np.power(y, 2.0) - 1.0) * phi_y / 144.0
    f_2_t = - np.power(nu * rho, 2.0) * (7.0 * np.power(y, 2.0) - 1.0) * phi_y / 24.0

    # return alpha * alpha * t * (
    #         g_y + a_t * np.sqrt(t) + (b_t + c_t) * t + (e_t + d_t + f_1_t + f_2_t) * np.power(t, 1.5))
    return alpha * alpha * t * ( g_y + a_t * np.sqrt(t) + (b_t + c_t) * t +  (e_t + d_t + f_1_t + f_2_t) * np.power(t, 1.5))


def get_quadratic_option_normal_sabr_watanabe_expansion(f0, k, t, alpha, nu, rho):
    y = (k - f0) / (alpha * np.sqrt(t))
    rho_inv = np.sqrt(1.0 - rho * rho)
    phi_y = Tools.AnalyticTools.normal_pdf(0.0, 1.0, y)
    cphi_y_inv = ndtr(-y)
    g_y = Gq(y)

    # T^{1/2}
    a_t = 2.0 * rho * nu * phi_y

    # T
    # g_3
    b_1_t = np.power(nu * rho, 2.0) * y * phi_y / 6.0

    # g^2_2
    b_21_t = 0.25 * np.power(nu * rho, 2.0) * (np.power(y, 3.0) * phi_y + y * phi_y + 2.0 * cphi_y_inv)
    b_22_t = (np.power(nu * rho_inv, 2.0) / 6.0) * (2.0 * y * phi_y + 3.0 * cphi_y_inv)

    # T^{3/2}

    # T^{3/2}  g^3_2 / 6
    beta = (y * y - 1.0)
    m_y = beta * (0.125 * np.power(nu * rho, 3.0) * beta * beta +
                  0.25 * np.power(nu, 3.0) * rho * np.power(rho_inv, 2.0) * (2.0 * y * y + 1.0))

    c_3_t = m_y * phi_y / 3.0

    # T^{3/2} g_4
    c_1_t = np.power(nu * rho, 3.0) * y * phi_y / 12.0

    # T^{3/2} g_2*g_3
    c_21_t = np.power(nu * rho, 3.0) * (np.power(y, 4.0) - y * y + 1.0) * phi_y / 6.0
    # c_22_t = np.power(nu, 3.0) * np.power(rho_inv, 2.0) * rho * (
    #         y * y * phi_y - 3.0 * y * phi_y + 4.0 * phi_y - 3.0 * cphi_y_inv)

    # return alpha * alpha * t * (
    #         g_y + a_t * np.sqrt(t) + (b_1_t + b_21_t + b_22_t) * t + (c_1_t + c_21_t + c_3_t) * np.power(t, 1.5))

    return alpha * alpha * t * (
            g_y + a_t * np.sqrt(t) + (b_1_t + b_21_t + b_22_t) * t)


def get_option_normal_sabr_watanabe_expansion(f0, k, t, alpha, nu, rho, option_type):
    y = (k - f0) / (alpha * np.sqrt(t))
    rho_inv = np.sqrt(1.0 - rho * rho)
    phi_y = Tools.AnalyticTools.normal_pdf(0.0, 1.0, y)
    g_y = G(y)

    # T^{1/2}
    a_t = 0.5 * rho * nu * y

    # T g_3
    b_t = np.power(nu * rho, 2.0) * (y * y - 1.0) / 6.0

    # T 0.5 * g^2_2
    c_t = 0.125 * np.power(nu * rho * (y * y - 1.0), 2.0) + np.power(nu * rho_inv, 2.0) * (2.0 * y * y + 1.0) / 12.0

    # T^{3/2} g^3_2 / 6
    beta = (y * y - 1.0)
    m = beta * (0.125 * np.power(nu * rho, 3.0) * beta * beta + 0.25 * np.power(nu, 3.0) * rho * np.power(rho_inv,2.0) * (2.0 * y * y + 1.0))
    partial_m = 2.0 * y * (m / beta) + beta * (
            0.5 * np.power(nu * rho, 3.0) * y * beta + np.power(nu, 3.0) * rho * np.power(rho_inv, 2.0) * y)
    e_2_t = (y * m - partial_m) / 6.0

    # T^{3/2} g_4
    e_1_t = np.power(nu * rho, 3.0) * (np.power(y, 3.0) - 3.0 * y) / 24.0

    # g_2*g_3
    e_3_t = np.power(nu * rho, 3.0) * (y * y - 1.0) * (np.power(y, 3.0) - 3.0 * y) / 12.0
    # e_32_t = np.power(nu, 3.0) * np.power(rho_inv, 2.0) * rho * (np.power(y, 3.0) - 3.0 * y * y + 2.0 * y) / 2.0

    call_price = alpha * np.sqrt(t) * (g_y + phi_y * np.sqrt(t) * a_t + phi_y * t * (b_t + c_t) +
                                       np.power(t, 1.5) * phi_y * e_2_t + (e_1_t + e_3_t) * phi_y * np.power(t, 1.5))

    if option_type == 'c':
        return call_price
    else:
        return call_price - (f0 - k)


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
    b_t = 0.125 * np.power(nu * rho * (y * y - 1.0), 2.0) # g2
    c_t = np.power(nu * rho, 2.0) * (y * y - 1.0) / 12.0 + np.power(nu * rho_inv, 2.0) * (2.0 * y * y + 1.0) / 12.0 #g3

    # T^{3/2}
    # g_2 * g_3
    e_23_t = (y * y - 1.0) * (
            np.power(nu, 3.0) * rho * (2.0 * np.power(y, 3.0) - 3.0 * y) + np.power(nu * rho, 3.0) * (
                np.power(y, 3.0) - 3.0 * y)) / 24.0

    # g^2_3
    e_22_t = np.power(nu * rho, 3.0) * (y * np.power(y * y - 1.0, 2.0) * (y * y - 7.0)) / 48.0

    # g_4
    e_4_1_t = np.power(nu * rho, 3.0) * (np.power(y, 4.0) - 3.0 * y) / 288.0

    call_price = alpha * np.sqrt(t) * (g_y + phi_y * np.sqrt(t) * a_t + phi_y * t * (b_t + c_t) +
                                       phi_y * np.power(t, 1.5) * (e_23_t + e_22_t + e_4_1_t))

    if option_type == 'c':
        return call_price
    else:
        return call_price - (f0 - k)
