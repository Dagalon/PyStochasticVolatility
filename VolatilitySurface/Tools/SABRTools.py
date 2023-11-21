__author__ = 'David Garcia Lorite'

#
# Copyright 2020 David Garcia Lorite
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
#

import numpy as np
import numba as nb

from VolatilitySurface.Tools import ParameterTools
from AnalyticEngines.LocalVolatility.Dupire import DupireFormulas
from Tools import Types, AnalyticTools
from ncephes import ndtr


@nb.jit("f8[:](f8[:],f8[:])", nopython=True, nogil=True)
def dot_product(a, b):
    n = len(a)
    output = np.zeros(n)
    for i in range(0, n):
        output[i] = a[i] * b[i]

    return output


@nb.jit("f8(f8,f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def sabr_short_term_local_vol(alpha_t, rho_t, nu_t, f_t, k, dt):
    w_t = alpha_t * np.sqrt(dt)
    z_w = (np.log(k / f_t) / w_t) + 0.5 * w_t
    local_vol = alpha_t * alpha_t * np.exp(
        nu_t * nu_t * dt + 2.0 * nu_t * rho_t * np.sqrt(dt) * (z_w - rho_t * nu_t * np.sqrt(dt)))
    return np.sqrt(local_vol)


@nb.jit("f8(f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def sabr_vol_jit(alpha, rho, v, z, t):
    epsilon = 1E-07
    v = np.minimum(v, 1000.0)

    if t == 0.0 or alpha == 0.0:
        return 0.0

    else:
        if v < epsilon:
            sigma = alpha
        else:
            if 0.0 <= z < epsilon:
                z = epsilon
            elif -epsilon < z < 0.0:
                z = -epsilon

            y = (v / alpha) * z
            y_prime = (np.sqrt(1 - 2 * rho * y + y ** 2) + y - rho) / (1 - rho)
            x = np.log(y_prime)

            order_0 = alpha * (y / x)
            order_1 = order_0 * ((0.25 * rho * v * alpha + (2 - 3 * np.power(rho, 2)) * (np.power(v, 2) / 24)) * t)
            sigma = order_0 + order_1

        return sigma


@nb.jit("f8(f8,f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def sabr_normal_jit(f, k, alpha, rho, v, t):
    if f == k:
        return alpha * (1.0 + v * v * t * (2.0 - 3.0 * rho * rho) / 24.0)
    else:
        psi = (v / alpha) * (f - k)
        phi = np.log((np.sqrt(1.0 - 2.0 * rho * psi + psi * psi) - rho + psi) / (1.0 - rho))
        return alpha * (psi / phi) * (1.0 + v * v * t * (2.0 - 3.0 * rho * rho) / 24.0)


# @nb.jit("f8(f8,f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def sabr_normal_quadratic_jit(f, k, alpha, rho, v, t):
    sigma_n = sabr_normal_jit(f, k, alpha, rho, v, t)
    return sigma_n * (1.0 + v * v * t / 6.0)


# @nb.jit("f8(f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def sabr_normal_quadratic_swap_vol_jit(f, alpha, rho, v, t):
    sigma_n = sabr_normal_jit(f, f, alpha, rho, v, t)
    return sigma_n * (1.0 + (4.0 + 3.0 * rho * rho) * v * v * t / 24.0)


# @nb.jit("f8(f8,f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def sabr_normal_forward_adjusted(f, k, alpha, rho, v, t):
    sigma_q_k = sabr_normal_quadratic_jit(f, k, alpha, rho, v, t)
    sigma_q_f = sabr_normal_quadratic_jit(f, f, alpha, rho, v, t)
    if np.abs(f - k) < 1e-10:
        return f
    else:
        return f + 0.5 * t * (sigma_q_k * sigma_q_k - sigma_q_f * sigma_q_f) / (k - f)


def quadratic_european_normal_sabr(f, k, alpha, rho, v, t, option_type):
    f_adjusted = sabr_normal_forward_adjusted(f, k, alpha, rho, v, t)
    sigma_q = sabr_normal_quadratic_jit(f, k, alpha, rho, v, t)
    s_q = sabr_normal_quadratic_swap_vol_jit(f, alpha, rho, v, t)
    d = (f_adjusted - k) / (sigma_q * np.sqrt(t))

    f1 = np.power(f - k, 2.0) + s_q * s_q * t
    f2 = (f_adjusted - k) * sigma_q * np.sqrt(t)

    if option_type == 'c':
        return f1 * ndtr(d) + f2 * AnalyticTools.normal_pdf(0.0, 1.0, d)
    elif option_type == 'p':
        return f1 * ndtr(d) - f2 * AnalyticTools.normal_pdf(0.0, 1.0, d)
    elif option_type == 's':
        return f1
    else:
        return -1


# Tools for computing de derivative in local vol function in case SABR dynamic
@nb.jit("f8(f8,f8)", nopython=True, nogil=True)
def f_s(y, rho):
    return np.sqrt(1.0 - 2.0 * rho * y + np.multiply(y, y))


@nb.jit("f8(f8,f8)", nopython=True, nogil=True)
def f_l(y, rho):
    return np.log((f_s(y, rho) - rho + y) / (1.0 - rho))


@nb.jit("f8[:](f8,f8)", nopython=True, nogil=True)
def f_first_second_der(y, rho):
    out_derivative = np.zeros(2)

    if np.abs(y) < 1e-02:
        # derivative order 1
        c_0_order_1 = -0.5 * rho
        c_1_order_1 = 2.0 * (-0.25 * rho * rho + 1.0 / 6.0)
        c_2_order_1 = 0.125 * (6.0 * rho * rho - 5.0) * rho
        out_derivative[0] = c_0_order_1 + c_1_order_1 * y - c_2_order_1 * y * y

        # derivative order 2
        c_2_order_2 = 12.0 * (- (5.0 / 16.0) * np.power(rho, 4) + (rho * rho / 3.0) - (17.0 / 360.0))
        out_derivative[1] = c_1_order_1 - 2.0 * c_2_order_1 * y + c_2_order_2 * y * y

    else:
        f_l_ = f_l(y, rho)
        f_s_ = f_s(y, rho)

        a_u = f_l_ * f_s_ - y
        a_l = f_l_ * f_l_ * f_s_

        # numerical derivative
        # y_epsilon_right = y + 0.0001
        # y_epsilon_left = y - 0.0001
        #
        # f_epsilon_right = y_epsilon_right / f_l(y_epsilon_right, rho)
        # f_epsilon_left = y_epsilon_left / f_l(y_epsilon_left, rho)
        # f = (y / f_l_)

        # numerical_derive = (f_epsilon_right - f) / 0.0001
        # numerical_second_derive = (f_epsilon_right - 2.0 * f + f_epsilon_left) / (0.0001**2)

        out_derivative[0] = np.divide(a_u, a_l)  # first derivative y.

        b_u = f_l_ * (3.0 * rho * y - y * y - 2.0) + 2.0 * f_s_ * y
        b_l = np.power(f_l_ * f_s_, 3.0)

        out_derivative[1] = b_u / b_l  # second derivative y.

    return out_derivative


@nb.jit("f8(f8,f8)", nopython=True, nogil=True)
def x(rho, z):
    return np.log((np.sqrt((z - rho) ** 2 + (1.0 - rho * rho)) + z - rho) / (1.0 - rho))


@nb.jit("f8[:](f8,f8)", nopython=True, nogil=True)
def get_x_z_rho_derivative(rho, z):
    epsilon_rho = 0.0001
    out_derivative = np.zeros(3)
    out_derivative[0] = x(rho, z)  # value function in z and t
    out_derivative[1] = 1.0 / np.sqrt(1.0 - 2.0 * rho * z + z * z)
    out_derivative[2] = 0.5 * (x(rho + epsilon_rho, z) - x(rho - epsilon_rho, z)) / epsilon_rho  # partial rho

    return out_derivative


@nb.jit("f8[:](f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def f_partial_der_parameters(z, t, alpha, rho, nu):
    # 0 der with respect alpha
    # 1 der with respect nu
    # 2 der with respect rho

    out_derivative = np.zeros(4)

    epsilon_t = (0.25 * rho * nu * alpha + (2 - 3 * np.power(rho, 2)) * (np.power(nu, 2) / 24)) * t
    epsilon_t_nu = (0.25 * rho * alpha + (2.0 - 3.0 * rho * rho) * nu) * t / 12.0
    epsilon_t_alpha = 0.25 * rho * nu * t
    epsilon_t_rho = 0.25 * (nu * alpha - rho * nu * nu) * t
    z_cap = np.maximum(Types.MIN_VALUE_LOG_MONEYNESS, z) if z >= 0 \
        else np.minimum(- Types.MIN_VALUE_LOG_MONEYNESS, z)

    y = (nu / alpha) * z_cap
    x_z_rho = get_x_z_rho_derivative(rho, y)

    sigma_h = sabr_vol_jit(alpha, rho, nu, z_cap, t)

    # partial  alpha
    out_derivative[0] = (1 + epsilon_t) * (np.power(y / x_z_rho[0], 2.0) * x_z_rho[1]) + \
                        (alpha * y / x_z_rho[0]) * epsilon_t_alpha

    # partial  alpha
    out_derivative[0] = (1 + epsilon_t) * (np.power(y / x_z_rho[0], 2.0) * x_z_rho[1]) + \
                        (alpha * y / x_z_rho[0]) * epsilon_t_alpha

    # partial nu
    out_derivative[1] = (sigma_h - alpha * (1 + epsilon_t) * (np.power(y / x_z_rho[0], 2.0) * x_z_rho[1])) / nu + \
                        (alpha * y / x_z_rho[0]) * epsilon_t_nu

    # partial rho
    out_derivative[2] = - (alpha * y * x_z_rho[2] / (x_z_rho[0] * x_z_rho[0])) * (1 + epsilon_t) + \
                        (alpha * y / x_z_rho[0]) * epsilon_t_rho

    # partial t
    out_derivative[3] = alpha * (y / x_z_rho[0]) * (epsilon_t / t)

    return out_derivative


@nb.jit("f8[:](f8[:],f8[:],f8,f8,f8,f8,f8[:])", nopython=True, nogil=True)
def get_sabr_loc_vol(p_nu, p_rho, sigma_atm_t, partial_sigma_atm, t, f0_t, x_t):
    # x_t = log(f_t_T)
    # f_0_t forward at value_date with maturity t
    # z_t = log(f0_t/k_t) where k_t = exp(x_t - mu_t)

    # parameters at time t
    nu_t = ParameterTools.nu_sabr(p_nu, t)
    rho_t = ParameterTools.rho_sabr(p_rho, t)
    alpha_t = ParameterTools.alpha_atm_sabr(rho_t, nu_t, sigma_atm_t, t)

    no_paths = len(x_t)
    loc_vol = np.zeros(no_paths)
    epsilon_t = (0.25 * alpha_t * nu_t * rho_t + nu_t * nu_t * (2.0 - 3.0 * rho_t * rho_t) / 24.0) * t
    log_f0_t = np.log(f0_t)

    # derivative parameters with respect t
    partial_alpha_t = partial_sigma_atm
    partial_rho_t = ParameterTools.rho_sabr_partial_t(p_rho, t)
    partial_nu_t = ParameterTools.nu_sabr_partial_t(p_nu, t)

    for i in range(0, no_paths):
        k_t = np.exp(x_t[i])
        z_t = log_f0_t - x_t[i]
        y_t = (nu_t / alpha_t) * z_t
        sigma_t = sabr_vol_jit(alpha_t, rho_t, nu_t, z_t, t)
        first_second_f_der_t = f_first_second_der(y_t, rho_t)

        # We use the local vol transform to implied vols.
        # First and second derivative with respect to strike of the Hagan's implied vol formula.
        sigma_partial_k = - (nu_t / k_t) * first_second_f_der_t[0] * (1.0 + epsilon_t)
        sigma_partial_k_k = (nu_t / (k_t * k_t)) * (first_second_f_der_t[0] + (nu_t / alpha_t) *
                                                    first_second_f_der_t[1]) * (1.0 + epsilon_t)

        # Time derivative of the Hagan's implied vol formula.
        sigma_partial_parameters = f_partial_der_parameters(z_t, t, alpha_t, rho_t, nu_t)
        sigma_partial_t = sigma_partial_parameters[0] * partial_alpha_t + sigma_partial_parameters[1] * partial_nu_t + \
                          sigma_partial_parameters[2] * partial_rho_t + sigma_partial_parameters[3]

        if t < 5.0 / 365.0:
            t_adj = np.maximum(t, 1.0 / 365.0)
            loc_vol[i] = sabr_vol_jit(alpha_t, rho_t, nu_t, z_t, t_adj)
        else:
            loc_vol[i] = DupireFormulas.local_vol_from_implied_vol(t, sigma_t, z_t, k_t, sigma_partial_t,
                                                                   sigma_partial_k, sigma_partial_k_k)

    return loc_vol
