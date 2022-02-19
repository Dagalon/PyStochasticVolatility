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
# See the License for the specific language governing permissions and limitations under the License.
#

import numba as nb
import numpy as np
from Tools import Types
from scipy.special import beta
from scipy.integrate import quad_vec
from MC_Engines.MC_RBergomi import ToolsVariance


# @nb.jit("f8(f8[:],f8,f8,f8)", nopython=True, nogil=True)
def get_vol_swap_approximation_sabr(parameters: Types.ndarray, t0: float, t1: float, sigma_t0: float):
    alpha = parameters[0]
    nu = parameters[1]
    rho = parameters[2]

    return sigma_t0 * (1.0 + (nu * nu / 12.0 + 0.25 * (rho * nu * alpha) -
                       0.125 * np.power(rho * nu, 2.0)) * (t1 - t0))


@nb.jit("f8(f8[:],f8,f8,f8)", nopython=True, nogil=True)
def get_vol_swap_approximation_heston(parameters: Types.ndarray, t0: float, t1: float, sigma_t0: float):
    k = parameters[0]
    theta = parameters[1]
    epsilon = parameters[2]

    adjusment = 0.5 * (((0.5 * k * theta - epsilon * epsilon / 8.0) * (1.0 / sigma_t0)) - 0.5 * k * sigma_t0) + (
            epsilon * epsilon) / (48.0 * sigma_t0)

    return sigma_t0 + (t1 - t0) * adjusment


def get_vol_swap_local_vol(t0: float, t1: float, f0: float, lv_f0: float,
                           fd_lv_f0: float, sd_lv_f0: float):
    adjustment = f0 * f0 * (0.25 * sd_lv_f0 * lv_f0 + fd_lv_f0 * fd_lv_f0 * (0.25 * lv_f0 * lv_f0 - 2.0 / 3.0))
    return lv_f0 * (1.0 + (t1 - t0) * adjustment)


def get_iv_atm_local_vol_approximation(f0: float, lv_f0: float, fd_lv_f0: float, sd_lv_f0: float, t: float):
    vol_swap_approx = get_vol_swap_local_vol(0.0, t, f0, lv_f0, fd_lv_f0, sd_lv_f0)
    adjustment = - 0.125 * fd_lv_f0 * fd_lv_f0 * lv_f0 * f0 - (1.0 / 6.0) * sd_lv_f0 * lv_f0 * lv_f0 * f0 + \
                 (1.0 / 12.0) * fd_lv_f0 * lv_f0 * lv_f0

    return vol_swap_approx + f0 * t * adjustment


def get_variance_swap_rbergomi(parameters: Types.ndarray, sigma_0: float, t: float):
    nu = parameters[0]
    h = parameters[2]
    integral_result = quad_vec(lambda s: sigma_0 * sigma_0 * np.exp(nu * nu * np.power(s, 2.0 * h)), 0.0, t)
    return np.sqrt(integral_result[0] / t)


# tengo que corregir esto
def get_vol_swap_rbergomi(parameters: Types.ndarray, sigma_0: float, t: float):
    nu = parameters[0]
    h = parameters[2]

    h_1_2 = h + 0.5
    h_1 = h + 1.0

    adjustment = 0.125 * (nu * nu * h * sigma_0) / (h_1_2 * h_1_2 * h_1)
    return sigma_0 + adjustment * np.power(t, 2.0 * h)


@nb.jit("f8(f8[:],f8)", nopython=True, nogil=True)
def get_iv_atm_heston_approximation(parameters: Types.ndarray, t: float):
    epsilon = parameters[2]
    rho = parameters[3]
    v0 = parameters[4]
    sigma_0 = np.sqrt(v0)

    vol_swap = get_vol_swap_approximation_heston(parameters, 0.0, t, sigma_0)
    adjustment = np.power(epsilon * rho, 2.0) / (96.0 * sigma_0) + 0.125 * rho * epsilon * sigma_0

    return vol_swap + t * adjustment


# @nb.jit("f8(f8[:],f8,f8,f8,i1[:])", nopython=True, nogil=True)
def get_iv_atm_rbergomi_approximation(parameters: Types.ndarray, vol_swap: float, sigma_0: float, t: float,
                                      approximation_type: str = 'var_swap'):
    nu = parameters[0]
    rho = parameters[1]
    h = parameters[2]

    h_1_2 = (h + 0.5)
    h_3_2 = (h + 1.5)
    h_1 = (h + 1.0)

    if approximation_type == 'var_swap':
        part_1 = 3.0 / (h_3_2 * h_3_2)
        part_2 = 2.0 * beta(h_3_2, h_3_2)
        part_3 = 1.0 / h_1
        part_var_swap = 0.5 * nu * nu * h * sigma_0 / (h_1 * h_1_2 * h_1_2)
        # rho_term = 0.25 * rho * sigma_0 * sigma_0 * np.sqrt(2.0 * h) / (h_1_2 + h_3_2)

        adjustment = sigma_0 * np.power(nu * rho, 2.0) * (h / (h_1_2 * h_1_2)) * (part_1 - part_2 - part_3) \
                     - part_var_swap

    else:
        part_1 = 3.0 / (h_3_2 * h_3_2)
        part_2 = 2.0 * beta(h_3_2, h_3_2)
        part_3 = 1.0 / h_1

        adjustment = sigma_0 * np.power(nu * rho, 2.0) * (h / (h_1_2 * h_1_2)) * (part_1 - part_2 - part_3)

    return vol_swap + adjustment * np.power(t, 2.0 * h)


# @nb.jit("f8(f8[:],f8)", nopython=True, nogil=True)
def get_iv_atm_sabr_approximation(parameters: Types.ndarray, t: float):
    alpha = parameters[0]
    nu = parameters[1]
    rho = parameters[2]

    vol_swap = get_vol_swap_approximation_sabr(parameters, 0.0, t, alpha)
    adjustment = (0.25 * rho * nu * alpha * alpha - 0.125 * rho * rho * nu * nu * alpha)
    return vol_swap + t * adjustment


# function to compute VIX_t from MC simulation
@nb.jit("f8[:](f8,f8,f8,f8,f8,f8[:],f8,i8)", nopython=True, nogil=True)
def get_vix_rbergomi_t(t0, t1, delta_vix, nu, h, v_t, v0, no_integration_points):
    no_elements = len(v_t)
    vix_2_t = np.zeros(no_elements)
    t_i = np.linspace(t0, t1, no_integration_points)
    rho_s = np.zeros(no_integration_points)

    for j in range(0, no_integration_points):
        rho_s[j] = ToolsVariance.get_volterra_covariance(t0, t_i[j], h)

    for k in range(0, no_elements):
        for j in range(1, no_integration_points):
            d_i = (t_i[j] - t_i[j - 1])
            w_i_h = (np.log(v_t[k] / v_t[0]) + nu * nu * np.power(t0, 2.0 * h)) / (2.0 * nu)
            w_i_1 = 2.0 * nu * rho_s[j - 1] * (w_i_h - nu * rho_s[j - 1]) / (np.power(t0, 2.0 * h)) + nu * nu * np.power(t_i[j - 1], 2.0 * h)
            w_i = 2.0 * nu * rho_s[j] * (w_i_h - nu * rho_s[j]) / (np.power(t0, 2.0 * h)) + nu * nu * np.power(t_i[j], 2.0 * h)
            vix_2_t[k] += 0.5 * (np.exp(w_i_1) + np.exp(w_i)) * d_i * (v0 / delta_vix)

    return np.sqrt(vix_2_t)

