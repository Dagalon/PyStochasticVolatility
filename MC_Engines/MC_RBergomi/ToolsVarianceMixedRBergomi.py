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

import numpy as np
import numba as nb

from Tools.Types import ndarray
from Tools import AnalyticTools
from ncephes import hyp2f1
from MC_Engines.MC_RBergomi.ToolsVariance import get_volterra_covariance, get_covariance_w_v_w_t


@nb.jit("f8[:](f8[:], f8, f8)", nopython=True, nogil=True)
def get_variance(t: float, h_short: float, h_long: float):
    return np.power(t, 2.0 * h_short) + np.power(t, 2.0 * h_long) + \
           (4.0 / (h_short + h_long)) * np.sqrt(h_short * h_long) * np.power(t, h_short + h_long)


@nb.jit("f8[:](f8[:], f8)", nopython=True, nogil=True)
def get_variance_rbergomi(t: ndarray, h: float):
    return np.power(t, 2.0 * h)


@nb.jit("f8(f8, f8, f8, f8)", nopython=True, nogil=True)
def get_mixed_term_covariance(s: float, t: float, h_short: float, h_long: float):
    if t >= s:
        x = s / t
        output = 2.0 * np.sqrt(h_short * h_long) * hyp2f1(0.5 - h_short, 1.0, h_long + 1.5, x) * \
                 np.power(t, -0.5 + h_short) * np.power(s, 0.5 + h_long) / (h_long + 0.5)
    else:
        x = t / s
        output = 2.0 * np.sqrt(h_short * h_long) * hyp2f1(0.5 - h_long, 1.0, h_short + 1.5, x) * \
                 np.power(s, -0.5 + h_long) * np.power(t, 0.5 + h_short) / (h_short + 0.5)

    return output


@nb.jit("f8(f8, f8, f8, f8)", nopython=True, nogil=True)
def get_w_s_t_covariance(s: float, t: float, h_short: float, h_long: float):
    x = s / t
    term_short = get_volterra_covariance(s, t, h_short)
    term_long = get_volterra_covariance(s, t, h_long)
    term_mixed_1 = get_mixed_term_covariance(s, t, h_short, h_long)
    term_mixed_2 = get_mixed_term_covariance(t, s, h_short, h_long)

    return term_short + term_long + term_mixed_1 + term_mixed_2


@nb.jit("f8(f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def get_w_s_b_t_covariance(s: float, t: float, h_short: float, h_long: float, rho: float):
    term_short = get_covariance_w_v_w_t(s, t, rho, h_short)
    term_long = get_covariance_w_v_w_t(s, t, rho, h_long)
    return term_short + term_long


@nb.jit("f8[:,:](f8[:], f8, f8, f8)", nopython=True, nogil=True)
def get_covariance_matrix(t_i_s: ndarray, h_short: float, h_long: float, rho: float):
    no_time_steps = len(t_i_s)
    cov = np.zeros(shape=(3 * no_time_steps, 3 * no_time_steps))

    # First row
    for i in range(0, no_time_steps):
        for j in range(0, no_time_steps):
            cov[i, j] = np.minimum(t_i_s[i], t_i_s[j])

    for i in range(0, no_time_steps):
        for j in range(no_time_steps, 2 * no_time_steps):
            cov[i, j] = get_covariance_w_v_w_t(t_i_s[j - no_time_steps], t_i_s[i], rho, h_short)

    for i in range(0, no_time_steps):
        for j in range(2 * no_time_steps, 3 * no_time_steps):
            cov[i, j] = get_covariance_w_v_w_t(t_i_s[j - 2 * no_time_steps], t_i_s[i], rho, h_long)

    # Second row
    for i in range(no_time_steps, 2 * no_time_steps):
        for j in range(0, no_time_steps):
            cov[i, j] = get_covariance_w_v_w_t(t_i_s[i - no_time_steps], t_i_s[j], rho, h_short)

    for i in range(no_time_steps, 2 * no_time_steps):
        for j in range(no_time_steps, 2 * no_time_steps):
            cov[i, j] = get_volterra_covariance(t_i_s[i - no_time_steps], t_i_s[j - no_time_steps], h_short)

    for i in range(no_time_steps, 2 * no_time_steps):
        for j in range(2 * no_time_steps, 3 * no_time_steps):
            cov[i, j] = get_mixed_term_covariance(t_i_s[i - no_time_steps], t_i_s[j - 2 * no_time_steps], h_short, h_long)

    # Third row
    for i in range(2 * no_time_steps, 3 * no_time_steps):
        for j in range(0, no_time_steps):
            cov[i, j] = get_covariance_w_v_w_t(t_i_s[i - 2 * no_time_steps], t_i_s[j], rho, h_long)

    for i in range(2 * no_time_steps, 3 * no_time_steps):
        for j in range(no_time_steps, 2 * no_time_steps):
            cov[i, j] = get_mixed_term_covariance(t_i_s[i - 2 * no_time_steps], t_i_s[j - no_time_steps], h_short, h_long)

    for i in range(2 * no_time_steps, 3 * no_time_steps):
        for j in range(2 * no_time_steps, 3 * no_time_steps):
            cov[i, j] = get_volterra_covariance(t_i_s[i - 2 * no_time_steps], t_i_s[j - 2 * no_time_steps], h_long)

    return cov


@nb.jit("(f8, f8, f8, f8, f8, f8, f8[:,:], f8[:,:], f8[:], i8)", nopython=True, nogil=True)
def generate_paths_mixed_rbergomi(s0: float,
                                  sigma_0: float,
                                  nu_short: float,
                                  nu_long: float,
                                  h_short: float,
                                  h_long: float,
                                  noise: ndarray,
                                  cholk_cov: ndarray,
                                  t_i_s: ndarray,
                                  no_paths: int):
    no_time_steps = len(t_i_s)

    paths = np.zeros(shape=(no_paths, no_time_steps))
    int_v_t = np.zeros(shape=(no_paths, no_time_steps - 1))
    int_v_short_t = np.zeros(shape=(no_paths, no_time_steps - 1))
    int_v_long_t = np.zeros(shape=(no_paths, no_time_steps - 1))

    sigma_i_1 = np.zeros(shape=(no_paths, no_time_steps))
    sigma_short_i_1 = np.zeros(shape=(no_paths, no_time_steps))
    sigma_long_i_1 = np.zeros(shape=(no_paths, no_time_steps))

    sigma_short_i_1[:, 0] = 0.5 * sigma_0
    sigma_long_i_1[:, 0] = 0.5 * sigma_0
    sigma_i_1[:, 0] = sigma_0
    paths[:, 0] = s0

    # we compute before a loop of variance of the variance process
    var_w_t_short = get_variance_rbergomi(t_i_s[1:], h_short)
    var_w_t_long = get_variance_rbergomi(t_i_s[1:], h_long)

    for k in range(0, no_paths):
        w_t_k = AnalyticTools.apply_lower_tridiagonal_matrix(cholk_cov, noise[:, k])

        w_i_s_1 = 0.0
        w_i_h_short_1 = 0.0
        w_i_h_long_1 = 0.0
        var_short_w_t_i_1 = 0.0
        var_long_w_t_i_1 = 0.0

        for j in range(1, no_time_steps):
            delta_i_s = t_i_s[j] - t_i_s[j - 1]

            # Brownian and Gaussian increments
            d_w_i_s = w_t_k[j - 1] - w_i_s_1
            d_w_i_h_short = w_t_k[j + no_time_steps - 2] - w_i_h_short_1
            d_w_i_h_long = w_t_k[j + 2 * no_time_steps - 3] - w_i_h_long_1

            sigma_short_i_1[k, j] = sigma_short_i_1[k, j - 1] * np.exp(
                - 0.5 * nu_short * nu_short * (var_w_t_short[j - 1] - var_short_w_t_i_1) +
                nu_short * d_w_i_h_short)

            sigma_long_i_1[k, j] = sigma_long_i_1[k, j - 1] * np.exp(
                - 0.5 * nu_long * nu_long * (var_w_t_long[j - 1] - var_long_w_t_i_1) +
                nu_long * d_w_i_h_long)

            sigma_i_1[k, j] = sigma_short_i_1[k, j] + sigma_long_i_1[k, j]

            int_v_short_t[k, j - 1] = delta_i_s * 0.5 * (sigma_short_i_1[k, j - 1] * sigma_short_i_1[k, j - 1] +
                                                         sigma_short_i_1[k, j] * sigma_short_i_1[k, j])

            int_v_long_t[k, j - 1] = delta_i_s * 0.5 * (sigma_long_i_1[k, j - 1] * sigma_long_i_1[k, j - 1] +
                                                        sigma_long_i_1[k, j] * sigma_long_i_1[k, j])

            int_v_t[k, j - 1] = delta_i_s * sigma_i_1[k, j] * sigma_i_1[k, j]

            paths[k, j] = paths[k, j - 1] * np.exp(- 0.5 * int_v_t[k, j - 1] +
                                                   sigma_i_1[k, j - 1] * d_w_i_s)

            # Keep the last brownians and variance of the RL process
            w_i_s_1 = w_t_k[j - 1]
            w_i_h_short_1 = w_t_k[j + no_time_steps - 2]
            w_i_h_long_1 = w_t_k[j + 2 * no_time_steps - 3]

            var_short_w_t_i_1 = var_w_t_short[j - 1]
            var_long_w_t_i_1 = var_w_t_long[j - 1]

    return paths, sigma_i_1, int_v_t
