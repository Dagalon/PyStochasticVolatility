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

from ncephes import ndtr


@nb.jit("f8(f8,f8,f8,f8,i8)", nopython=True, nogil=True)
def black_scholes(f0: float, k: float, sigma: float, t: float, is_call: int):
    sqtr_t = np.sqrt(t)
    d_1 = (np.log(f0 / k) / (sigma * sqtr_t)) + 0.5 * sigma * sqtr_t
    d_2 = d_1 - sigma * sqtr_t
    return f0 * ndtr(d_1) - k * ndtr(d_2)


@nb.jit("f8[:](f8[:], f8)", nopython=True, nogil=True)
def call_operator(x, strike):
    no_paths = len(x)
    results = np.empty(3)
    acum = 0.0
    acum_pow = 0.0
    acum_digital = 0.0

    for i in range(0, no_paths):
        index = 0.0
        if x[i] > strike:
            index = 1.0
        val = (x[i] - strike) * index
        acum += val
        acum_pow += val * val
        acum_digital += index

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)
    results[2] = acum_digital / no_paths

    return results


@nb.jit("f8[:](f8[:],f8,f8[:],f8,f8)", nopython=True, nogil=True)
def call_operator_control_variate(x, x0, vol_swap_t, k, t):
    no_paths = len(x)
    bs_prices = np.zeros(no_paths)
    v_prices = np.zeros(no_paths)

    results = np.empty(2)

    acum = 0.0
    acum_pow = 0.0

    for i in range(0, no_paths):
        bs_prices[i] = black_scholes(x0, k, vol_swap_t[i], t, 1)
        v_prices[i] = np.maximum(x[i] - k, 0.0)

    mean_bs_price = np.sum(bs_prices) / no_paths
    mean_v_price = np.sum(v_prices) / no_paths
    num_b = 0.0
    den_b = 0.0
    for i in range(0, no_paths):
        num_b += (v_prices[i] - mean_v_price) * (bs_prices[i] - mean_bs_price)
        den_b += (bs_prices[i] - mean_bs_price) * (bs_prices[i] - mean_bs_price)

    b_estimated = (num_b / den_b)

    for i in range(0, no_paths):
        value = v_prices[i] - b_estimated * (bs_prices[i] - mean_bs_price)
        acum += value
        acum_pow += value * value

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)

    return results


@nb.jit("f8[:](f8[:],f8,f8[:],f8,f8)", nopython=True, nogil=True)
def put_operator_control_variate(x, x0, vol_swap_t, k, t):
    no_paths = len(x)
    bs_prices = np.zeros(no_paths)
    v_prices = np.zeros(no_paths)

    results = np.empty(2)

    acum = 0.0
    acum_pow = 0.0

    for i in range(0, no_paths):
        bs_prices[i] = black_scholes(x0, k, vol_swap_t[i], t, -1)
        v_prices[i] = np.maximum(k - x[i], 0.0)

    mean_bs_price = np.sum(bs_prices) / no_paths
    mean_v_price = np.sum(v_prices) / no_paths
    num_b = 0.0
    den_b = 0.0
    for i in range(0, no_paths):
        num_b += (v_prices[i] - mean_v_price) * (bs_prices[i] - mean_bs_price)
        den_b += (bs_prices[i] - mean_bs_price) * (bs_prices[i] - mean_bs_price)

    b_estimated = (num_b / den_b)

    for i in range(0, no_paths):
        value = v_prices[i] - b_estimated * (bs_prices[i] - mean_bs_price)
        acum += value
        acum_pow += value * value

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)

    return results


@nb.jit("f8[:](f8[:], f8)", nopython=True, nogil=True)
def put_operator(x, strike):
    no_paths = len(x)
    results = np.empty(3)
    acum = 0.0
    acum_pow = 0.0
    acum_digital = 0.0

    for i in range(0, no_paths):
        index = 0.0
        if x[i] < strike:
            index = 1.0
        val = (strike - x[i]) * index
        acum += val
        acum_digital += index
        acum_pow += val * val

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)
    results[2] = acum_digital / no_paths

    return results


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def h_delta(x, delta, strike):
    if x < (strike - delta):
        return 0.0
    elif x > (strike + delta):
        return 1.0
    else:
        return 0.5 * (x - (strike - delta)) / delta


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def g_put_delta(x, delta, strike):
    if x < (strike - delta):
        return - (delta + (strike - delta - x))
    elif x > (strike + delta):
        return 0.0
    else:
        return - 0.25 * (strike + delta - x) * (strike + delta - x) / delta


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def g_call_delta(x, delta, strike):
    if x < (strike - delta):
        return 0.0
    elif x > (strike + delta):
        # return delta + (x - strike - delta)
        return x - strike
    else:
        return 0.25 * (x - (strike - delta)) * (x - (strike - delta)) / delta


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def f_call_delta(x, delta, strike):
    return np.maximum(x - strike, 0.0) - g_call_delta(x, delta, strike)


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def i_gamma(x, delta, strike):
    if np.abs(x - strike) < delta:
        return 0.5 / delta
    else:
        return 0.0


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def f_gamma(x, delta, strike):
    return np.maximum(x - strike, 0.0) - g_call_delta(x, delta, strike)


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def f_put_delta(x, delta, strike):
    return np.maximum(strike - x, 0.0) - g_put_delta(x, delta, strike)


@nb.jit("f8[:](f8[:], f8, f8, f8[:], f8)", nopython=True, nogil=True)
def malliavin_delta_call_put(x, strike, f0, weights, call_put):
    no_paths = len(x)
    results = np.empty(2)
    delta = 0.001
    # delta = 0.01 * strike
    acum = 0.0
    acum_pow = 0.0

    if call_put > 0:
        for i in range(0, no_paths):
            acum_part_1 = (x[i] / f0) * h_delta(x[i], delta, strike)
            acum_part_2 = f_call_delta(x[i], delta, strike) * weights[i]
            val = (acum_part_1 + acum_part_2)
            acum += val
            acum_pow += val * val

        results[0] = acum / no_paths
        results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)

        return results
    else:
        for i in range(0, no_paths):
            acum_part_1 = (x[i] / f0) * (1.0 - h_delta(x[i], delta, strike))
            acum_part_2 = f_put_delta(x[i], delta, strike) * weights[i]
            val = (acum_part_1 + acum_part_2)
            acum += val
            acum_pow += val * val

        results[0] = acum / no_paths
        results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)

        return results


@nb.jit("f8[:](f8[:], f8, f8, f8[:])", nopython=True, nogil=True)
def malliavin_gamma_call_put(x, strike, f0, gamma_weights):
    no_paths = len(x)
    results = np.empty(2)
    # delta = 0.01
    delta = 0.0001 * strike
    acum = 0.0
    acum_pow = 0.0

    for i in range(0, no_paths):
        acum_part_1 = np.power((x[i] / f0), 2.0) * i_gamma(x[i], delta, strike)
        acum_part_2 = gamma_weights[i] * f_gamma(x[i], delta, strike)
        val = (acum_part_1 + acum_part_2)
        acum += val
        acum_pow += val * val

    results[0] = acum / no_paths
    results[1] = np.sqrt((acum_pow / no_paths - results[0] * results[0]) / no_paths)

    return results



