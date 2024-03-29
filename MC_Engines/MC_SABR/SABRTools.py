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
from Tools.AnalyticTools import dot_wise


@nb.jit("(f8,f8,f8[:],f8[:],f8[:],f8[:])", nopython=True, nogil=True)
def get_delta_weight(t_i_1, t_i, sigma_t_i_1, sigma_t_i, w_i_f, delta_weight):
    alpha1 = 1.0
    alpha2 = 0.0
    no_paths = len(w_i_f)
    sqr_delta_time = np.sqrt(t_i - t_i_1)
    for i in range(0, no_paths):
        delta_weight[i] += w_i_f[i] * sqr_delta_time * ((alpha1 / sigma_t_i_1[i]) + (alpha2 / sigma_t_i[i]))


@nb.jit("(f8,f8,f8[:],f8[:],f8[:], f8[:])", nopython=True, nogil=True)
def get_var_weight(t_i_1, t_i, sigma_t_i_1, sigma_t_i, w_i_f, var_weight):
    alpha1 = 1.0
    alpha2 = 0.0
    no_paths = len(w_i_f)
    sqr_delta_time = np.sqrt(t_i - t_i_1)
    for i in range(0, no_paths):
        v_i_i = sigma_t_i_1[i] * sigma_t_i_1[i]
        v_i = sigma_t_i[i] * sigma_t_i[i]
        var_weight[i] += w_i_f[i] * sqr_delta_time * ((alpha1 / v_i_i) + (alpha2 / v_i))


@nb.jit("(f8[:],f8[:],f8[:],f8,f8,f8[:])", nopython=True, nogil=True)
def get_gamma_weight(delta_weight, var_weight, inv_variance, rho, t,  gamma_weight):
    no_paths = len(delta_weight)
    rho_c = np.sqrt(1.0 - rho * rho)
    for i in range(0, no_paths):
        gamma_weight[i] = np.power(var_weight[i], 2.0) - inv_variance[i] - rho_c * t * delta_weight[i]


@nb.jit("f8[:](f8,f8,f8[:],f8[:],f8,f8)", nopython=True, nogil=True, parallel=True)
def get_integral_variance(t_i_1: float,
                          t_i: float,
                          sigma_t_i_1: ndarray,
                          sigma_t_i: ndarray,
                          w_i_1: float,
                          w_i: float):
    delta = (t_i - t_i_1)
    v_i_1 = sigma_t_i_1 * sigma_t_i_1
    v_i = sigma_t_i * sigma_t_i
    return delta * (w_i_1 * v_i_1 + w_i * v_i)


@nb.jit("f8[:](f8[:],f8[:],f8[:],f8,f8)", nopython=True, nogil=True, parallel=True)
def get_integral_sigma_w_t(diff_brownian: ndarray,
                           sigma_t_i_1: ndarray,
                           sigma_t_i: ndarray,
                           w_i_1: float,
                           w_i: float):

    return dot_wise((w_i_1 * sigma_t_i_1 + sigma_t_i * w_i), diff_brownian)
