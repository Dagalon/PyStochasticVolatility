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


@nb.jit("f8(f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def v_t_conditional_mean(k: float,
                         theta: float,
                         v_t_i_1: float,
                         t_i_1: float,
                         t_i: float):
    delta_time = (t_i - t_i_1)
    return theta + (v_t_i_1 - theta) * np.exp(- k * delta_time)


@nb.jit("f8(f8,f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def v_t_conditional_variance(k: float,
                             theta: float,
                             epsilon: float,
                             v_t_i_1: float,
                             t_i_1: float,
                             t_i: float):
    delta_time = (t_i - t_i_1)
    exp_k_t = (1.0 - np.exp(- k * delta_time)) / k
    return epsilon * epsilon * (v_t_i_1 * np.exp(- k * delta_time) * exp_k_t + 0.5 * theta * k * exp_k_t * exp_k_t)


@nb.jit("f8[:](f8,f8)", nopython=True, nogil=True)
def matching_qe_moments_qg(m: float,
                           s2: float):
    parameters = np.empty(2)

    phi_t_i = s2 / (m * m)
    inv_phi_t_i = 1.0 / phi_t_i
    aux_b_2 = 2.0 * inv_phi_t_i - 1.0
    parameters[0] = np.sqrt(aux_b_2 + np.sqrt(aux_b_2 * (aux_b_2 + 1.0)))
    parameters[1] = m / (1.0 + np.power(parameters[0], 2.0))

    return parameters


@nb.jit("f8[:](f8,f8)", nopython=True, nogil=True)
def matching_qe_moments_exp(m: float,
                            s2: float):
    parameters = np.empty(2)

    phi_t_i = s2 / (m * m)
    parameters[0] = (phi_t_i - 1.0) / (phi_t_i + 1.0)
    parameters[1] = (1.0 - parameters[0]) / m

    return parameters


@nb.jit("f8(f8,f8,f8)", nopython=True, nogil=True)
def inv_exp_heston(p: float,
                   beta: float,
                   u: float):
    if u < p:
        return 0.00001
    else:
        return np.log((1.0 - p) / (1.0 - u)) / beta


@nb.jit("f8[:](f8,f8,f8[:],f8[:],f8,f8)", nopython=True, nogil=True)
def get_integral_variance(t_i_1: float,
                          t_i: float,
                          v_t_i_1: ndarray,
                          v_t_i: ndarray,
                          w_i_1: float,
                          w_i: float):
    delta = (t_i - t_i_1)
    return delta * (w_i_1 * v_t_i_1 + w_i * v_t_i)


@nb.jit("(f8,f8,f8[:],f8[:],f8[:],f8[:])")
def get_delta_weight(t_i_1, t_i, v_t_i_1, v_t_i, w_i_f, delta_weight):
    alpha1 = 1.0
    alpha2 = 0.0
    no_paths = len(w_i_f)
    sqr_delta_time = np.sqrt(t_i - t_i_1)
    for i in range(0, no_paths):
        delta_weight[i] += w_i_f[i] * sqr_delta_time * ((alpha1 / np.sqrt(v_t_i_1[i])) + (alpha2 / np.sqrt(v_t_i[i])))


@nb.jit("(f8,f8,f8[:],f8[:],f8[:], f8[:])")
def get_var_weight(t_i_1, t_i, v_t_i_1, v_t_i, w_i_f,  var_weight):
    alpha1 = 0.5
    alpha2 = 0.5
    no_paths = len(w_i_f)
    sqr_delta_time = np.sqrt(t_i - t_i_1)
    for i in range(0, no_paths):
        var_weight[i] += w_i_f[i] * sqr_delta_time * ((alpha1 / np.sqrt(v_t_i_1[i])) + (alpha2 / np.sqrt(v_t_i[i])))


# @nb.jit("(f8[:],f8[:],f8[:],f8,f8,f8[:])")
def get_gamma_weight(delta_weight, var_weight, inv_variance, rho, t,  gamma_weight):
    no_paths = len(delta_weight)
    rho_c = np.sqrt(1.0 - rho * rho)
    for i in range(0, no_paths):
        gamma_weight[i] = np.power(var_weight[i], 2.0) - inv_variance[i] - rho_c * t * delta_weight[i]



