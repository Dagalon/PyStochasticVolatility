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


# We suppose that the distributions of the jumps is exp(N(jumpmean, jumpstd))
@nb.jit("c16[:](f8[:], f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def get_merton_cf(w, t, x, sigma, jumpmean, jumpstd, lambda_t):
    jumpmean_transform = jumpmean - 0.5 * jumpstd * jumpstd
    alpha = (np.exp(jumpmean_transform + 0.5 * jumpstd * jumpstd) - 1.0) * lambda_t * t
    x_i_u = (x - 0.5 * sigma * sigma * t) * 1j * w
    x_u = - 0.5 * np.power(w * sigma, 2.0) * t
    nu_t = (np.exp((1j * w * jumpmean_transform - 0.5 * np.power(w * jumpstd, 2.0))) - 1.0) * lambda_t * t - alpha
    return np.exp(x_i_u + x_u + nu_t)


@nb.jit("c16[:](f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def get_bates_cf(w, t, x, v, r_t, theta, rho, k, epsilon, jump_mean, jump_std, jump_intensity, b, u):
    a = k * theta

    p = u * 1j * w - 0.5 * w * w
    q = b - rho * epsilon * 1j * w
    r = 0.5 * epsilon * epsilon

    d = np.sqrt(q * q - 4.0 * p * r)
    aux_b_t = b - rho * epsilon * 1j * w + d
    c = (aux_b_t - 2.0 * d) / aux_b_t

    d_t = ((1.0 - np.exp(-d * t)) / (1.0 - c * np.exp(-d * t))) * (aux_b_t - 2.0 * d) / (epsilon * epsilon)

    aux_c_t = (1.0 - c * np.exp(-d * t)) / (1.0 - c)
    c_t = r_t * 1j * t * w + (a / (epsilon * epsilon)) * ((aux_b_t - 2.0 * d) * t - 2.0 * np.log(aux_c_t))

    jump_var = 0.5 * jump_std * jump_std
    aux_jump1 = np.exp(jump_mean + jump_var) - 1
    aux_jump2 = np.exp(jump_mean * 1j * w-jump_var * w * w)-1
    jump_term = jump_intensity * t * (aux_jump2 - 1j * w * aux_jump1)

    return np.exp(jump_term+c_t + d_t * v + 1j * w * x)


# @nb.jit("c16(c16, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def h_bates_lewis(w, v, t, k, theta, epsilon, rho):
    var = epsilon * epsilon
    b = (2 / var) * (1j * w * rho * epsilon + k)
    ksi = np.sqrt((b * b + 4 * (w * w - 1j * w) / var))
    g = 0.5 * (b - ksi)
    h = (b - ksi) / (b + ksi)
    variance_t = 0.5 * var * t

    aux1 = np.exp(-ksi * variance_t)
    aux2 = np.log((1 - h * aux1) / (1 - h))

    return np.exp(2 * k * theta / var * (variance_t * g - aux2) + v * g * (1 - aux1) / (1 - h * aux1))


# @nb.jit("f8(c16, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def f_lewis_bate(w, t, spot, v, r_t, theta, rho, k, epsilon, lambda_jump, mu_jump, sigma_jump, strike):
    w_shift = w + 0.5 * 1j
    beta = np.exp(mu_jump + 0.5 * sigma_jump * sigma_jump) - 1
    phi_hat = np.exp(-1j * mu_jump * w_shift - 0.5 * w_shift * w_shift * sigma_jump * sigma_jump)
    aux1 = np.log(spot / strike) + (r_t - lambda_jump * beta) * t
    aux2 = lambda_jump * t * (phi_hat - 1)
    value = np.exp(-1j * w_shift * aux1 + aux2) * h_bates_lewis(w_shift, v, t, k, theta, epsilon, rho) / (w_shift ** 2 - 1j * w_shift)
    return value.real






