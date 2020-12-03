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


@nb.jit("c16[:](f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def get_cf(w, t, x, v, r_t, theta, rho, k, epsilon, b, u):
    a = k * theta

    p = u * 1j * w - 0.5 * w * w
    q = b - rho * epsilon * 1j * w
    r = 0.5 * epsilon * epsilon

    d = np.sqrt(q * q - 4.0 * p * r)
    aux_b_t = b - rho * epsilon * 1j * w + d
    g = aux_b_t / (aux_b_t - 2.0 * d)

    aux_b_t = b - rho * epsilon * 1j * w + d
    d_t = ((1.0 - np.exp(d * t)) / (1.0 - g * np.exp(d * t))) * aux_b_t / (epsilon * epsilon)

    aux_c_t = (1.0 - g * np.exp(d * t)) / (1.0 - g)
    c_t = r_t * 1j * t * w + (a / (epsilon * epsilon)) * (aux_b_t * t - 2.0 * np.log(aux_c_t))

    return np.exp(c_t + d_t * v + 1j * w * x)


@nb.jit("c16[:](f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def get_trap_cf(w, t, x, v, r_t, theta, rho, k, epsilon, b, u):
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

    return np.exp(c_t + d_t * v + 1j * w * x)


# nb.jit("f8[:](f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def f_heston(w, t, x, v, r_t, theta, rho, k, epsilon, b, u, strike):
    k_log = np.log(strike)
    y = get_trap_cf(np.asfortranarray(w), t, x, v, r_t, theta, rho, k, epsilon, b, u)
    return (np.cos(k_log * w) * y.imag - np.sin(k_log * w) * y.real) / w


@nb.jit("f8[:](f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def f_gamma_heston(w, t, x, v, r_t, theta, rho, k, epsilon, b, u, strike):
    k_log = np.log(strike)
    y = get_trap_cf(np.asfortranarray(w), t, x, v, r_t, theta, rho, k, epsilon, b, u)
    return np.cos(k_log * w) * y.real + np.sin(k_log * w) * y.imag


# @nb.jit("f8[:](f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def f_attari_heston(w, t, v, spot, r_t, theta, rho, k, epsilon, b, u, strike):
    df = np.exp(- r_t * t)
    x = np.log(spot)
    f2 = get_trap_cf(np.asfortranarray(w), t, x, v, r_t, theta, rho, k, epsilon, b, u)
    phi2 = f2 * np.exp(- 1j * w * (x + r_t * t))
    l = np.log(df * strike / spot)
    r2_u = phi2.real
    i2_u = phi2.imag
    a1_u = (r2_u + i2_u / w) * np.cos(w * l)
    a2_u = (i2_u - r2_u / w) * np.sin(w * l)

    return (a1_u + a2_u) / (1.0 + np.power(w, 2.0))


@nb.jit("f8[:](f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def f_delta_attari_heston(w, t, v, spot, r_t, theta, rho, k, epsilon, b, u, strike):
    df = np.exp(- r_t * t)
    x = np.log(spot)
    f2 = get_trap_cf(np.asfortranarray(w), t, x, v, r_t, theta, rho, k, epsilon, b, u)
    phi2 = f2 * np.exp(- 1j * w * (x + r_t * t))
    l = np.log(df * strike / spot)
    r2_u = phi2.real
    i2_u = phi2.imag
    a1_u = (r2_u + i2_u / w) * np.sin(w * l) * w
    a2_u = - (i2_u - r2_u / w) * np.cos(w * l) * w

    return (a1_u + a2_u) / ((1.0 + np.power(w, 2.0)) * spot)


@nb.jit("f8[:](f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def f_dual_delta_attari_heston(w, t, v, spot, r_t, theta, rho, k, epsilon, b, u, strike):
    df = np.exp(- r_t * t)
    x = np.log(spot)
    f2 = get_trap_cf(np.asfortranarray(w), t, x, v, r_t, theta, rho, k, epsilon, b, u)
    phi2 = f2 * np.exp(- 1j * w * (x + r_t * t))
    l = np.log(df * strike / spot)
    r2_u = phi2.real
    i2_u = phi2.imag
    a1_u = - (r2_u + i2_u / w) * np.sin(w * l)
    a2_u = (i2_u - r2_u / w) * np.cos(w * l)

    return w * (a1_u + a2_u) / (strike * (1.0 + np.power(w, 2.0)))


@nb.jit("f8[:](f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def f_gamma_attari_heston(w, t, v, spot, r_t, theta, rho, k, epsilon, b, u, strike):
    df = np.exp(- r_t * t)
    x = np.log(spot)
    f2 = get_trap_cf(np.asfortranarray(w), t, x, v, r_t, theta, rho, k, epsilon, b, u)
    phi2 = f2 * np.exp(- 1j * w * (x + r_t * t))
    l = np.log(df * strike / spot)
    r2_u = phi2.real
    i2_u = phi2.imag

    pow_spot = np.power(spot, 2.0)
    f_u_s = ((r2_u + i2_u / w) * np.sin(w * l) - (i2_u - r2_u / w) * np.cos(w * l)) * w / pow_spot
    d_f_u_s = ((r2_u + i2_u / w) * np.cos(w * l) + (i2_u - r2_u / w) * np.sin(w * l)) * w * w / pow_spot

    return - (d_f_u_s + f_u_s) / (1.0 + np.power(w, 2.0))
