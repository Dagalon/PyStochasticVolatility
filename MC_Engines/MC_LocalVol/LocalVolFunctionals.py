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


@nb.jit("f8[:](f8,f8[:],f8,f8)", nopython=True, nogil=True)
def cev_diffusion(t: float, x: Types.ndarray, beta: float, sigma: float):
    no_elements = len(x)
    output = np.zeros(no_elements)
    for i in range(0, no_elements):
        output[i] = sigma * np.power(x[i], beta)

    return output


@nb.jit("f8[:](f8,f8[:],f8,f8,f8,f8)", nopython=True, nogil=True)
def local_vol_normal_sabr(t: float, x: Types.ndarray, x0: float, alpha: float, rho: float, nu: float):
    no_elements = len(x)
    output = np.zeros(no_elements)
    for i in range(0, no_elements):
        y_i = (x[i] - x0) / alpha
        output[i] = alpha * np.sqrt(1.0 + 2.0 * rho * nu * y_i + nu * nu * y_i * y_i)

    return output


@nb.jit("f8[:](f8,f8[:],f8,f8)", nopython=True, nogil=True)
def first_derive_cev_diffusion(t: float, x: Types.ndarray, beta: float, sigma: float):
    no_elements = len(x)
    output = np.zeros(no_elements)
    for i in range(0, no_elements):
        output[i] = sigma * beta * np.power(x[i], beta - 1.0)

    return output


@nb.jit("f8[:](f8,f8[:],f8,f8)", nopython=True, nogil=True)
def second_derive_cev_diffusion(t: float, x: Types.ndarray, beta: float, sigma: float):
    no_elements = len(x)
    output = np.zeros(no_elements)
    for i in range(0, no_elements):
        output[i] = sigma * beta * (beta - 1.0) * np.power(x[i], beta - 2.0)

    return output


@nb.jit("f8[:](f8,f8[:],f8,f8)", nopython=True, nogil=True)
def log_cev_diffusion(t: float, x: Types.ndarray, beta: float, sigma: float):
    no_elements = len(x)
    output = np.zeros(no_elements)
    for i in range(0, no_elements):
        output[i] = sigma * np.power(np.exp(np.maximum(x[i], 0.00001)), beta)

    return output


@nb.jit("f8[:](f8,f8[:],f8,f8,f8,f8)", nopython=True, nogil=True)
def local_vol_log_normal_sabr(t: float, x: Types.ndarray, x0: float, alpha: float, rho: float, nu: float):
    no_elements = len(x)
    output = np.zeros(no_elements)
    for i in range(0, no_elements):
        y_i = np.log(x[i] / x0) / alpha
        output[i] = alpha * np.sqrt(1.0 + 2.0 * rho * nu * y_i + nu * nu * y_i * y_i)

    return output


@nb.jit("f8[:,:](f8,f8[:],f8,f8,f8,f8)", nopython=True, nogil=True)
def derivatives_local_vol_log_normal_sabr(t: float, x: Types.ndarray, x0: float, alpha: float, rho: float, nu: float):
    no_elements = len(x)
    output = np.zeros((no_elements, 3))
    for i in range(0, no_elements):
        y_i = np.log(x[i] / x0) / alpha
        y_i_prime = 1.0 / (x[i] * alpha)
        y_i_second_prime = - 1.0 / (x[i] * x[i] * alpha)

        # value
        output[i, 0] = alpha * np.sqrt(1.0 + 2.0 * rho * nu * y_i + nu * nu * y_i * y_i)

        # first derivative
        output[i, 1] = alpha * alpha * y_i_prime * (rho * nu + nu * nu * y_i) / output[i, 0]

        # second derivative
        output[i, 2] = (alpha * alpha * y_i_second_prime * (rho * nu + nu * nu * y_i) +
                        np.power(alpha * nu * y_i_prime, 2.0) - np.power(output[i, 1], 2.0)) / output[i, 0]

    return output
