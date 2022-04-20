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

# from ncephes import ndtr
from scipy.special import ndtr
from Tools.AnalyticTools import normal_pdf


@nb.jit("f8[:](f8[:], f8[:], f8, f8)", nopython=True, nogil=True)
def get_conditional_moment_order_one(alpha_t,
                                     alpha,
                                     nu,
                                     t):
    no_paths = len(alpha_t)
    mean_z = np.empty(no_paths)
    nu_t = nu * np.sqrt(t)
    mult = 0.5 * alpha * alpha * np.sqrt(t) / nu

    for i in range(0, no_paths):
        x_z = np.log(alpha_t[i] / alpha[i]) / nu_t
        d_right = x_z + nu_t
        d_left = x_z - nu_t

        mean_z[i] = mult[i] * (ndtr(d_right) - ndtr(d_left)) / normal_pdf(0.0, 1.0, d_right)

    return mean_z


@nb.jit("f8[:](f8[:], f8[:], f8, f8)", nopython=True, nogil=True)
def get_conditional_moment_order_two(alpha_t,
                                     alpha,
                                     nu,
                                     t):
    no_paths = len(alpha_t)
    moment_z = np.empty(no_paths)
    nu_t = nu * np.sqrt(t)
    mult = 0.25 * np.power(alpha, 4.0) * np.sqrt(t) / np.power(nu, 3.0)

    for i in range(0, no_paths):
        shift = (1.0 + np.power(alpha_t[i] / alpha[i], 2.0))
        x_z = np.log(alpha_t[i] / alpha[i]) / nu_t
        d_right = x_z + nu_t
        d_left = x_z - nu_t
        term_1 = mult[i] * shift * (ndtr(d_right) - ndtr(d_left)) / normal_pdf(0.0, 1.0, d_right)
        term_2 = mult[i] * (ndtr(d_right + nu_t) - ndtr(d_left - nu_t)) / normal_pdf(0.0, 1.0, d_right + nu_t)
        moment_z[i] = - term_1 + term_2

    return moment_z


@nb.jit("f8[:](f8[:], f8, f8[:], f8, f8[:])", nopython=True, nogil=True)
def get_variance(alpha,
                 nu,
                 alpha_t,
                 t,
                 z):

    no_paths = len(alpha_t)
    path = np.empty(no_paths)

    m_1 = get_conditional_moment_order_one(alpha_t, alpha, nu, t)
    m_2 = get_conditional_moment_order_two(alpha_t, alpha, nu, t)

    for i in range(0, no_paths):
        ln_mu = 2.0 * np.log(m_1[i]) - 0.5 * np.log(m_2[i])
        ln_sigma = np.sqrt(np.log(m_2[i]) - 2.0 * np.log(m_1[i]))
        path[i] = np.exp(ln_mu + ln_sigma * z[i])

    return path


@nb.jit("f8[:](f8, f8[:], f8[:], f8, f8, f8[:])", nopython=True, nogil=True)
def get_conditional_variance_t0(nu,
                                alpha_t0,
                                alpha_t1,
                                t0,
                                t1,
                                z):

    no_paths = len(alpha_t1)
    path = np.empty(no_paths)

    m_1 = get_conditional_moment_order_one(alpha_t1, alpha_t0, nu, t1 - t0)
    m_2 = get_conditional_moment_order_two(alpha_t1, alpha_t0, nu, t1 - t0)

    for i in range(0, no_paths):
        ln_mu = 2.0 * np.log(m_1[i]) - 0.5 * np.log(m_2[i])
        ln_sigma = np.sqrt(np.log(m_2[i]) - 2.0 * np.log(m_1[i]))
        path[i] = np.exp(ln_mu + ln_sigma * z[i])

    return path
