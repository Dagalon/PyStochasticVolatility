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

from MC_Engines.GenericSDE.SDESimulation import sde_euler_simulation
from MC_Engines.GenericSDE.SDE import bs_sigma_flat, bs_drift_flat
from Tools.Types import ndarray, EULER_SCHEME_TYPE
from typing import Callable
from functools import partial
from py_vollib.black_scholes_merton.greeks import analytical, numerical
from py_vollib.black_scholes_merton import black_scholes_merton


def get_malliavin_greeks_bs_flat(s0: float,
                                 t: float,
                                 no_steps: int,
                                 no_paths: int,
                                 r: float,
                                 q: float,
                                 sigma: float,
                                 payoff: Callable[[ndarray], ndarray],
                                 euler_scheme: EULER_SCHEME_TYPE):
    shift_delta = 0.01
    shift_vega = 0.01

    z = np.random.standard_normal(size=(no_paths, no_steps - 1))
    drift_t = partial(bs_drift_flat, rate_t=r, dividend_t=q)
    sigma_t = partial(bs_sigma_flat, sigma_t=sigma)
    sigma_t_shift = partial(bs_sigma_flat, sigma_t=sigma + shift_vega)

    s_t = sde_euler_simulation(0.0, t, s0, no_steps, no_paths, z, drift_t, sigma_t, euler_scheme)

    s_t_shift_delta = sde_euler_simulation(0.0, t, s0 + shift_delta, no_steps, no_paths, z, drift_t, sigma_t,
                                           euler_scheme)

    s_t_shift_vega = sde_euler_simulation(0.0, t, s0, no_steps, no_paths, z, drift_t, sigma_t_shift, euler_scheme)

    phi_t = payoff(s_t[:, -1])
    mc_price = np.average(phi_t) * np.exp(- r * t)
    mc_price_delta = np.average(payoff(s_t_shift_delta[:, -1])) * np.exp(- r * t)
    mc_price_vega = np.average(payoff(s_t_shift_vega[:, -1])) * np.exp(- r * t)

    delta = analytical.delta('c', s0, 90, t, r, sigma, q)
    price = black_scholes_merton('c', s0, 90, t, r, sigma, q)
    vega = analytical.vega('c', s0, 90, t, r, sigma, q) * 100

    mc_delta = (mc_price_delta - mc_price) / shift_delta
    malliavin_delta = get_malliavin_delta_bs_flat(phi_t, s_t[:, -1], s0, sigma, r, q, t)

    mc_vega = (mc_price_vega - mc_price) / shift_vega
    malliavin_vega = get_malliavin_vega_bs_flat(phi_t, s_t[:, -1], s0, sigma, r, q, t)

    return mc_delta, malliavin_delta


@nb.jit("f8(f8[:], f8[:], f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def get_malliavin_delta_bs_flat(phi_t: ndarray,
                                s_t: ndarray,
                                s0: float,
                                sigma: float,
                                r: float,
                                q: float,
                                t: float):
    no_paths = len(s_t)
    malliavin_delta = 0.0
    f_t = s0 * np.exp((r - q) * t)

    for i in range(0, no_paths):
        w_T_i = (np.log(s_t[i] / f_t) + 0.5 * sigma * sigma * t) / sigma
        malliavin_delta += (phi_t[i] * w_T_i) / (sigma * s0 * t)

    return np.exp(- r * t) * malliavin_delta / no_paths


@nb.jit("f8(f8[:], f8[:], f8, f8, f8, f8, f8)", nopython=True, nogil=True)
def get_malliavin_vega_bs_flat(phi_t: ndarray,
                               s_t: ndarray,
                               s0: float,
                               sigma: float,
                               r: float,
                               q: float,
                               t: float):
    no_paths = len(s_t)
    malliavin_vega = 0.0
    f_t = s0 * np.exp((r - q) * t)

    for i in range(0, no_paths):
        w_T_i = (np.log(s_t[i] / f_t) + 0.5 * sigma * sigma * t) / sigma
        malliavin_vega += ((w_T_i * w_T_i) / (sigma * t) - w_T_i - (1.0 / sigma)) * phi_t[i]

    return np.exp(- r * t) * malliavin_vega / no_paths
