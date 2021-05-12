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

from MC_Engines.MC_SABR import VarianceSamplingMatchingMoment
from Tools import AnalyticTools
from Tools.Types import Vector, ndarray, SABR_OUTPUT, TYPE_STANDARD_NORMAL_SAMPLING
from MC_Engines.MC_SABR import SABRTools


def get_path_one_step(t0: float,
                      t1: float,
                      parameters: Vector,
                      f0: float,
                      no_paths: int,
                      rnd_generator) -> Vector:
    alpha = parameters[0]
    nu = parameters[1]
    rho = parameters[2]

    z = rnd_generator.normal(0.0, 1.0, no_paths, sampling_type=TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC)
    z_int = rnd_generator.normal(0.0, 1.0, no_paths, sampling_type=TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC)

    alpha_t = get_vol_sampling(t0,
                               t1,
                               alpha,
                               nu,
                               z)

    var_t0_t1 = VarianceSamplingMatchingMoment.get_variance(np.full(no_paths, alpha, dtype=np.float),
                                                            nu,
                                                            alpha_t,
                                                            t1,
                                                            z_int)

    return get_underlying_sampling(f0,
                                   alpha,
                                   rho,
                                   nu,
                                   var_t0_t1,
                                   alpha_t,
                                   no_paths,
                                   rnd_generator)


def get_vol_sampling(t0: float,
                     t1: float,
                     sigma_i_1: ndarray,
                     nu: float,
                     z: ndarray) -> ndarray:
    delta_time = (t1 - t0)
    sqrt_delta_time = np.sqrt(delta_time)
    drift = sigma_i_1 * np.exp(- 0.5 * nu * nu * delta_time)
    noise = np.exp(nu * sqrt_delta_time * z)
    return drift * noise


def get_time_steps(t0: float, t1: float, no_time_steps: int, **kwargs):
    if len(kwargs) > 0:
        extra_points = kwargs['extra_sampling_points']
        basis_sampling_dates = np.linspace(t0, t1, no_time_steps).tolist()
        full_points = np.array(list(set(extra_points + basis_sampling_dates)))
        return sorted(full_points)
    else:
        return np.linspace(t0, t1, no_time_steps)


def get_underlying_sampling(f0: float,
                            alpha: float,
                            rho: float,
                            nu: float,
                            var_t0_t1: Vector,
                            alpha_t: Vector,
                            no_paths: int,
                            rnd_generator):

    z = rnd_generator.normal(size=no_paths)
    return get_jit_paths(f0, var_t0_t1, alpha_t, alpha, nu, rho, z)


@nb.jit("f8[:](f8,f8[:],f8[:],f8,f8,f8,f8[:])", nopython=True, nogil=True)
def get_jit_paths(f0, var_t0_t1, alpha_t, alpha, nu, rho, z):
    no_paths = len(var_t0_t1)
    f_t = np.empty(no_paths)
    rho_inv = np.sqrt(1.0 - rho * rho)

    for i in range(0, no_paths):
        mu = np.log(f0) - 0.5 * var_t0_t1[i] + (rho / nu) * (alpha_t[i] - alpha)
        sigma = rho_inv * np.sqrt(var_t0_t1[i])
        f_t[i] = np.exp(mu + sigma * z[i])

    return f_t


def get_path_multi_step(t0: float,
                        t1: float,
                        parameters: Vector,
                        f0: float,
                        no_paths: int,
                        no_time_steps: int,
                        type_random_number: TYPE_STANDARD_NORMAL_SAMPLING,
                        rnd_generator,
                        **kwargs) -> map:

    alpha = parameters[0]
    nu = parameters[1]
    rho = parameters[2]
    rho_inv = np.sqrt(1.0 - rho * rho)

    no_paths = 2 * no_paths if type_random_number == TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

    t_i = get_time_steps(t0, t1, no_time_steps, **kwargs)
    no_time_steps = len(t_i)

    delta_t_i = np.diff(t_i)

    s_t = np.empty((no_paths, no_time_steps))
    sigma_t = np.empty((no_paths, no_time_steps))
    int_v_t_paths = np.zeros(shape=(no_paths, no_time_steps - 1))

    s_t[:, 0] = f0
    sigma_t[:, 0] = alpha

    int_sigma_t_i = np.empty((no_paths, no_time_steps - 1))
    sigma_t_i_1 = np.empty(no_paths)
    sigma_t_i_1[:] = alpha

    delta_weight = np.zeros(no_paths)
    gamma_weight = np.zeros(no_paths)
    var_weight = np.zeros(no_paths)
    inv_variance = np.zeros(no_paths)

    map_output = {}

    for i_step in range(1, no_time_steps):
        z_i = rnd_generator.normal(0.0, 1.0, no_paths, type_random_number)
        z_sigma = rnd_generator.normal(0.0, 1.0, no_paths, type_random_number)
        sigma_t_i = get_vol_sampling(t_i[i_step - 1], t_i[i_step], sigma_t_i_1, nu, z_sigma)
        sigma_t[:, i_step] = sigma_t_i

        int_sigma_t_i[:, i_step - 1] = 0.5 * (sigma_t_i_1 * sigma_t_i_1 * delta_t_i[i_step - 1] +
                                              sigma_t_i * sigma_t_i * delta_t_i[i_step - 1])

        diff_sigma = (rho / nu) * (sigma_t_i - sigma_t_i_1)
        noise_sigma = AnalyticTools.dot_wise(np.sqrt(int_sigma_t_i[:, i_step - 1]), z_i)
        SABRTools.get_delta_weight(t_i[i_step - 1], t_i[i_step], sigma_t_i_1, sigma_t_i, z_sigma, delta_weight)
        SABRTools.get_var_weight(t_i[i_step - 1], t_i[i_step], sigma_t_i_1, sigma_t_i, z_sigma, var_weight)

        inv_variance += SABRTools.get_integral_variance(t_i[i_step - 1], t_i[i_step], 1.0 / sigma_t_i_1,
                                                        1.0 / sigma_t_i, 0.5, 0.5)

        np.copyto(int_v_t_paths[:, i_step - 1], SABRTools.get_integral_variance(t_i[i_step - 1], t_i[i_step],
                                                                                sigma_t_i_1, sigma_t_i, 0.5, 0.5))

        sigma_t_i_1 = sigma_t_i.copy()
        s_t[:, i_step] = AnalyticTools.dot_wise(s_t[:, i_step - 1],
                                                np.exp(- 0.5 * int_sigma_t_i[:, i_step - 1] +
                                                diff_sigma + rho_inv * noise_sigma))

    map_output[SABR_OUTPUT.DELTA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL] = delta_weight
    map_output[SABR_OUTPUT.PATHS] = s_t
    map_output[SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS] = int_v_t_paths
    map_output[SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS] = int_sigma_t_i
    map_output[SABR_OUTPUT.SIGMA_PATHS] = sigma_t
    map_output[SABR_OUTPUT.TIMES] = t_i

    SABRTools.get_gamma_weight(delta_weight, var_weight, inv_variance, rho, t1, gamma_weight)
    map_output[SABR_OUTPUT.GAMMA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL] = np.multiply(gamma_weight, 1.0 / (
                (1.0 - rho * rho) * np.power(t1 * f0, 2.0)))

    return map_output
