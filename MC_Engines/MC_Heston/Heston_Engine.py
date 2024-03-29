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

from Tools.Types import Vector, ndarray, HESTON_OUTPUT
from MC_Engines.MC_Heston import HestonTools, VarianceMC
from Tools import AnalyticTools, Types


def get_time_steps(t0: float, t1: float, no_time_steps: int, **kwargs):
    if len(kwargs) > 0:
        extra_points = kwargs['extra_sampling_points']
        basis_sampling_dates = np.linspace(t0, t1, no_time_steps).tolist()
        full_points = np.array(list(set(extra_points + basis_sampling_dates)))
        return sorted(full_points)
    else:
        return np.linspace(t0, t1, no_time_steps)


def get_path_multi_step(t0: float,
                        t1: float,
                        parameters: Vector,
                        f0: float,
                        v0: float,
                        no_paths: int,
                        no_time_steps: int,
                        type_random_numbers: Types.TYPE_STANDARD_NORMAL_SAMPLING,
                        rnd_generator,
                        **kwargs) -> ndarray:

    k = parameters[0]
    theta = parameters[1]
    epsilon = parameters[2]
    rho = parameters[3]

    no_paths = 2 * no_paths if type_random_numbers == Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

    t_i = get_time_steps(t0, t1, no_time_steps, **kwargs)
    no_time_steps = len(t_i)

    t_i = np.linspace(t0, t1, no_time_steps)
    delta_t_i = np.diff(t_i)

    f_t = np.empty((no_paths, no_time_steps))
    f_t[:, 0] = f0

    delta_weight = np.zeros(no_paths)
    gamma_weight = np.zeros(no_paths)
    var_weight = np.zeros(no_paths)
    inv_variance = np.zeros(no_paths)

    ln_x_t_paths = np.zeros(shape=(no_paths, no_time_steps))
    int_v_t_paths = np.zeros(shape=(no_paths, no_time_steps - 1))
    v_t_paths = np.zeros(shape=(no_paths, no_time_steps))

    ln_x_t_paths[:, 0] = np.log(f0)
    v_t_paths[:, 0] = v0

    map_out_put = {}

    for i in range(1, no_time_steps):
        u_variance = rnd_generator.uniform(0.0, 1.0, no_paths)
        z_f = rnd_generator.normal(0.0, 1.0, no_paths, type_random_numbers)

        np.copyto(v_t_paths[:, i],
                  VarianceMC.get_variance(k, theta, epsilon, 1.5, t_i[i - 1], t_i[i], v_t_paths[:, i - 1],
                                          u_variance, no_paths))
        np.copyto(int_v_t_paths[:, i - 1], HestonTools.get_integral_variance(t_i[i - 1], t_i[i], v_t_paths[:, i - 1],
                                                                             v_t_paths[:, i], 0.5, 0.5))

        HestonTools.get_delta_weight(t_i[i - 1], t_i[i], v_t_paths[:, i - 1], v_t_paths[:, i], z_f, delta_weight)
        HestonTools.get_var_weight(t_i[i - 1], t_i[i], v_t_paths[:, i - 1], v_t_paths[:, i], z_f, var_weight)

        inv_variance += HestonTools.get_integral_variance(t_i[i - 1], t_i[i], 1.0 / v_t_paths[:, i - 1],
                                                          1.0 / v_t_paths[:, i], 0.5, 0.5)

        k0 = - delta_t_i[i - 1] * (rho * k * theta) / epsilon
        k1 = 0.5 * delta_t_i[i - 1] * ((k * rho) / epsilon - 0.5) - rho / epsilon
        k2 = 0.5 * delta_t_i[i - 1] * ((k * rho) / epsilon - 0.5) + rho / epsilon
        k3 = 0.5 * delta_t_i[i - 1] * (1.0 - rho * rho)

        np.copyto(ln_x_t_paths[:, i], ln_x_t_paths[:, i - 1] + k0 + k1 * v_t_paths[:, i - 1] + k2 * v_t_paths[:, i] +
                  np.sqrt(k3) * AnalyticTools.dot_wise(np.sqrt(v_t_paths[:, i - 1] + v_t_paths[:, i]), z_f))

    map_out_put[HESTON_OUTPUT.PATHS] = np.exp(ln_x_t_paths)
    map_out_put[HESTON_OUTPUT.INTEGRAL_VARIANCE_PATHS] = int_v_t_paths
    map_out_put[HESTON_OUTPUT.DELTA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL] = np.multiply(delta_weight, 1.0 / (np.sqrt(1.0 - rho * rho) * t1 * f0))
    map_out_put[HESTON_OUTPUT.SPOT_VARIANCE_PATHS] = v_t_paths
    map_out_put[HESTON_OUTPUT.TIMES] = t_i

    HestonTools.get_gamma_weight(delta_weight, var_weight, inv_variance, rho, t1, gamma_weight)

    map_out_put[HESTON_OUTPUT.GAMMA_MALLIAVIN_WEIGHTS_PATHS_TERMINAL] = np.multiply(gamma_weight, 1.0 / ((1.0 - rho * rho) * np.power(t1 * f0, 2.0)))

    return map_out_put
