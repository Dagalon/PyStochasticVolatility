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
from Tools import Types
from Tools import AnalyticTools


def get_path_multi_step(t0: float,
                        t1: float,
                        parameters: Types.ndarray,
                        f0: float,
                        v0: float,
                        no_paths: int,
                        no_time_steps: int,
                        type_random_number: Types.TYPE_STANDARD_NORMAL_SAMPLING,
                        rnd_generator,
                        **kwargs) -> map:
    nu_1 = parameters[0]
    nu_2 = parameters[1]
    theta = parameters[2]
    rho = parameters[3]

    rho_inv = np.sqrt(1.0 - rho * rho)

    no_paths = 2 * no_paths if type_random_number == Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

    t_i = get_time_steps(t0, t1, no_time_steps, **kwargs)
    no_time_steps = len(t_i)

    delta_t_i = np.diff(t_i)

    x_t = np.empty((no_paths, no_time_steps))
    v_t = np.empty((no_paths, no_time_steps))
    int_variance_t_i = np.empty((no_paths, no_time_steps - 1))

    x_t[:, 0] = np.log(f0)
    v_t[:, 0] = v0

    v_t_1_i_1 = np.empty(no_paths)
    v_t_2_i_1 = np.empty(no_paths)

    v_t_1_i_1[:] = v0
    v_t_2_i_1[:] = v0

    map_output = {}

    for i_step in range(1, no_time_steps):
        z_i = rnd_generator.normal(0.0, 1.0, no_paths, type_random_number)
        z_sigma = rnd_generator.normal(0.0, 1.0, no_paths, type_random_number)
        v_t[:, i_step] = get_variance_sampling(t_i[i_step - 1], t_i[i_step], v_t_1_i_1, v_t_2_i_1,
                                               theta, nu_1, nu_2, z_sigma)

        int_variance_t_i[:, i_step - 1] = 0.5 * (v_t[:, i_step - 1] * delta_t_i[i_step - 1] +
                                                 v_t[:, i_step - 1] * delta_t_i[i_step - 1])

        x_t[:, i_step] = x_t[:, i_step - 1] - 0.5 * int_variance_t_i[:, i_step - 1] + \
                         AnalyticTools.dot_wise(np.sqrt(v_t[:, i_step - 1]),
                                                (rho * z_sigma + rho_inv * z_i) * np.sqrt(delta_t_i[i_step - 1]))

    map_output[Types.MIXEDLOGNORMAL_OUTPUT.SPOT_VARIANCE_PATHS] = v_t
    map_output[Types.MIXEDLOGNORMAL_OUTPUT.TIMES] = t_i
    map_output[Types.MIXEDLOGNORMAL_OUTPUT.INTEGRAL_VARIANCE_PATHS] = int_variance_t_i
    map_output[Types.MIXEDLOGNORMAL_OUTPUT.PATHS] = np.exp(x_t)

    return map_output


def get_time_steps(t0: float, t1: float, no_time_steps: int, **kwargs):
    if len(kwargs) > 0:
        extra_points = kwargs['extra_sampling_points']
        basis_sampling_dates = np.linspace(t0, t1, no_time_steps).tolist()
        full_points = np.array(list(set(extra_points + basis_sampling_dates)))
        return sorted(full_points)
    else:
        return np.linspace(t0, t1, no_time_steps)


def get_variance_sampling(t0: float,
                          t1: float,
                          v_1_i_1: Types.ndarray,
                          v_2_i_1: Types.ndarray,
                          theta: float,
                          nu_1: float,
                          nu_2: float,
                          z: Types.ndarray):
    delta_time = (t1 - t0)
    sqrt_delta_time = np.sqrt(delta_time)
    drift_1 = v_1_i_1 * np.exp(- 0.5 * nu_1 * nu_1 * delta_time)
    drift_2 = v_2_i_1 * np.exp(- 0.5 * nu_2 * nu_2 * delta_time)
    noise_1 = np.exp(nu_1 * sqrt_delta_time * z)
    noise_2 = np.exp(nu_2 * sqrt_delta_time * z)
    np.copyto(v_1_i_1, drift_1 * noise_1)
    np.copyto(v_2_i_1, drift_2 * noise_2)
    return (1.0 - theta) * v_1_i_1 + theta * v_2_i_1
