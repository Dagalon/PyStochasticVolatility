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

from Tools import Types, AnalyticTools
from typing import Callable


def get_path_multi_step(t0: float,
                        t1: float,
                        f0: float,
                        no_paths: int,
                        no_time_steps: int,
                        type_random_number: Types.TYPE_STANDARD_NORMAL_SAMPLING,
                        local_vol: Callable[[float, Types.ndarray], Types.ndarray],
                        rnd_generator,
                        **kwargs) -> map:

    no_paths = 2 * no_paths if type_random_number == Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

    t_i = get_time_steps(t0, t1, no_time_steps, **kwargs)
    no_time_steps = len(t_i)

    t_i = np.linspace(t0, t1, no_time_steps)
    delta_t_i = np.diff(t_i)

    x_t = np.empty((no_paths, no_time_steps))
    int_v_t = np.empty((no_paths, no_time_steps - 1))
    v_t = np.empty((no_paths, no_time_steps))

    x_t[:, 0] = np.log(f0)
    v_t[:, 0] = local_vol(t0, x_t[:, 0])

    sigma_i_1 = np.zeros(no_paths)
    sigma_i = np.zeros(no_paths)
    sigma_t = np.zeros(no_paths)
    x_t_i_mean = np.zeros(no_paths)

    map_output = {}

    for i_step in range(1, no_time_steps):
        z_i = rnd_generator.normal(0.0, 1.0, no_paths, type_random_number)
        np.copyto(sigma_i_1, local_vol(t_i[i_step - 1], x_t[:, i_step - 1]))
        np.copyto(x_t_i_mean, x_t[:, i_step - 1]-0.5 * np.power(sigma_i_1, 2.0))
        np.copyto(sigma_i, local_vol(t_i[i_step], x_t_i_mean))
        np.copyto(sigma_t, 0.5 * (sigma_i_1 + sigma_i))
        v_t[:, i_step] = np.power(sigma_t, 2.0)
        int_v_t[:, i_step - 1] = v_t[:, i_step] * delta_t_i[i_step - 1]
        x_t[:, i_step] = np.add(x_t[:, i_step - 1],
                                - 0.5 * v_t[:, i_step] * delta_t_i[i_step - 1] +
                                np.sqrt(delta_t_i[i_step - 1]) * AnalyticTools.dot_wise(sigma_t, z_i))

    map_output[Types.LOCAL_VOL_OUTPUT.TIMES] = t_i

    map_output[Types.LOCAL_VOL_OUTPUT.PATHS] = np.exp(x_t)
    map_output[Types.LOCAL_VOL_OUTPUT.SPOT_VARIANCE_PATHS] = v_t
    map_output[Types.LOCAL_VOL_OUTPUT.INTEGRAL_VARIANCE_PATHS] = int_v_t

    return map_output


def get_bachelier_path_multi_step(t0: float,
                        t1: float,
                        f0: float,
                        no_paths: int,
                        no_time_steps: int,
                        type_random_number: Types.TYPE_STANDARD_NORMAL_SAMPLING,
                        local_vol: Callable[[float, Types.ndarray], Types.ndarray],
                        rnd_generator,
                        **kwargs) -> map:

    no_paths = 2 * no_paths if type_random_number == Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

    t_i = get_time_steps(t0, t1, no_time_steps, **kwargs)
    no_time_steps = len(t_i)

    t_i = np.linspace(t0, t1, no_time_steps)
    delta_t_i = np.diff(t_i)

    f_t = np.empty((no_paths, no_time_steps))
    int_v_t = np.empty((no_paths, no_time_steps - 1))
    v_t = np.empty((no_paths, no_time_steps))

    f_t[:, 0] = f0
    v_t[:, 0] = local_vol(t0, f_t[:, 0])

    sigma_i_1 = np.zeros(no_paths)
    sigma_i = np.zeros(no_paths)
    sigma_t = np.zeros(no_paths)

    map_output = {}

    for i_step in range(1, no_time_steps):
        z_i = rnd_generator.normal(0.0, 1.0, no_paths, type_random_number)

        np.copyto(sigma_i_1, local_vol(t_i[i_step - 1], f_t[:, i_step - 1]))
        np.copyto(sigma_i, local_vol(t_i[i_step], f_t[:, i_step - 1]))

        np.copyto(sigma_t, 0.5 * (sigma_i_1 + sigma_i))
        v_t[:, i_step] = np.power(sigma_t, 2.0)
        int_v_t[:, i_step - 1] = v_t[:, i_step] * delta_t_i[i_step - 1]

        f_t[:, i_step] = np.add(f_t[:, i_step - 1], np.sqrt(delta_t_i[i_step - 1]) * AnalyticTools.dot_wise(sigma_t, z_i))

    map_output[Types.LOCAL_VOL_OUTPUT.TIMES] = t_i

    map_output[Types.LOCAL_VOL_OUTPUT.PATHS] = f_t
    map_output[Types.LOCAL_VOL_OUTPUT.SPOT_VARIANCE_PATHS] = v_t
    map_output[Types.LOCAL_VOL_OUTPUT.INTEGRAL_VARIANCE_PATHS] = int_v_t

    return map_output


def get_time_steps(t0: float, t1: float, no_time_steps: int, **kwargs):
    if len(kwargs) > 0:
        extra_points = kwargs['extra_sampling_points']
        basis_sampling_dates = np.linspace(t0, t1, no_time_steps).tolist()
        full_points = np.array(list(set(extra_points + basis_sampling_dates)))
        return sorted(full_points)
    else:
        return np.linspace(t0, t1, no_time_steps)

