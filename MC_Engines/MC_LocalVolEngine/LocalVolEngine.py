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
                        rnd_generator) -> map:

    no_paths = 2 * no_paths if type_random_number == Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

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

    map_output[Types.LOCAL_VOL_OUTPUT.PATHS] = np.exp(x_t)
    map_output[Types.LOCAL_VOL_OUTPUT.SPOT_VARIANCE_PATHS] = v_t
    map_output[Types.LOCAL_VOL_OUTPUT.INTEGRAL_VARIANCE_PATHS] = int_v_t

    return map_output

