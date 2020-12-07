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

from Tools.Types import Vector, ndarray
from MC_Engines.MC_Bergomi2F import Bergomi2fTools
from Tools import Types


def get_path_multi_step(t0: float,
                        t1: float,
                        parameters: ndarray,
                        f0: float,
                        v0: float,
                        no_paths: int,
                        no_time_steps: int,
                        type_random_numbers: Types.TYPE_STANDARD_NORMAL_SAMPLING,
                        rnd_generator) -> ndarray:
    theta = parameters[0]
    nu_x = parameters[1]
    nu_y = parameters[2]
    rho_xy = parameters[3]
    rho_xf = parameters[4]
    rho_yf = parameters[5]

    cov = np.zeros(shape=(3, 3))
    cov[0, 0] = 1.0
    cov[0, 1] = rho_xy
    cov[0, 2] = rho_xf
    cov[1, 0] = rho_xy
    cov[1, 1] = 1.0
    cov[1, 2] = rho_yf
    cov[2, 0] = rho_xf
    cov[2, 1] = rho_yf
    cov[2, 2] = 1.0

    cholesky_cov = np.linalg.cholesky(cov)

    no_paths = 2 * no_paths if type_random_numbers == Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

    t_k = np.linspace(t0, t1, no_time_steps)

    ln_x_t_paths = np.zeros(shape=(no_paths, no_time_steps))
    ln_v_t_paths = np.zeros(shape=(no_paths, no_time_steps))
    int_v_t_paths = np.zeros(shape=(no_paths, no_time_steps - 1))

    ln_x_t_paths[:, 0] = np.log(f0)
    ln_v_t_paths[:, 0] = np.log(v0)

    map_out_put = {}
    alpha_theta = np.sqrt(nu_x * nu_x * (1.0 - theta) * (1.0 - theta) + nu_y * nu_y * theta * theta + \
                          2.0 * (1.0 - theta) * theta * nu_x * nu_y * rho_xy)

    w_t_i_1 = np.zeros(shape=(3, no_paths))
    for j in range(1, no_time_steps):
        w_t_i = np.sqrt(t_k[j] - t_k[j - 1]) * np.matmul(cholesky_cov, rnd_generator.normal(0.0, 1.0, (3, no_paths),
                                                                                            type_random_numbers))

        Bergomi2fTools.get_log_spot(alpha_theta, nu_x, nu_y, theta, t_k[j - 1], t_k[j], no_paths,
                                    w_t_i_1, w_t_i, ln_x_t_paths[:, j - 1], ln_x_t_paths[:, j],
                                      ln_v_t_paths[:, j - 1], ln_v_t_paths[:, j], int_v_t_paths[:, j - 1])

    map_out_put[Types.BERGOMI2F_OUTPUT.PATHS] = np.exp(ln_x_t_paths)
    map_out_put[Types.BERGOMI2F_OUTPUT.SPOT_VARIANCE_PATHS] = np.exp(ln_v_t_paths)
    map_out_put[Types.BERGOMI2F_OUTPUT.INTEGRAL_VARIANCE_PATHS] = int_v_t_paths
    map_out_put[Types.BERGOMI2F_OUTPUT.TIMES] = t_k

    return map_out_put
