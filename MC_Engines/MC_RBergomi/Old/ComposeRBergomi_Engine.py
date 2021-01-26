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

from MC_Engines.MC_RBergomi import ToolsVariance
from Tools.Types import Vector, ndarray, TYPE_STANDARD_NORMAL_SAMPLING, RBERGOMI_OUTPUT


def get_path_multi_step(t0: float,
                        t1: float,
                        parameters: Vector,
                        f0: float,
                        sigma_0: float,
                        no_paths: int,
                        no_time_steps: int,
                        type_random_number: TYPE_STANDARD_NORMAL_SAMPLING,
                        rnd_generator,
                        **kwargs) -> map:
    nu = parameters[0]
    rho = parameters[1]
    h_short = parameters[2]
    h_long = parameters[3]

    no_paths = 2 * no_paths if type_random_number == TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

    t_i_s = np.array(get_time_steps(t0, t1, no_time_steps, **kwargs))
    no_time_steps = len(t_i_s)

    s_t = np.empty((no_paths, no_time_steps))
    s_t[:, 0] = f0

    z_i_s = rnd_generator.normal(mu=0.0, sigma=1.0, size=(2 * (no_time_steps - 1), no_paths),
                                 sampling_type=type_random_number)
    map_out_put = {}
    outputs = ToolsVariance.generate_paths_compose_rbergomi(f0,
                                                            sigma_0,
                                                            nu,
                                                            h_short,
                                                            h_long,
                                                            z_i_s,
                                                            np.linalg.cholesky(
                                                                ToolsVariance.get_covariance_matrix(t_i_s[1:], h_short,
                                                                                                    rho)),
                                                            np.linalg.cholesky(
                                                                ToolsVariance.get_covariance_matrix(t_i_s[1:], h_long,
                                                                                                    rho)),
                                                            t_i_s,
                                                            no_paths)

    map_out_put[RBERGOMI_OUTPUT.PATHS] = outputs[0]
    map_out_put[RBERGOMI_OUTPUT.SPOT_VOLATILITY_PATHS] = outputs[1]
    map_out_put[RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS] = outputs[1] * outputs[1]
    map_out_put[RBERGOMI_OUTPUT.INTEGRAL_VARIANCE_PATHS] = outputs[2]
    map_out_put[RBERGOMI_OUTPUT.TIMES] = t_i_s

    return map_out_put


def get_time_steps(t0: float, t1: float, no_time_steps: int, **kwargs):
    if len(kwargs) > 0:
        extra_points = kwargs['extra_sampling_points']
        basis_sampling_dates = np.linspace(t0, t1, no_time_steps).tolist()
        full_points = np.array(list(set(extra_points + basis_sampling_dates)))
        return sorted(full_points)
    else:
        return np.linspace(t0, t1, no_time_steps)
