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
import QuantLib as ql

from Tools.Types import CHEYETTE_OUTPUT
from MC_Engines.MC_Cheyette import CheyetteTools


def get_path_multi_step(tis, x0, y0, ft: ql.ForwardCurve, k, no_paths, eta_vol, rnd_generator) -> map:
    no_time_steps = len(tis)

    x_paths = np.empty((no_paths, no_time_steps ))
    y_paths = np.empty((no_paths, no_time_steps))
    rate_paths = np.empty((no_paths, no_time_steps))
    bt_paths = np.empty((no_paths, no_time_steps))

    # output
    map_output = {}

    # Start of the path
    x_paths[:, 0] = x0
    y_paths[:, 0] = y0
    rate_paths[:, 0] = ((ft.discount(tis[0]) / ft.discount(tis[1])) - 1.0) / (tis[1] - tis[0])
    bt_paths[:, 0] = 1.0
    int_t_delta = np.zeros(no_paths)
    int_t = np.zeros(no_paths)

    # run simulation
    for j in range(1, no_time_steps):
        delta_t = tis[j] - tis[j - 1]
        zs = rnd_generator.normal(mu=0.0, sigma=1.0, size=(no_paths, 2))
        beta_t = CheyetteTools.beta(tis[j - 1], tis[j], k)
        gamma_t = CheyetteTools.gamma(0.0, delta_t, k)

        eta_i = eta_vol(tis[j - 1], x_paths[:, j - 1], y_paths[:, j - 1])
        v_t = CheyetteTools.variance(tis[j - 1], tis[j], k, eta_i)
        y_paths[:, j] = beta_t * beta_t * y_paths[:, j - 1] + v_t
        x_paths[:, j] = beta_t * x_paths[:, j - 1] + gamma_t * y_paths[:, j - 1] + np.multiply(np.sqrt(v_t), zs[:, 0])

        # Cholesky
        std_b_t = np.sqrt(CheyetteTools.variance_bank_account(delta_t, k)) * eta_i
        cov = CheyetteTools.covariance_bank_account_rate(delta_t, k) * np.multiply(eta_i, eta_i)
        rho = cov / (np.sqrt(v_t) * std_b_t)

        # bank account sampling
        df_i_1 = ft.discount(tis[j - 1])
        df_i = ft.discount(tis[j])
        f0 = (df_i_1 / df_i - 1.0) / (tis[j] - tis[j - 1])
        int_t_0 = np.log(df_i_1 / df_i)

        int_t_delta = (int_t + CheyetteTools.gamma(0.0, delta_t, k) * x_paths[:, j-1] + CheyetteTools.int_lambda(delta_t, k) * y_paths[:, j-1] +
                       std_b_t * (np.multiply(rho, zs[:, 0]) + np.multiply(np.sqrt(1.0 - rho * rho), zs[:, 1]))) + int_t_0

        # forward rate
        rate_paths[:, j] = x_paths[:, j] + f0
        bt_paths[:, j] = np.exp(int_t_delta)
        int_t = int_t_delta

    map_output[CHEYETTE_OUTPUT.PATHS_X] = x_paths
    map_output[CHEYETTE_OUTPUT.PATHS_Y] = y_paths
    map_output[CHEYETTE_OUTPUT.RATE] = rate_paths
    map_output[CHEYETTE_OUTPUT.BANK_ACCOUNT] = bt_paths

    return map_output


def get_path_multi_step_forward_measure(tis, x0, y0, ft: ql.ForwardCurve, k, no_paths, eta_vol, rnd_generator) -> map:
    no_time_steps = len(tis)

    x_paths = np.empty((no_paths, no_time_steps))
    y_paths = np.empty((no_paths, no_time_steps))
    rate_paths = np.empty((no_paths, no_time_steps))

    # output
    map_output = {}

    # Start of the path
    x_paths[:, 0] = x0
    y_paths[:, 0] = y0
    rate_paths[:, 0] = ((ft.discount(tis[0]) / ft.discount(tis[1])) - 1.0) / (tis[1] - tis[0])

    # run simulation
    for j in range(1, no_time_steps):
        delta_t = tis[j] - tis[j - 1]
        zs = rnd_generator.normal(mu=0.0, sigma=1.0, size=no_paths)
        beta_t = CheyetteTools.beta(tis[j - 1], tis[j], k)
        gamma_t = CheyetteTools.gamma(0.0, delta_t, k)
        eta_t_j = eta_vol(tis[j-1], x_paths[:, j - 1], y_paths[:, j - 1])
        # wy_t = (delta_t - gamma_t) / k

        v_t = CheyetteTools.variance(tis[j - 1], tis[j], k, eta_t_j)

        # paths
        y_paths[:, j] = beta_t * beta_t * y_paths[:, j - 1] + v_t
        x_paths[:, j] = (beta_t * x_paths[:, j - 1] + gamma_t * y_paths[:, j - 1]
                         - CheyetteTools.get_drift_forward_measure(tis[j-1], tis[j], tis[-1], k, eta_t_j)
                         + np.multiply(np.sqrt(v_t), zs))

        # forward rate
        df_i_1 = ft.discount(tis[j - 1])
        df_i = ft.discount(tis[j])
        f0 = (df_i_1 / df_i - 1.0) / (tis[j] - tis[j - 1])
        rate_paths[:, j] = x_paths[:, j] + f0

    map_output[CHEYETTE_OUTPUT.PATHS_X] = x_paths
    map_output[CHEYETTE_OUTPUT.PATHS_Y] = y_paths
    map_output[CHEYETTE_OUTPUT.RATE] = rate_paths

    return map_output
