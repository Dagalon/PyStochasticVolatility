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

from Tools.AnalyticTools import dot_wise


def get_variance(alpha,
                 nu,
                 alpha_t,
                 t,
                 no_deep,
                 rnd_generator):
    no_paths = len(alpha_t)
    no_steps = int(2 ** no_deep)
    t_i = np.linspace(0.0, t, no_steps)
    v_t = np.zeros(no_paths)
    w_T = (np.log(alpha_t / alpha) + 0.5 * nu * nu * t) / nu

    alpha_i_1 = alpha * np.ones(no_paths)
    alpha_i = np.empty(no_paths)

    m_w_t = get_full_path_brownian_bridge(0.0, t, no_deep, np.zeros(no_paths), w_T, rnd_generator)

    for k in range(1, no_steps):
        noise = np.subtract(m_w_t[:, k], m_w_t[:, k-1])
        np.copyto(alpha_i, dot_wise(alpha_i_1, np.exp(-0.5 * (t_i[k] - t_i[k - 1]) * nu * nu + nu * noise)))
        np.copyto(v_t,
                  np.add(v_t, (t_i[k] - t_i[k - 1]) * 0.5 * np.add(np.power(alpha_i_1, 2.0), np.power(alpha_i, 2.0))))
        np.copyto(alpha_i_1, alpha_i)

    np.copyto(v_t, np.add(v_t, (t_i[-1] - t_i[-2]) * 0.5 * (np.power(alpha_i_1, 2.0) + np.power(alpha_t, 2.0))))
    return v_t


@nb.jit("f8[:](f8,f8,f8,f8[:],f8[:],f8[:])", nopython=True, nogil=True)
def get_brownian_bridge(u, s, t, w_u, w_t, z):
    d_u_s = s - u
    d_s_t = t - s
    d_u_t = t - u

    mean = (d_s_t * w_u + d_u_s * w_t) / d_u_t
    variance = d_s_t * d_u_s / d_u_t
    return mean + np.sqrt(variance) * z


def get_full_path_brownian_bridge(t0, t1, n, z_t0, z_t1, rnd_generator):
    no_paths = len(z_t1)
    no_steps = int(2 ** n)
    delta_step = (t1 - t0) / no_steps

    m_paths = np.empty(shape=(no_paths, no_steps + 1))
    w_t_i = np.empty(no_paths)

    m_paths[:, 0] = z_t0
    m_paths[:, no_steps] = z_t1

    for n_i in range(1, n + 1):
        scale_factor = 2 ** (n - n_i)
        even_nodes = [scale_factor * k for k in range(0, int(2 ** n_i) + 1) if k % 2 == 0]
        odd_nodes = [scale_factor * k for k in range(0, int(2 ** n_i) + 1) if k % 2 == 1]

        for n_j in range(0, len(odd_nodes)):
            np.copyto(w_t_i, get_brownian_bridge(even_nodes[n_j] * delta_step,
                                                 odd_nodes[n_j] * delta_step,
                                                 even_nodes[n_j + 1] * delta_step,
                                                 m_paths[:, even_nodes[n_j]],
                                                 m_paths[:, even_nodes[n_j + 1]],
                                                 rnd_generator.normal(0.0, 1.0, no_paths)))

            np.copyto(m_paths[:, odd_nodes[n_j]], w_t_i)

    return m_paths
