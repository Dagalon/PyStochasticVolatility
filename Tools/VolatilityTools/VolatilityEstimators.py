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
#

import numpy as np
import numba as nb

from Tools import Types, AnalyticTools


def get_mean_sigma(t: float, b0: float, b1: float, b2: float):
    tau_t_var = 0.5 * (np.exp(2.0 * b2 * t) - 1.0) / b2
    return np.exp(2.0 * b0 + 2.0 * b1 * b1 * tau_t_var)


@nb.jit("(f8[:,:],f8[:], i8)", nopython=True, nogil=True)
def get_integrated_variance_from_sim(v_paths_paths: Types.ndarray, t_i_s: Types.ndarray, no_paths: int):
    variance = np.zeros(no_paths)
    no_time_steps = len(t_i_s)
    for i in range(0, no_paths):
        for j in range(1, no_time_steps):
            delta_time = t_i_s[j] - t_i_s[j - 1]
            variance[i] += delta_time * v_paths_paths[i, j - 1]

    return variance


def get_integrated_variance_estimator(paths: Types.ndarray, no_paths: int, freq_sampling: int, t_i: Types.ndarray,
                                      method: Types.ESTIMATOR_TYPE):
    if method == Types.ESTIMATOR_TYPE.INTEGRATED_VARIANCE_FOURIER:
        return get_integrated_variance_fourier(paths, t_i, freq_sampling, no_paths)

    elif method == Types.ESTIMATOR_TYPE.INTEGRATED_VARIANCE_EMPIRICAL:
        return get_integrated_variance_empirical(paths, t_i, freq_sampling, no_paths)

    elif method == Types.ESTIMATOR_TYPE.INTEGRATED_VARIANCE_EMPIRICAL:
        pass
    else:
        print("The estimator %s" % (str(method)))


@nb.jit("f8[:](f8[:,:],f8[:],i8,i8)", nopython=True, nogil=True, parallel=True)
def get_integrated_variance_fourier(path: Types.ndarray, t_k: Types.ndarray, freq_sampling: int, no_paths: int):
    no_time_steps = len(t_k)
    sigma_n = np.zeros(no_paths)
    n_kernel = int(len(t_k) * 0.5)
    for k in range(0, no_paths):
        for i in range(0, no_time_steps - freq_sampling, freq_sampling):
            delta_x_i = path[k, i + freq_sampling] - path[k, i]
            for j in range(0, no_time_steps - freq_sampling, freq_sampling):
                diff = (t_k[i] - t_k[j]) * (2.0 * np.pi / t_k[-1])
                delta_x_j = path[k, j + freq_sampling] - path[k, j]
                sigma_n[k] += AnalyticTools.dirichlet_kernel(diff, n_kernel) * delta_x_i * delta_x_j

    return sigma_n


@nb.jit("f8[:](f8[:,:],f8[:],i8,i8)", nopython=True, nogil=True)
def get_integrated_variance_empirical(path: Types.ndarray, t_k: Types.ndarray, freq_sampling: int, no_paths: int):
    no_time_steps = len(t_k)
    sigma_n = np.zeros(no_paths)

    for k in range(0, no_paths):
        for j in range(freq_sampling, no_time_steps, freq_sampling):
            diff = path[k, j] - path[k, j - freq_sampling]
            sigma_n[k] += (diff * diff)

    return sigma_n


@nb.jit("f8[:](f8[:,:],f8[:], i8, f8)", nopython=True, nogil=True, parallel=True)
def get_spot_variance_fourier(path: Types.ndarray, t_k: Types.ndarray, no_paths: int, t: float):
    no_time_steps = len(t_k)
    # n_kernel = int(0.5 * no_time_steps)
    n_kernel = int(0.85 * np.power(no_time_steps, 0.75))
    b = 0.125
    # m_kernel = int(b * (0.5 / np.pi) * np.sqrt(no_time_steps) * np.log(no_time_steps)) + 4.2
    m_kernel = np.maximum(int((1.0 / (2.0 * np.pi)) * 0.125 * np.sqrt(np.power(no_time_steps, 0.75)) * np.log(np.power(no_time_steps, 0.75))), 1.0)
    sigma_n = np.zeros(no_paths)
    # coefficients = np.zeros(shape=(no_paths, 2 * m_kernel + 1))

    # for m in range(0, 2 * m_kernel + 1):
    #     coefficients[:, m] = get_fourier_coefficient(path, t_k, no_paths, m - m_kernel)
    #
    # spot_variance_estimation = np.zeros(no_paths)
    # t_new = (2.0 * np.pi / t_k[-1]) * t
    # for k in range(0, no_paths):
    #     aux_var = 0.0
    #     for m in range(0, 2 * m_kernel + 1):
    #         aux_var += (1.0 - np.abs(m - m_kernel) / m_kernel) * np.exp(1j * (m - m_kernel) * t_new) * coefficients[k, m]
    #
    #     spot_variance_estimation[k] = aux_var.real
    #
    # return spot_variance_estimation

    for k in range(0, no_paths):
        for i in range(0, no_time_steps - 1):
            delta_x_i = path[k, i + 1] - path[k, i]
            for j in range(0, no_time_steps - 1):
                diff = (t_k[j] - t_k[i]) * (2.0 * np.pi / t_k[-1])
                diff_t = (t - t_k[j]) * (2.0 * np.pi / t_k[-1])
                delta_x_j = path[k, j + 1] - path[k, j]
                dirichlet_kernel = AnalyticTools.dirichlet_kernel(diff, n_kernel)
                fejer_kernel = AnalyticTools.fejer_kernel(diff_t, m_kernel)
                sigma_n[k] += (dirichlet_kernel * fejer_kernel * delta_x_i * delta_x_j)

    # return 0.5 * sigma_n / np.pi
    return sigma_n


@nb.jit("f8[:](f8[:,:],f8[:], i8, i8)", nopython=True, nogil=True)
def get_fourier_coefficient(path: Types.ndarray, t_k: Types.ndarray, no_paths: int, s: int):
    no_time_steps = len(t_k)
    n_kernel = int(0.5 * no_time_steps)
    coefficients = np.empty(no_paths)

    for k in range(0, no_paths):
        aux_value = 0
        for i in range(0, no_time_steps - 1):
            delta_x_i = path[k, i + 1] - path[k, i]
            for j in range(0, no_time_steps - 1):
                diff = (t_k[j] - t_k[i]) * (2.0 * np.pi / t_k[-1])
                delta_x_j = path[k, j + 1] - path[k, j]
                dirichlet_kernel = AnalyticTools.dirichlet_kernel(diff, n_kernel)
                aux_value += delta_x_i * delta_x_j * dirichlet_kernel * np.exp(-1j * (2.0 * np.pi / t_k[-1]) * t_k[j] * s)

        coefficients[k] = aux_value.real

    return coefficients
