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
from scipy.optimize import curve_fit


def get_mean_ratio_rs(x_t: Types.ndarray, chunksize: int = 1):
    no_elements = len(x_t)
    index = list(range(0, no_elements, chunksize))
    no_packages = len(index)

    mean_n = np.zeros(no_packages - 1)
    r_n = np.zeros(no_packages - 1)
    s_n = np.zeros(no_packages - 1)
    ratio = np.zeros(no_packages - 1)

    for i in range(1, no_packages):
        mean_n[i - 1] = np.mean(x_t[index[i - 1]:index[i]])
        s_n[i - 1] = np.std(x_t[index[i - 1]:index[i]])
        z_n = np.cumsum(x_t[index[i - 1]:index[i]] - mean_n[i - 1])
        r_n[i - 1] = z_n.max() - z_n.min()
        ratio[i - 1] = r_n[i - 1] / s_n[i - 1]

    return np.mean(ratio)


def get_estimator_rs(x_t: Types.ndarray, lower_chunksize: int = 0, upper_chunksize: int = 1):
    rs = []
    log_no_elements = []

    for i in range(lower_chunksize, upper_chunksize + 1):
        rs.append(get_mean_ratio_rs(x_t, 2 ** i))
        log_no_elements.append(i * np.log(2))

    def func(x, a, b):
        return a + b * x

    popt, pcov = curve_fit(func, log_no_elements, np.log(rs))
    estimated_rs = func(np.array(log_no_elements), *popt)

    return popt[0], popt[1], log_no_elements, np.log(rs), estimated_rs


def get_estimator_pe(x_t: Types.ndarray, size_f: int):
    no_elements = len(x_t)
    f_i = np.linspace(-0.5, 0.5, size_f)
    sr_n = np.zeros(size_f)
    t_i = np.arange(0, no_elements, 1)

    for i in range(0, size_f):
        z_n = x_t - np.mean(x_t)
        out = np.exp(- 2.0 * np.pi * 1j * t_i * f_i[i]) * z_n
        sr_n[i] = np.power(np.sum(np.abs(out)), 2.0) / no_elements

    mid_point = int((size_f + 1) * 0.5)
    max_size_f = int(np.power(mid_point, 4 / 5))
    log_filtered_f_i = np.log(f_i[mid_point + 1: mid_point + 2 + max_size_f])
    log_filtered_sr_n = np.log(sr_n[mid_point + 1: mid_point + 2 + max_size_f])

    def func(x, a, b):
        return a + b * x

    popt, pcov = curve_fit(func, log_filtered_f_i, log_filtered_sr_n)
    estimated_gamma = func(log_filtered_f_i, *popt)
    hurst_parameter = 0.5 * (1-popt[1])

    return popt[0], popt[1], hurst_parameter, log_filtered_f_i, log_filtered_sr_n, estimated_gamma
