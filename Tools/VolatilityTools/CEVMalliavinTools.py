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

import numba as nb
from Tools.Types import ndarray
import numpy as np


@nb.jit("f8[:,:](f8, f8[:,:], f8[:,:])", nopython=True, nogil=True)
def transform_cev_malliavin(rho: float,
                            z_t_paths: ndarray,
                            d_z_t: ndarray):
    dim = z_t_paths.shape
    path_d_z_t = np.empty(shape=dim)

    for i in range(0, dim[0]):
        for j in range(0, dim[1]):
            path_d_z_t[i, j] = d_z_t[i, j] * np.power(z_t_paths[i, j], rho / (1.0 - rho)) / (1.0 - rho)

    return path_d_z_t


@nb.jit("f8[:](f8[:], f8[:])", nopython=True, nogil=True)
def get_error(y_t_n, y_t):
    n = len(y_t_n)
    error = np.zeros(n)

    for i in range(0, n):
        error[i] = np.abs(y_t_n[i] - y_t[i])

    return error


@nb.jit("f8(f8[:], f8[:])", nopython=True, nogil=True)
def get_mean_error(y_t_n, y_t):
    n = len(y_t_n)
    error = 0.0

    for i in range(0, n):
        error += np.abs(y_t_n[i] - y_t[i]) / n

    return error


@nb.jit("f8(f8[:], f8[:])", nopython=True, nogil=True)
def get_square_error(y_t_n, y_t):
    n = len(y_t_n)
    error = 0.0

    for i in range(0, n):
        error += np.power(y_t_n[i] - y_t[i], 2.0) / n

    return np.sqrt(error)
