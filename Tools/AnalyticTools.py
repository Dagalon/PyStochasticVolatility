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
import numpy as np


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def normal_pdf(mean=0.0, sigma=1.0, x=0.0):
    return np.exp(-0.5 * np.power(x - mean, 2.0) / sigma) / np.sqrt(2 * np.pi * sigma)


@nb.jit("f8(f8, f8, f8)", nopython=True, nogil=True)
def log_normal_pdf(mean=0.0, sigma=1.0, x=0.0):
    return normal_pdf(mean, sigma, (np.log(x) - mean) / sigma) * 1.0 / (sigma * x)


@nb.jit("f8(f8,f8,i4)", nopython=True, nogil=True)
def get_bessel_moments(t, nu, n):
    out = 0.0
    a_k = np.empty(n + 1)
    b_k = np.empty(n + 1)
    n_factorial = 1

    for i in range(0, n + 1):
        a_k[i] = 2.0 * i * nu + 2 * i * i
        n_factorial *= (i+1)

    n_factorial = n_factorial /(n+1)

    for i in range(0, n + 1):
        prod = 1.0
        for j in range(0, n + 1):
            if j != i:
                prod *= (a_k[i] - a_k[j])

        b_k[i] = n_factorial / prod
        out += b_k[i] * np.exp(a_k[i] * t)

    return out


@nb.jit("f8[:](f8[:],f8[:])", nopython=True, nogil=True)
def dot_wise(x, y):
    no_elements = len(x)
    out = np.empty(no_elements)

    for i in range(0, no_elements):
        out[i] = x[i] * y[i]

    return out


@nb.jit("f8(f8[:],f8[:])", nopython=True, nogil=True)
def scalar_product(x, y):
    no_elements = len(x)
    total = 0.0
    for i in range(0, no_elements):
        total += x[i] * y[i]

    return total


@nb.jit("f8[:](f8[:,:],f8[:])", nopython=True, nogil=True)
def apply_lower_tridiagonal_matrix(a, b):
    no_elements = len(b)
    output = np.zeros(no_elements)
    for i in range(0, no_elements):
        for j in range(0, i+1):
            output[i] += a[i, j] * b[j]

    return output


@nb.jit("f8(f8,f8)", nopython=True, nogil=True)
def dirichlet_kernel(t: float, n: float):
    if np.abs(t) > 0.0:
        return np.sin((n + 0.5) * t) / (np.sin(0.5 * t) * (2.0 * n + 1.0))
    else:
        return 1.0


@nb.jit("f8(f8,f8)", nopython=True, nogil=True)
def fejer_kernel(t: float, n: float):
    if np.abs(t) > 0.0:
        return (1.0 / n) * np.power(np.sin(0.5 * n * t) / np.sin(0.5 * t), 2.0)
    else:
        return 1.0




