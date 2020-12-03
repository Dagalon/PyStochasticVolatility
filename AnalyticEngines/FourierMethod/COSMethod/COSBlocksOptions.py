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


@nb.jit("f8[:](f8,f8,f8,f8,i8)", nopython=True, nogil=True, parallel=True)
def digital_block(a: float, b: float, c: float, d: float, no_terms: int):
    alpha_d = (d - a) / (b - a)
    alpha_c = (c - a) / (b - a)
    output = np.zeros(no_terms)
    for i in range(0, no_terms):
        if i == 0:
            output[i] = d - c
        else:
            output[i] = (np.sin(i * np.pi * alpha_d) - np.sin(i * np.pi * alpha_c)) * (b - a) / (i * np.pi)

    return output


@nb.jit("f8[:](f8,f8,f8,f8,i8)", nopython=True, nogil=True, parallel=True)
def call_put_block(a: float, b: float, c: float, d: float, no_terms: int):
    alpha_d = (d - a) / (b - a)
    alpha_c = (c - a) / (b - a)
    exp_d = np.exp(d)
    exp_c = np.exp(c)
    output = np.zeros(no_terms)
    for i in range(0, no_terms):
        beta = np.pi * i / (b - a)
        a_k_1 = np.cos(i * np.pi * alpha_d) * exp_d - np.cos(i * np.pi * alpha_c) * exp_c
        a_k_2 = beta * (np.sin(i * np.pi * alpha_d) * exp_d - np.sin(i * np.pi * alpha_c) * exp_c)
        multiplier = 1.0 / (1 + beta * beta)
        output[i] = multiplier * (a_k_1 + a_k_2)

    return output
