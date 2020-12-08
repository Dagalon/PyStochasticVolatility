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


@nb.jit("f8(f8,f8,f8, f8)", nopython=True, nogil=True)
def get_variance_swap(v0: float, k: float, theta: float, t: float):
    exp_t = np.exp(- k * t)
    first_term = v0 * (1.0 - exp_t) / k
    second_term = theta * (t - (1.0 - exp_t) / k)
    return np.sqrt((first_term + second_term) / t)


@nb.jit("f8(f8,f8,f8,f8,f8)", nopython=True, nogil=True)
def get_rho_term_var_swap(v0: float, k: float, theta: float, epsilon: float, t: float):
    exp_t = np.exp(- k * t)
    first_term = ((epsilon * v0) / k) * ((1.0 - exp_t) / k - exp_t * t)
    second_term = ((epsilon * theta) / k) * (t - 2.0 * (1.0 - exp_t) / k + t * exp_t)
    return first_term + second_term
