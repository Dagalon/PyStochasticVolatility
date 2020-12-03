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

from Tools.Types import ndarray, min_value


def a_cev(rho: float, mu: float, sigma: float, z_t: float, t: float, x: float):
    x_epsilon = np.maximum(min_value, x)
    return (1.0 - rho) * (mu + 0.5 * rho * sigma * sigma * np.power(x_epsilon, -2.0))


def f_analytic_cev(rho: float, mu: float, sigma: float, z_t: float, t: ndarray):
    multiplier = (1.0 - rho) * (mu + 0.5 * rho * sigma * sigma * np.power(z_t, -2.0))
    return z_t * np.exp(multiplier * t)
