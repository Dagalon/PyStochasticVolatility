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
from typing import List

Vector = List[float]


def iv_atm_second_order(T: float, parameters: Vector) -> float:
    """

    :param T: maturity.
    :param parameters: parameters[0] is alpha, parameters[1] is nu and parameters[2] is rho.
    :return: implied volatility expansion with our method.
    """
    a1 = parameters[0]
    a2 = 0.25 * np.power(parameters[0], 2.0) * parameters[1] * parameters[2]
    a3 = - 0.125 * np.power(parameters[0], 3.0) * np.power(parameters[1], 2.0)
    return a1 + a2 * T + a3 * np.power(T, 2.0)


def iv_atm_ln_hagan(T: float, parameters: Vector) -> float:
    """

    :param T: maturity.
    :param parameters: parameters[0] is alpha, parameters[1] is nu and parameters[2] is rho.
    :return: orignal Hagan's implied volatility expansion.
    """
    a0 = parameters[0]
    a1 = 0.25 * np.power(parameters[0], 2.0) * parameters[1] * parameters[2] + \
         parameters[0] * np.power(parameters[1], 2.0) * (2.0 - 3.0 * np.power(parameters[2], 2.0)) / 24.0

    return a0 + a1 * T


def iv_atm_variance_swap(T: float, parameters: Vector) -> float:
    """

       :param T: maturity.
       :param parameters: parameters[0] is alpha, parameters[1] is nu and parameters[2] is rho.
       :return: implied using elisa&kenichiro's expansion
    """
    alpha = parameters[0]
    nu = 0.5 * parameters[1]
    rho = parameters[2]

    a0 = alpha - 0.25 * np.power(rho, 2.0) * np.power(alpha, 3.0)
    a1 = alpha * rho * nu * ((- 5.0 / 32.0) * rho * nu - 0.25 * rho * nu * np.power(alpha, 2.0) + 0.125 * alpha)

    return a0 + a1 * T
