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

from Tools.Types import min_value, ndarray
import numpy as np


def derive_cev_drift(mu: float, t: float, x: ndarray):
    return mu * x


def derive_cev_sigma(sigma: float, rho: float, t: float, x: ndarray):
    x_epsilon = np.maximum(min_value, x)
    return sigma * rho * np.power(x_epsilon, rho - 1.0)


def cev_drift(mu: float, t: float, x: ndarray):
    return mu * x


def cev_sigma(sigma: float, rho: float, t: float, x: ndarray):
    return sigma * np.power(x, rho)


def z_drift(mu: float, rho: float, sigma: float, t: float, x: ndarray):
    x_epsilon = np.maximum(min_value, x)
    return (1.0 - rho) * np.add(mu * x_epsilon, np.divide(-0.5 * rho * sigma * sigma, x_epsilon))


def z_sigma(sigma: float, rho: float, t: float, x: ndarray):
    return (1.0 - rho) * sigma * np.ones(len(x))


def bs_drift_flat(t: float, x: ndarray, rate_t: float, dividend_t: float,):
    return (rate_t - dividend_t) * x


def bs_sigma_flat(t: float, x: ndarray, sigma_t: float):
    return sigma_t * x



