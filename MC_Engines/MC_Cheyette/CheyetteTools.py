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

import QuantLib as ql
import numpy as np
from scipy import integrate
from Tools.Types import ndarray


def beta(ta, tb, k, m=1):  # integral exp(-\int_{t_a}^{t_b} k_s ds)
    delta = (tb - ta)
    if m * delta * k < 1e-08:
        return 1.0 - delta * k
    else:
        return np.exp(- m * delta * k)


def gamma(ta, tb, k):  # integral exp(-\int_{t_a}^{t_b} exp(-\int_{u}^{t_b} k_s ds) ds
    return (np.exp(-k * ta) - np.exp(-k * tb)) / k


def variance(ta, tb, k, eta_vol, x: ndarray, y: ndarray) -> ndarray:
    delta = (tb - ta)
    m = np.exp(-2.0 * delta * k)
    return m * delta * np.power(eta_vol(ta, x, y), 2.0)


def get_drift_forward_measure(ta, tb, k, eta_vol, x: ndarray, y: ndarray) -> ndarray:
    m = (np.exp(- k * ta) - np.exp(- k * tb)) / k
    return m * np.power(eta_vol(ta, x, y), 2.0)


def get_zero_coupon(t_start: float, t_end: float, k: float, ft: ql.ForwardCurve, x: ndarray,
                    y: ndarray) -> ndarray:

    m = ft.discount(t_end) / ft.discount(t_start)
    g = gamma(t_start, t_end, k)
    return m * np.exp(- g * x - 0.5 * g * g * y)


def y_moment_linear_eta_vol(a: float, b: float, k: float, t: float):
    e_2_t = b * b * gamma(0.0, t, 2.0 * k)
    e_4_t = ((np.power(a * b, 2.0) * 0.5 / k) * (t - gamma(0.0, t, 2.0 * k)) + (a * np.power(b, 3.0) / k)
             * (t - gamma(0.0, t, 2.0 * k)))
    return e_2_t + e_4_t


def x_moment_linear_eta_vol(a: float, b: float, k: float, t: float):
    gt = gamma(0.0, t, k)
    gt2t = gamma(t, 2.0 * t, k)
    m4 = (t - 1.5 * gt + 0.5 * gt2t)
    e_2_t = 0.5 * (b * b / k) * (gt - gt2t)
    e_4_t = 0.5 * np.power(b * a / k, 2.0) * m4 + (a * np.power(b, 3.0) / (k * k)) * m4

    return e_2_t + e_4_t

def linear_lv_gamma_future(s: float, k: float, a: float, b: float):
    return 0.5 * a * a * b * gamma(0.0, s, k)

def linear_lv_gamma_arrears(s: float, k: float, a: float, b: float, tp=0):
    return 0.5 * a * a * b * gamma(0.0, s, k)  - (a*b*b/k) * (gamma(0.0, s, k) - 0.5 * gamma(tp-s, tp+s, k))

def linear_lv_gamma_square_arrears(s: float, k: float, a: float, b: float, tp=0):
    return  np.power(a * b, 2.0) * gamma(0.0, s, 2.0 * k) + np.power(a * b, 2.0) * gamma(0.0, s, k)  - 2.0 * (np.power(b, 3.0) * a / k) * (gamma(0.0, s, k) - 0.5 * gamma(tp-s, tp+s, k))

def ca_linear_lv_future_fras(t0: float, ta: float, tb: float, k: float, a: float, b: float, ft: ql.ForwardCurve):
    delta = (tb - ta)
    df_ta = ft.discount(ta)
    df_tb = ft.discount(tb)
    m = (gamma(0, (tb - t0), k) - gamma(0, (ta- t0), k)) *  df_ta / (df_tb * delta)

    f_convexity = lambda t: np.exp(-k*(t0 - t)) * (b + linear_lv_gamma_future(t, k, a, b)) * gamma(0.0, tb - t, k)

    integral_value = integrate.quad(f_convexity, 0.0, t0)

    return m * b * integral_value[0]


def ca_linear_lv_arrears_fras(ta: float, tb: float, k: float, a: float, b: float, ft: ql.ForwardCurve):

    f_convexity = lambda t: (b * b + linear_lv_gamma_square_arrears(t, k, a, b, ta)) * (gamma(0.0, tb - t, k) - gamma(0.0, ta - t, k))**2
    integral_value = integrate.quad(f_convexity, 0.0, ta)


    return integral_value[0]
