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


def covariance_bank_account_rate(delta: float, k: float):
    return (gamma(0.0, delta, k) - gamma(0.0, delta, 2.0 * k)) / k


def int_lambda(delta: float, k: float):
    return (delta - gamma(0.0, delta, k)) / k


def variance_bank_account(delta: float, k: float):
    return (delta + gamma(0.0, delta, 2.0 * k) - 2.0 * gamma(0.0, delta, k)) / (k * k)


def beta(ta, tb, k, m=1):  # integral exp(-\int_{t_a}^{t_b} k_s ds)
    delta = (tb - ta)
    if m * delta * k < 1e-08:
        return 1.0 - delta * k
    else:
        return np.exp(- m * delta * k)


def gamma(ta, tb, k):  # integral exp(-\int_{t_a}^{t_b} exp(-\int_{u}^{t_b} k_s ds) ds

    if k * (tb - ta) < 1e-08:
        return np.exp(-k * ta) * (tb - ta)
    else:
        return (np.exp(-k * ta) - np.exp(-k * tb)) / k


def variance(ta, tb, k, eta_i: ndarray) -> ndarray:
    delta = (tb - ta)
    # m = np.exp(-2.0 * delta * k)
    m = gamma(0.0, delta, 2.0 * k)
    return m * np.multiply(eta_i, eta_i)


def get_drift_forward_measure(ta, tb, tp, k, eta_vol_i: ndarray) -> ndarray:
    delta = tb - ta
    m = (delta - gamma(tp - ta - delta, tp - ta, k)) / k
    return m * np.multiply(eta_vol_i, eta_vol_i)


def get_zero_coupon(t_start: float, t_end: float, k: float, ft: ql.ForwardCurve, x: ndarray,
                    y: ndarray) -> ndarray:
    m = ft.discount(t_end) / ft.discount(t_start)
    g = gamma(t_start, t_end, k)
    return m * np.exp(- g * x - 0.5 * g * g * y)


def y_moment_linear_eta_vol(a: float, b: float, k: float, t: float):
    e_2_t = b * b * gamma(0.0, t, 2.0 * k)
    e_4_t_1 = (a * np.power(b, 3.0) / (k * k)) * (gamma(0, t, 2.0 * k) + t * np.exp(- 2.0 * k * t) - 2.0 * gamma(t, 2.0 * t, k))
    e_4_t_2 = 0.5 * (np.power(a * b, 2.0) / k) * (gamma(0.0, t, 2.0 * k) -  t * np.exp(- 2.0 * k * t))
    e_4_t_3 = 0.5 * (np.power(b * a, 2.0) / k) * (gamma(0.0, t, 2.0 * k) - t * np.exp(- 2.0 * k * t))
    return e_2_t + e_4_t_1 + e_4_t_2 + e_4_t_3

def y_moment_quadratic_eta_vol(a: float, b: float, c: float, k: float, t: float):
    e_2_t = c * c * gamma(0.0, t, 2.0 * k)
    e_4_t_1 = (b * np.power(c, 3.0) / (k * k)) * (gamma(0, t, 2.0 * k) + t * np.exp(- 2.0 * k * t) - 2.0 * gamma(t, 2.0 * t, k))
    e_4_t_2 = 0.5 * (np.power(b * c, 2.0) / k) * (gamma(0.0, t, 2.0 * k) -  t * np.exp(- 2.0 * k * t))
    e_4_t_3 =  (a * np.power(c, 3.0) / k) * (gamma(0.0, t, 2.0 * k) - t * np.exp(- 2.0 * k * t))
    return e_2_t + e_4_t_1 + e_4_t_2 + e_4_t_3


def x_moment_linear_eta_vol(a: float, b: float, k: float, t: float):
    g2t = gamma(0.0, t, k)
    e_2_t = 0.5 * (b * b / k) * (g2t - gamma(t, 2.0 * t, k))
    m4_1 = (- t * np.exp(- 2.0 * k * t)  - 2.0 * t * np.exp(- k * t)  + 2.5 * gamma(t, 2.0 * t, k) + 0.5 *  gamma(0.0,  t, k)) / k

    e_4_t_1 = (a * np.power(b, 3.0) / (k * k)) * m4_1

    m4_2 = (t * np.exp(- 2.0 * k * t) + 0.5 * gamma(0.0, t, k) - 1.5 * gamma(t, 2.0 * t, k)) / k
    e_4_t_2 = 0.5 * (np.power(a * b, 2.0) / k) *  m4_2
    e_4_t_3 = 0.5 * (np.power(b * a, 2.0) / k) * m4_2

    return e_2_t + e_4_t_1 + e_4_t_2 + e_4_t_3

def linear_lv_gamma_future(s: float, k: float, a: float, b: float):
    return 0.5 * a * a * b * gamma(0.0, s, k)

def linear_lv_gamma_arrears(s: float, k: float, a: float, b: float, tp=0):
    return 0.5 * a * a * b * gamma(0.0, s, k) - (a * b * b / k) * (gamma(0.0, s, k) - 0.5 * gamma(tp - s, tp + s, k))


def linear_lv_gamma_square_arrears(s: float, k: float, a: float, b: float, tp=0.0):
    return (np.power(a * b, 2.0) * gamma(0.0, s, 2.0 * k) + np.power(a * b, 2.0) * gamma(0.0, s, k)
            - 2.0 * (np.power(b, 3.0) * a / k) * (gamma(0.0, s, k) - 0.5 * gamma(tp - s, tp + s, k)))


def ca_linear_lv_future_fras(t0: float, ta: float, tb: float, k: float, a: float, b: float, ft: ql.ForwardCurve):
    delta = (tb - ta)
    df_ta = ft.discount(ta)
    df_tb = ft.discount(tb)
    m = (gamma(0, (tb - t0), k) - gamma(0, (ta - t0), k)) * df_ta / (df_tb * delta)

    f_convexity = lambda t: np.exp(-k * (t0 - t)) * (b + linear_lv_gamma_future(t, k, a, b)) * gamma(0.0, tb - t, k)

    integral_value = integrate.quad(f_convexity, 0.0, t0)

    return m * b * integral_value[0]


def ca_linear_lv_arrears_fras(ta: float, tb: float, k: float, a: float, b: float, ft: ql.ForwardCurve):
    f_convexity = lambda t: (b * b + linear_lv_gamma_square_arrears(t, k, a, b, ta)) * (
                gamma(0.0, tb - t, k) - gamma(0.0, ta - t, k)) ** 2
    integral_value = integrate.quad(f_convexity, 0.0, ta)

    return integral_value[0]
