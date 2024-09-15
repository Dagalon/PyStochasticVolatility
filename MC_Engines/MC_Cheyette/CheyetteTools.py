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


def gamma(ta, tb, k):
    # integral exp(-\int_{t_a}^{t_b} exp(-\int_{u}^{t_b} k_s ds) ds

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


# linear local volatility moments

def y_moment_linear_eta_vol(a: float, b: float, k: float, t: float):
    e_2_t = b * b * gamma(0.0, t, 2.0 * k)
    e_4_t_1 = 0.5 * np.power(a * b, 2.0) * (gamma(0.0, t, 2.0 * k) - np.exp(- 2.0 * k * t) * t) / k
    e_4_t_2 = np.power(a * b, 2.0) * (gamma(0.0, t, 2.0 * k) - gamma(t, 2.0 * t, k)) / (2.0 * k)
    e_4_t_3 = b * np.power(a * b / k, 2.0) * (
            np.exp(- 2.0 * k * t) * t - 2.0 * gamma(t, 2.0 * t, k) + gamma(0.0, t, 2.0 * k))

    return e_2_t + e_4_t_1 + e_4_t_2 + e_4_t_3


def y_moment_linear_eta_vol_tp(a: float, b: float, k: float, t: float, t_p: float):
    e_2_t = b * b * gamma(0.0, t, 2.0 * k)
    e_4_t_1 = 0.5 * np.power(a * b, 2.0) * (gamma(0.0, t, 2.0 * k) - np.exp(- 2.0 * k * t) * t) / k
    e_4_t_2 = np.power(a * b, 2.0) * (gamma(0.0, t, 2.0 * k) - gamma(t, 2.0 * t, k)) / (2.0 * k)
    e_4_t_3 = b * np.power(a * b / k, 2.0) * (
            np.exp(- 2.0 * k * t) * t - 2.0 * gamma(t, 2.0 * t, k) + gamma(0.0, t, 2.0 * k))

    e_4_t_4_a = 2.0 * a * b * np.power(b / k, 2.0) * (gamma(0.0, t, 2.0 * k) - gamma(t, 2.0 * t, k))
    e_4_t_4_b = a * b * np.power(b / k, 2.0) * (
                gamma(t_p - t, t_p + 2.0 * t, k) / 3.0 - gamma(t_p + t, t_p + 2.0 * t, k))

    return e_2_t + e_4_t_1 + e_4_t_2 + e_4_t_3 - e_4_t_4_a + e_4_t_4_b


def x_moment_linear_eta_vol(a: float, b: float, k: float, t: float):
    g2t = gamma(0.0, t, k)
    e_2_t = 0.5 * (b * b / k) * (g2t - gamma(t, 2.0 * t, k))

    e_4_t_1 = 0.5 * np.power(a * b / k, 2.0) * (
            t * np.exp(- 2.0 * k * t) - 1.5 * gamma(t, 2.0 * t, k) + 0.5 * gamma(0.0, t, k))
    e_4_t_2 = (0.5 * np.power(a * b, 2.0) / (k * k)) * (
            0.5 * gamma(0.0, t, k) + 0.5 * gamma(t, 2.0 * t, k) - t * np.exp(- k * t))
    e_4_t_3 = b * np.power(a * b / k, 2.0) * (0.5 * gamma(0.0, t, k) + 2.5 * gamma(t, 2.0 * t, k)
                                              - 2.0 * np.exp(- k * t) * t - np.exp(- 2.0 * k * t) * t)

    return e_2_t + e_4_t_1 + e_4_t_2 + e_4_t_3


def x_moment_linear_eta_vol_tp(a: float, b: float, k: float, t: float, t_p: float):
    g2t = gamma(0.0, t, k)
    e_2_t = 0.5 * (b * b / k) * (g2t - gamma(t, 2.0 * t, k))

    e_4_t_1 = 0.5 * np.power(a * b / k, 2.0) * (
            t * np.exp(- 2.0 * k * t) - 1.5 * gamma(t, 2.0 * t, k) + 0.5 * gamma(0.0, t, k))
    e_4_t_2 = (0.5 * np.power(a * b, 2.0) / (k * k)) * (
            0.5 * gamma(0.0, t, k) + 0.5 * gamma(t, 2.0 * t, k) - t * np.exp(- k * t))
    e_4_t_3 = b * np.power(a * b / k, 2.0) * (0.5 * gamma(0.0, t, k) + 2.5 * gamma(t, 2.0 * t, k)
                                              - 2.0 * np.exp(- k * t) * t - np.exp(- 2.0 * k * t) * t)

    # extra term Girsanov's drift
    e_4_first_term = 2.0 * (a * b / k) * np.power(b / k, 2.0) * (0.5 * gamma(0.0, t, k)
                                                                 + 0.5 * gamma(t, 2.0 * t, k) - np.exp(-k * t) * t)

    e_4_second_term = (a * b / k) * np.power(b / k, 2.0) * (
                - np.exp(-k * (t + t_p)) * t + gamma(t + t_p, 2.0 * t + t_p, k)
                + gamma(t_p - t, t + t_p, k) / 6.0 - gamma(t_p + t, 2.0 * t + t_p, k) / 3.0)

    return e_2_t + e_4_t_1 + e_4_t_2 + e_4_t_3 - e_4_first_term + e_4_second_term


# ----------------------------------------------------------------------------------------------------------------------

# quadratic local volatility moments


def y_moment_quadratic_eta_vol(a: float, b: float, c: float, k: float, t: float):
    e_2_t = c * c * gamma(0.0, t, 2.0 * k)
    e_4_t_1 = (b * np.power(c, 3.0) / (k * k)) * (
            gamma(0, t, 2.0 * k) + t * np.exp(- 2.0 * k * t) - 2.0 * gamma(t, 2.0 * t, k))
    e_4_t_2 = 0.5 * (np.power(b * c, 2.0) / k) * (gamma(0.0, t, 2.0 * k) - t * np.exp(- 2.0 * k * t))
    e_4_t_3 = (a * np.power(c, 3.0) / k) * (gamma(0.0, t, 2.0 * k) - t * np.exp(- 2.0 * k * t))
    return e_2_t + e_4_t_1 + e_4_t_2 + e_4_t_3


# ----------------------------------------------------------------------------------------------------------------------


def linear_lv_gamma_future(s: float, k: float, a: float, b: float):
    # return (np.power(a * b, 2.0) * gamma(0.0, s, 2.0 * k) - 2.0 * (a * np.power(b, 3.0) / k)
    #         * (gamma(0.0, s, k) - 0.5 * gamma(b - s, b + s, k)))
    # return 0.5 * a * a * b * gamma(0.0, s, k)
    f1 = lambda t: (np.exp(- 2.0 * k * (s - t)) * 2.0 * a * a * b * b)
    integral1 = integrate.quad(f1, 0.0, s)
    term1 = 0.5 * integral1[0]

    # term2
    f2 = lambda t: 2.0 * (np.exp(- k * (s - t)) * (a * mean_bar_x_linear_vol(a, b, k, t) + b) *
                          gamma(0.0, b - t, k) * (a * mean_bar_x_linear_vol(a, b, k, t) + b) ** 2)
    integral2 = integrate.quad(f2, 0.0, s)
    term2 = 0.5 * integral2[0]

    return term1 - term2


def linear_lv_gamma_square_arrears(s: float, k: float, a: float, b: float, tp=0.0):
    # term1
    f1 = lambda t: (np.exp(- 2.0 * k * (s - t)) * 2.0 * a * a * (a * mean_bar_x_linear_vol_forward_measure(a, b, k, t, tp) + b)**2)
    integral1 = integrate.quad(f1, 0.0, s)
    term1 = 0.5 * integral1[0]

    # term2
    f2 = lambda t: 2.0 * (np.exp(- k * (s - t)) * a * (a * mean_bar_x_linear_vol_forward_measure(a, b, k, t, tp) + b) *
                          gamma(0.0, tp - t, k) * (a * mean_bar_x_linear_vol_forward_measure(a, b, k, t, tp) + b) ** 2)
    integral2 = integrate.quad(f2, 0.0, s)
    term2 = 0.5 * integral2[0]

    return term1 - term2


def ca_linear_lv_future_fras(t0: float, ta: float, tb: float, k: float, a: float, b: float, ft: ql.ForwardCurve):
    delta = (tb - ta)
    df_ta = ft.discount(ta)
    df_tb = ft.discount(tb)
    m = (gamma(0, (tb - t0), k) - gamma(0, (ta - t0), k)) * df_ta / (df_tb * delta)

    f_convexity = lambda t: np.exp(-k * (t0 - t)) * (b * b + linear_lv_gamma_future(t, k, a, b)) * gamma(0.0, tb - t, k)

    integral_value = integrate.quad(f_convexity, 0.0, t0)

    return m * integral_value[0]


def mean_bar_x_linear_vol(a_t: float, b_t: float, k_t: float, t: float):
    f = lambda s: np.exp(-k_t * (t - s)) * gamma(0.0, t - s, k_t) * (
                a_t * x_moment_linear_eta_vol(a_t, b_t, k_t, s) + b_t) ** 2
    integral = integrate.quad(f, 0.0, t)
    return integral[0]


def mean_bar_x_linear_vol_forward_measure(a_t: float, b_t: float, k_t: float, t: float, t_p: float):
    f = lambda s: np.exp(-k_t * (t - s)) * y_moment_linear_eta_vol_tp(a_t, b_t, k_t, s, t_p)
    integral = integrate.quad(f, 0.0, t)
    return integral[0]


def ca_linear_lv_arrears_fras(ta: float, tb: float, k: float, a: float, b: float, t_p: float):
    f_moment = lambda t: mean_bar_x_linear_vol_forward_measure(a, b, k, t, t_p)
    f_convexity = lambda t: (((a * f_moment(t) + b) ** 2 + linear_lv_gamma_square_arrears(t, k, a, b, t_p))
                             * (gamma(0.0, tb - t, k) - gamma(0.0, ta - t, k)) ** 2)
    integral_value = integrate.quad(f_convexity, 0.0, ta)

    return integral_value[0]
