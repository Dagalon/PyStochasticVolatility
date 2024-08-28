import QuantLib as ql
import numpy as np
from Tools.Types import ndarray
from MC_Engines.MC_Cheyette import CheyetteTools


def get_fair_swap_rate(float_tenor_period: ql.Period, fixed_tenor_period: ql.Period,
                       start_date: ql.Date, tenor_period: ql.Period, ft: ql.ForwardCurve,
                       k: float, x_t: ndarray, y_t: ndarray):

    fixing_day_counter = ql.Thirty360()
    curve_day_couter = ft.dayCounter()
    ref_date = ft.referenceDate()

    no_paths = len(x_t)

    fixed_schedule = ql.MakeSchedule(start_date, start_date + tenor_period, fixed_tenor_period, None, ql.TARGET(),
                                     ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Backward,
                                     False)

    float_schedule = ql.MakeSchedule(start_date, start_date + tenor_period, float_tenor_period, None, ql.TARGET(),
                                     ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Backward,
                                     False)

    fixed_dates = list(fixed_schedule)
    float_dates = list(float_schedule)

    no_fixed_dates = len(fixed_dates)
    annuity = np.zeros(no_paths)
    der_annuity = np.zeros(no_paths)

    # annuity
    for i in range(no_fixed_dates - 1):
        t0_fix = curve_day_couter.yearFraction(ref_date, start_date)
        t1_fix = curve_day_couter.yearFraction(ref_date, fixed_dates[i + 1])
        delta_time = fixing_day_counter.yearFraction(fixed_dates[i], fixed_dates[i + 1])
        df = CheyetteTools.get_zero_coupon(t0_fix, t1_fix, k, ft, x_t, y_t)
        annuity = annuity + delta_time * df
        der_annuity = der_annuity - delta_time * CheyetteTools.gamma(t0_fix, t1_fix, k) * df

    # floating leg
    t0_float = curve_day_couter.yearFraction(ref_date, start_date)
    t1_float = curve_day_couter.yearFraction(ref_date, float_dates[-1])
    df_last_floating_date = CheyetteTools.get_zero_coupon(t0_float, t1_float, k, ft, x_t, y_t)

    fairRate = (1.0 - df_last_floating_date) / annuity
    der_fairRate = CheyetteTools.gamma(t0_float, t1_float, k) * df_last_floating_date / annuity - fairRate * der_annuity / annuity

    return fairRate, der_fairRate


def get_annuity(value_date: ql.Date, fixed_tenor_period: ql.Period, start_date: ql.Date, end_date: ql.Date,
                ft: ql.ForwardCurve, k: float, x_t: ndarray, y_t: ndarray) -> tuple:
    fixing_day_counter = ql.Thirty360()
    curve_day_counter = ft.dayCounter()
    ref_date = ft.referenceDate()
    t0 = curve_day_counter.yearFraction(ref_date, value_date)

    no_paths = len(x_t)

    fixed_schedule = ql.MakeSchedule(start_date, end_date, fixed_tenor_period, None, ql.TARGET(),
                                     ql.ModifiedFollowing, ql.ModifiedFollowing, ql.DateGeneration.Backward,
                                     False)

    fixed_dates = list(fixed_schedule)

    no_fixed_dates = len(fixed_dates)
    annuity = np.zeros(no_paths)
    der_annuity = np.zeros(no_paths)

    # annuity
    for i in range(no_fixed_dates - 1):
        t0_fix = curve_day_counter.yearFraction(ref_date, start_date)
        t1_fix = curve_day_counter.yearFraction(ref_date, fixed_dates[i + 1])
        delta_time = fixing_day_counter.yearFraction(fixed_dates[i], fixed_dates[i + 1])
        zero_coupon = CheyetteTools.get_zero_coupon(t0, t1_fix, k, ft, x_t, y_t)
        annuity = annuity + delta_time * zero_coupon
        der_annuity = der_annuity - delta_time * zero_coupon * CheyetteTools.gamma(t0_fix, t1_fix, k)

    return annuity, der_annuity


def get_M_tp(value_date: ql.Date, fixed_tenor_period: ql.Period, start_date: ql.Date, end_date: ql.Date,
             ft: ql.ForwardCurve, k: float, x_t: ndarray, y_t: ndarray, t_p: ql.Date) -> tuple:
    curve_day_counter = ft.dayCounter()
    ref_date = ft.referenceDate()

    dt0 = curve_day_counter.yearFraction(ref_date, value_date)
    dtp = curve_day_counter.yearFraction(ref_date, t_p)

    annuity_values = get_annuity(value_date, fixed_tenor_period, start_date, end_date, ft, k, x_t, y_t)
    dftp = CheyetteTools.get_zero_coupon(dt0, dtp, k, ft, x_t, y_t)

    m = dftp / annuity_values[0]
    der_m = - np.multiply(m, CheyetteTools.gamma(dt0, dtp, k) + annuity_values[1] / annuity_values[0])

    return m, der_m


def get_malliavin_cms_convexity_vol_flat(ta: float, der_swap: float, m: float, der_m: float, k: float, sigma: float):
    return (der_swap * der_m) * (sigma * sigma / m) * CheyetteTools.gamma(0, ta, 2.0 * k)

