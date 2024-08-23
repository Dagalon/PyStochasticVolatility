import QuantLib as ql
import numpy as np
from Tools.Types import ndarray
from MC_Engines.MC_Cheyette import CheyetteTools


def get_fair_swap_rate(float_tenor_period: ql.Period, fixed_tenor_period: ql.Period,
                       start_date: ql.Date, tenor_period: ql.Period, ft: ql.FlatForward,
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
    annuity = np.empty((no_paths, 1), dtype=np.float64)

    # annuity
    for i in range(no_fixed_dates - 1):
        t0_fix = curve_day_couter.yearFraction(ref_date, start_date)
        t1_fix = curve_day_couter.yearFraction(ref_date, fixed_dates[i + 1])
        delta_time = fixing_day_counter.yearFraction(fixed_dates[i], fixed_dates[i + 1])
        annuity = annuity + delta_time * CheyetteTools.get_zero_coupon(t0_fix, t1_fix, k, ft, x_t, y_t)

    # floating leg
    t0_float = curve_day_couter.yearFraction(ref_date, start_date)
    t1_float = curve_day_couter.yearFraction(ref_date, float_dates[-1])
    df_last_floating_date = CheyetteTools.get_zero_coupon(t0_float, t1_float, k, ft, x_t, y_t)

    return (1.0 - df_last_floating_date) / annuity
