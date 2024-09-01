import QuantLib as ql
import numpy as np
import matplotlib.pylab as plt

from Tools import RNG
from Tools.Types import ndarray
from Tools.Types import CHEYETTE_OUTPUT
from MC_Engines.MC_Cheyette import Cheyette_Engine, CheyetteTools
import SwapTools

# general info
tenor = ql.Period("5Y")

# fix leg
fixedPeriod = ql.Period("1Y")

# float leg
floatPeriod = ql.Period("6M")

# period to simulate
periodToSimulate = ql.Period("3M")
startDate = ql.Date(14, 8, 2024)
endDate = startDate + ql.Period("10Y")
dates = []
currentDate = startDate

while currentDate <= endDate:
    currentDate = currentDate + periodToSimulate
    dates.append(currentDate)

dc = ql.Actual365Fixed()
tis = [dc.yearFraction(startDate, t) for t in dates]
scale = 26.0

# forward curve
dates_curve = [startDate, startDate + ql.Period("60Y")]
rates = [0.03, 0.03]
ft = ql.ForwardCurve(dates_curve, rates, ql.Actual360(), ql.TARGET())

# initial conditions
x0 = 0.0
y0 = 0.0
k = 0.00075
t = dc.yearFraction(startDate, endDate)

# mc info
no_paths = 750000
seed = 123456789

# linear local volatility
a = 0.0
b = 0.0
c = 0.03


def quadratic_eta_vol(a_t: float, b_t: float, c_t: float, t: float, x_t: ndarray, y_t: ndarray) -> ndarray:
    return a_t * np.power(x_t, 2.0) + b_t * x_t + c_t


convexity_adjustment_mc = []
convexity_adjustment_malliavin = []

for j, t in enumerate(tis):
    rnd_generator = RNG.RndGenerator(seed)

    # new time steps for each monte carlo
    no_time_steps = int(scale * tis[j]) + 1
    ts_j = np.linspace(0.0, tis[j], int(no_time_steps))

    # paths
    output = Cheyette_Engine.get_path_multi_step_forward_measure(ts_j, x0, y0, ft, k, no_paths,
                                                                 lambda ti, x, y: quadratic_eta_vol(a, b, c, ti, x, y),
                                                                 rnd_generator)

    fair_swap_rate = SwapTools.get_fair_swap_rate(floatPeriod, fixedPeriod, dates[j], tenor, ft, k,
                                                  output[CHEYETTE_OUTPUT.PATHS_X][:, -1],
                                                  output[CHEYETTE_OUTPUT.PATHS_Y][:, -1])[0]
    mean_fair_swap_rate = fair_swap_rate.mean()

    swap_values = SwapTools.get_fair_swap_rate(floatPeriod, fixedPeriod, dates[j], tenor, ft, k,
                                      np.zeros(1),
                                      np.zeros(1))

    s0 = swap_values[0][0]
    der_s0 = swap_values[1][0]

    convexity_adjustment_mc.append(mean_fair_swap_rate - s0)

    # Malliavin convexity
    annuity_values = SwapTools.get_annuity(dates[j], fixedPeriod, dates[j], dates[j] + tenor,
                                           ft, k, np.zeros(1), np.zeros(1))

    annuity_t0 = SwapTools.get_annuity(startDate, fixedPeriod, dates[j], dates[j] + tenor,
                                       ft, k, np.zeros(1), np.zeros(1))

    M_values_ta = SwapTools.get_M_tp(dates[j], fixedPeriod, dates[j], dates[j] + tenor,
                                     ft, k, np.zeros(1), np.zeros(1), dates[j])

    ca_malliavin = SwapTools.get_malliavin_cms_convexity_vol_flat(ts_j[-1], der_s0, M_values_ta[0], M_values_ta[1], k, c)

    convexity_adjustment_malliavin.append(ca_malliavin)


# plots
plt.plot(tis, convexity_adjustment_mc, label='CA Montecarlo', linestyle='--')
plt.plot(tis, convexity_adjustment_malliavin, label='CA Malliavin', linestyle='--')

plt.title(f'Convexity adjustment CMS 5Y  with a={a}, b={b} and c={c}')
plt.xlabel('T')
plt.legend()

plt.show()