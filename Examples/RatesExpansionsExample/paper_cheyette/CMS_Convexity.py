import QuantLib as ql
import numpy as np
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
periodToSimulate = ql.Period("7D")
startDate = ql.Date(14, 8, 2024)
endDate = startDate + ql.Period("10Y")
dates = []
currentDate = startDate

while currentDate <= endDate:
    currentDate = currentDate + periodToSimulate
    dates.append(currentDate)

dc = ql.Actual365Fixed()
tis = [dc.yearFraction(startDate, t) for t in dates]

# forward curve
dates = [ql.Date(12, 8, 2024), ql.Date(12, 8, 2084)]
rates = [0.03, 0.03]
ft = ql.ForwardCurve(dates, rates, ql.Actual360(), ql.TARGET())

# initial conditions
x0 = 0.0
y0 = 0.0
k = 0.00075
t = dc.yearFraction(startDate, endDate)
no_time_steps = np.floor(52 * t) + 1

# mc info
no_paths = 5
seed = 123456789
rnd_generator = RNG.RndGenerator(seed)

# linear local volatility
a = 0.3
b = 0.01


def linear_eta_vol(a_t: float, b_t: float, t: float, x_t: ndarray, y_t: ndarray) -> ndarray:
    return a_t * x_t + b_t


# paths
output = Cheyette_Engine.get_path_multi_step(tis, x0, y0, ft, k, no_paths,
                                             lambda ti, x, y: linear_eta_vol(a, b, ti, x, y), rnd_generator)

for j, t in enumerate(tis[1:]):
    fair_swap_rate = SwapTools.get_fair_swap_rate(floatPeriod, fixedPeriod, startDate, tenor, ft, k,
                                                  output[CHEYETTE_OUTPUT.PATHS_X][:, j + 1],
                                                  output[CHEYETTE_OUTPUT.PATHS_Y][:, j + 1])
    mean_fair_swap_rate = fair_swap_rate.mean()

    s0 = SwapTools.get_fair_swap_rate(floatPeriod, fixedPeriod, startDate, tenor, ft, k,
                                      np.zeros((1, 1)),
                                      np.zeros((1, 1)))[0, 0]
