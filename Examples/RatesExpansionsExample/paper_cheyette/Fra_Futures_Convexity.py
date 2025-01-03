import numpy as np
import QuantLib as ql
from MC_Engines.MC_Cheyette import Cheyette_Engine, CheyetteTools
from Tools import RNG
from Tools.Types import ndarray
from Tools.Types import CHEYETTE_OUTPUT
import matplotlib.pylab as plt

# linear local volatility
a = 0.3
b = 0.02


# linear local volatility
def linear_eta_vol(a_t: float, b_t: float, t: float, x_t: ndarray, y_t: ndarray) -> ndarray:
    return a_t * x_t + b_t


# mc info
no_paths = 1000000
seed = 123456789
rnd_generator = RNG.RndGenerator(seed)

# initial conditions
x0 = 0.0
y0 = 0.0
k = 0.00075
t = 5.0
no_time_steps = np.floor(52 * t) + 1

# tenor ois future
tenor = 1.0

# forward curve
dates = [ql.Date(12, 8, 2024), ql.Date(12, 8, 2084)]
rates = [0.03, 0.03]
ft = ql.ForwardCurve(dates, rates, ql.Actual360(), ql.TARGET())

tis = np.linspace(0.0, t, int(no_time_steps))

# paths
output = Cheyette_Engine.get_path_multi_step(tis, x0, y0, ft, k, no_paths,
                                             lambda ti, x, y: linear_eta_vol(a, b, ti, x, y),
                                             rnd_generator)

convexity_adjustment_mc = []
convexity_adjustment_app = []
fras_future = []
fras = []


for j, t in enumerate(tis[1:]):
    df = CheyetteTools.get_zero_coupon(t, t + tenor, k, ft,
                                       output[CHEYETTE_OUTPUT.PATHS_X][:, j + 1],
                                       output[CHEYETTE_OUTPUT.PATHS_Y][:, j + 1])

    fra_future = (1.0 / df - 1.0) / tenor
    fra = (ft.discount(t) / ft.discount(t + tenor) - 1.0) / tenor
    fras.append(fra)
    fras_future.append(np.mean(fra_future))

    ca = fras_future[-1] - fra
    ca_approximation = CheyetteTools.ca_linear_lv_future_fras(t, t, t + tenor, k, a, b, ft)

    convexity_adjustment_app.append(ca_approximation)
    convexity_adjustment_mc.append(ca)

# plots
plt.plot(tis[1:], convexity_adjustment_mc, label='CA Montecarlo', linestyle='--')
plt.plot(tis[1:], convexity_adjustment_app, label='CA Malliavin', linestyle='--')

plt.title(f'Convexity adjustment FRA Futures with a={a} and b={b}')
plt.xlabel('T')
plt.legend()

plt.show()
