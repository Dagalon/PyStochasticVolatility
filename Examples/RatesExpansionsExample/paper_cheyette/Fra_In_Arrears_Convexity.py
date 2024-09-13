import numpy as np
import QuantLib as ql

from MC_Engines.MC_Cheyette import Cheyette_Engine, CheyetteTools
from Tools import RNG
from Tools.Types import ndarray
from Tools.Types import CHEYETTE_OUTPUT
import matplotlib.pylab as plt

# linear local volatility
a = 0.3
b = 0.015


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
no_time_steps = np.floor(104 * t) + 1
nodes_number = np.floor(4 * t) + 1
tis = np.linspace(0.0, t, int(nodes_number))

# tenor ois future
tenor = 1.0

# forward curve
dates = [ql.Date(12, 8, 2024), ql.Date(12, 8, 2084)]
rates = [0.03, 0.03]
ft = ql.ForwardCurve(dates, rates, ql.Actual360(), ql.TARGET())

# outputs
convexity_adjustment_mc = []
convexity_adjustment_app = []
convexity_test = []
fras_arrear = []
fras = []

scale = 26.0

for j, t in enumerate(tis[1:]):
    rnd_generator = RNG.RndGenerator(seed)

    # new time steps for each monte carlo
    no_time_steps = int(scale * tis[j+1]) + 1
    ts_j = np.linspace(0.0, tis[j+1], int(no_time_steps))

    # paths
    output = Cheyette_Engine.get_path_multi_step_forward_measure(ts_j, x0, y0, ft, k, no_paths,
                                                                 lambda ti, x, y: linear_eta_vol(a, b, ti, x, y),
                                                                 rnd_generator)

    df = CheyetteTools.get_zero_coupon(ts_j[-1], ts_j[-1] + tenor, k, ft,
                                       output[CHEYETTE_OUTPUT.PATHS_X][:, -1],
                                       output[CHEYETTE_OUTPUT.PATHS_Y][:, -1])

    fra_arrears = (1.0 / df - 1.0) / tenor
    fra = (ft.discount(t) / ft.discount(t + tenor) - 1.0) / tenor
    fras.append(fra)
    fras_arrear.append(np.mean(fra_arrears))

    ca = fras_arrear[-1] - fra
    ca_approximation = CheyetteTools.ca_linear_lv_arrears_fras(t, t + tenor, k, a, b, ts_j[-1])
    convexity_adjustment_mc.append(ca)
    convexity_adjustment_app.append(ca_approximation)

# plots
plt.plot(tis[1:], convexity_adjustment_mc, label='CA Montecarlo', linestyle='--')
plt.plot(tis[1:], convexity_adjustment_app, label='CA Malliavin', linestyle='--')

plt.title(f'Convexity adjustment FRA Arrears with a={a} and b={b}')
plt.xlabel('T')
plt.legend()

plt.show()
