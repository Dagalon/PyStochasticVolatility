import numpy as np
import QuantLib as ql
from MC_Engines.MC_Cheyette import Cheyette_Engine, CheyetteTools
from Tools import RNG
from Tools.Types import ndarray
from Tools.Types import CHEYETTE_OUTPUT
import matplotlib.pylab as plt
from scipy.special import ndtri

# linear local volatility
a = 0.3
b = 0.2
c = 0.01

# linear local volatility
def quadratic_eta_vol(a_t: float, b_t: float, c_t: float, t: float, x_t: ndarray, y_t: ndarray) -> ndarray:
    return a_t * np.power(x_t, 2.0)  + b_t * x_t + c_t


# mc info
no_paths = 500000
seed = 123
rnd_generator = RNG.RndGenerator(seed)

# initial conditions
x0 = 0.0
y0 = 0.0
k = 0.00075
t = 5.0
no_time_steps = np.floor(26 * t) + 1

# tenor ois future
tenor = 1.0

# forward curve
dates = [ql.Date(12, 8, 2024), ql.Date(12, 8, 2084)]
rates = [0.03, 0.03]
ft = ql.ForwardCurve(dates, rates, ql.Actual360(), ql.TARGET())

tis = np.linspace(0.0, t, int(no_time_steps))

# paths
output = Cheyette_Engine.get_path_multi_step(tis, x0, y0, ft, k, no_paths,
                                             lambda ti, x, y: quadratic_eta_vol(a, b, c, ti, x, y), rnd_generator)

x_mc_moment = []
y_mc_moment = []
upper_y = []
lower_y = []
alpha = 0.95
y_std_mc = []
y_approx_moment = []
x_approx_moment = []

for j, t in enumerate(tis[1:]):
    # x_mc_moment.append(np.mean(output[CHEYETTE_OUTPUT.PATHS_X][:, j + 1]))
    y_mc_moment.append(np.mean(output[CHEYETTE_OUTPUT.PATHS_Y][:, j + 1]))

    y_approx_moment.append(CheyetteTools.y_moment_quadratic_eta_vol(a, b, c, k, t))
    # x_approx_moment.append(CheyetteTools.x_moment_linear_eta_vol(a, b, k, t))

# plots
# plt.plot(tis[1:], y_mc_moment, label='y_t mean MC', linestyle='--')
# plt.plot(tis[1:], x_mc_moment, label='x_t mean MC', linestyle='--')

plt.plot(tis[1:], y_mc_moment, label='y_t mean mc', linestyle='--')
plt.plot(tis[1:], y_approx_moment, label='y_t mean approximation', linestyle='--')
# plt.plot(tis[1:], upper_y, label='y_t upper mc', linestyle='--')
# plt.plot(tis[1:], lower_y, label='y_t lower mc', linestyle='--')

plt.title(f' Upper and lower band y_t  with a={a} and b={b}')
plt.xlabel('T')
plt.legend()

plt.show()
