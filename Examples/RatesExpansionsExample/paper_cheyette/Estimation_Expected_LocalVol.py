import numpy as np
import QuantLib as ql
from MC_Engines.MC_Cheyette import Cheyette_Engine, CheyetteTools
from Tools import RNG
from Tools.Types import ndarray
from Tools.Types import CHEYETTE_OUTPUT
import matplotlib.pylab as plt
from scipy.integrate import quad


# estimation of the local vol
def mean_bar_x_linear_vol(a_t: float, b_t: float, k_t: float, t: float):
    # f = lambda s: np.exp(-k_t * (t - s)) * CheyetteTools.gamma(0.0, t-s, k_t) * (a_t * CheyetteTools.x_moment_linear_eta_vol(a_t, b_t, k_t, s) + b_t) ** 2
    f = lambda s: np.exp(-k_t * (t - s)) * CheyetteTools.y_moment_linear_eta_vol(a_t, b_t, k_t, s)
    integral = quad(f, 0.0, t)
    return integral[0]


def estimation_quadratic_linear_local_vol(a_t: float, b_t: float, k_t: float, t: float) -> ndarray:
    # mean \bar{x}_t
    mean_bar_x = mean_bar_x_linear_vol(a_t, b_t, k_t, t)

    term1 = (a_t * mean_bar_x + b_t)**2
    # f1 = lambda s: (np.exp(- 2.0 * k_t * (t - s)) * 2.0 * a_t * a_t * (a_t * mean_bar_x_linear_vol(a_t, b_t, k_t, s) + b_t)**2)
    f1 = lambda s: (np.exp(- 2.0 * k_t * (t - s)) * 2.0 * a_t * a_t * (a_t * mean_bar_x_linear_vol(a_t, b_t, k_t, s) + b_t) ** 2)
    integral1 = quad(f1, 0.0, t)

    # term2 = 0.5 * CheyetteTools.gamma(0.0, t, 2.0 * k) * 2.0 * a_t * a_t * b_t * b_t
    term2 = 0.5 * integral1[0]
    return term1 + term2


# linear local volatility
a = 0.3
b = 0.02


# linear local volatility
def linear_eta_vol(a_t: float, b_t: float, t: float, x_t: ndarray, y_t: ndarray) -> ndarray:
    return a_t * x_t + b_t


# mc info
no_paths = 1000000
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
                                             lambda ti, x, y: linear_eta_vol(a, b, ti, x, y), rnd_generator)

x_mc_moment = []
y_mc_moment = []
quadratic_local_vol = []
quadratic_local_vol_app = []

for j, t in enumerate(tis[1:]):
    x_mc_moment.append(np.mean(output[CHEYETTE_OUTPUT.PATHS_X][:, j + 1]))
    y_mc_moment.append(np.mean(output[CHEYETTE_OUTPUT.PATHS_Y][:, j + 1]))
    quadratic_local_vol.append(np.mean(np.power(linear_eta_vol(a, b, t, output[CHEYETTE_OUTPUT.PATHS_X][:, j + 1],
                                                               output[CHEYETTE_OUTPUT.PATHS_Y][:, j + 1]), 2.0)))

    quadratic_local_vol_app.append(estimation_quadratic_linear_local_vol(a, b, k, t))


# plots

plt.plot(tis[1:], quadratic_local_vol, label='eta^2(t,x,y) by MC', linestyle='--')
plt.plot(tis[1:], quadratic_local_vol_app, label='eta^2(t,x,y) by approximation', linestyle='--')

# plt.ylim(0.0001, 0.001)
plt.title(f' E(eta^2(t,x_t,y_t))  with a={a} and b={b}')
plt.xlabel('T')
plt.legend()

plt.show()
