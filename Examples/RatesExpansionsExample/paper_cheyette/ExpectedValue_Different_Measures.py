import numpy as np
import QuantLib as ql
from scipy.integrate import quad

from MC_Engines.MC_Cheyette import Cheyette_Engine, CheyetteTools
from Tools import RNG
from Tools.Types import ndarray
from Tools.Types import CHEYETTE_OUTPUT
import matplotlib.pylab as plt
from scipy.integrate import quad


# estimation of the local vol
def mean_bar_x_linear_vol(a_t: float, b_t: float, k_t: float, t: float, t_p: float):
    # f = lambda s: np.exp(-k_t * (t - s)) * CheyetteTools.gamma(0.0, t-s, k_t) * (a_t * CheyetteTools.x_moment_linear_eta_vol(a_t, b_t, k_t, s) + b_t) ** 2
    f = lambda s: np.exp(-k_t * (t - s)) * CheyetteTools.y_moment_linear_eta_vol_tp(a_t, b_t, k_t, s, t_p)
    integral = quad(f, 0.0, t)
    return integral[0]


def gamma_adjustment_convexity(a_t: float, b_t: float, k_t: float, s: float, t_p: float):
    # term1
    f1 = lambda t: np.exp(- 2.0 * k_t * (s - t)) * 2.0 * a_t * a_t * (a_t * CheyetteTools.mean_bar_x_linear_vol_forward_measure(a_t, b_t, k_t, t, t_p) + b_t)**2
    integral1 = quad(f1, 0.0, s)
    term1 = 0.5 * integral1[0]

    # term2
    f2 = lambda t: 2.0 * (np.exp(- k_t * (s - t)) * a_t * (a_t * CheyetteTools.mean_bar_x_linear_vol_forward_measure(a_t, b_t, k_t, t, t_p) + b_t) *
                          CheyetteTools.gamma(0.0, t_p - t, k_t) * (
                                      a_t * CheyetteTools.mean_bar_x_linear_vol_forward_measure(a_t, b_t, k_t, t, t_p) + b) ** 2)
    integral2 = quad(f2, 0.0, s)
    term2 = 0.5 * integral2[0]

    return term1 - term2


def estimation_ca_linear_local_vol(a_t: float, b_t: float, k_t: float, t: float, t_p: float) -> ndarray:
    f1 = lambda s: (2.0 * b_t * a_t * CheyetteTools.gamma(t - s, t + t_p - 2.0 * s, k_t)
                    * ((a_t * mean_bar_x_linear_vol(a_t, b_t, k_t, s, t_p) + b_t) ** 2 + gamma_adjustment_convexity(a_t, b_t, k_t, s, t_p)))
    integral1 = quad(f1, 0.0, t)

    return integral1[0]


# linear local volatility
a = 0.2
b = 0.015


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

tis = [0.25 + i * 0.25 for i in range(0, 20)]

x_mc_moment_spot = []
y_mc_moment_spot = []

x_mc_moment_forward = []
y_mc_moment_forward = []

quadratic_local_vol_spot_measure = []
quadratic_local_vol_forward_measure = []
ca = []
quadratic_local_approximation = []

scale = 26

for j, t in enumerate(tis):
    rnd_spot_generator = RNG.RndGenerator(seed)
    rnd_forward_generator = RNG.RndGenerator(seed)

    # new time steps for each monte carlo
    no_time_steps = int(scale * t) + 1
    ts_j = np.linspace(0.0, t, int(no_time_steps))

    # paths
    output_spot_measure = Cheyette_Engine.get_path_multi_step(ts_j, x0, y0, ft, k, no_paths,
                                                              lambda ti, x, y: linear_eta_vol(a, b, ti, x, y),
                                                              rnd_spot_generator)

    output_forward_measure = Cheyette_Engine.get_path_multi_step(ts_j, x0, y0, ft, k, no_paths,
                                                                 lambda ti, x, y: linear_eta_vol(a, b, ti, x, y),
                                                                 rnd_forward_generator)

    x_mc_moment_spot.append(np.mean(output_spot_measure[CHEYETTE_OUTPUT.PATHS_X][:, -1]))
    y_mc_moment_spot.append(np.mean(output_spot_measure[CHEYETTE_OUTPUT.PATHS_Y][:, -1]))

    x_mc_moment_forward.append(np.mean(output_forward_measure[CHEYETTE_OUTPUT.PATHS_X][:, -1]))
    y_mc_moment_forward.append(np.mean(output_forward_measure[CHEYETTE_OUTPUT.PATHS_Y][:, -1]))

    quadratic_local_vol_spot_measure.append(
        np.mean(np.power(linear_eta_vol(a, b, t, output_spot_measure[CHEYETTE_OUTPUT.PATHS_X][:, -1],
                                        output_spot_measure[CHEYETTE_OUTPUT.PATHS_Y][:, -1]), 2.0)))

    quadratic_local_vol_forward_measure.append(
        np.mean(np.power(linear_eta_vol(a, b, t, output_forward_measure[CHEYETTE_OUTPUT.PATHS_X][:, -1],
                                        output_forward_measure[CHEYETTE_OUTPUT.PATHS_Y][:, -1]), 2.0)))

    ca.append(estimation_ca_linear_local_vol(a, b, k, t, t))

    quadratic_local_approximation.append(quadratic_local_vol_spot_measure[-1] - ca[-1])

# plots
plt.plot(tis, quadratic_local_vol_forward_measure, label='E(eta^2(t,x,y)) by MC', linestyle='--')
plt.plot(tis, quadratic_local_approximation, label='E(eta^2(t,x,y))  by approximation', linestyle='--')

# plt.ylim(0.0001, 0.001)
plt.title(f' E(eta^2(t,x_t,y_t))  with a={a} and b={b}')
plt.xlabel('T')
plt.legend()

plt.show()
