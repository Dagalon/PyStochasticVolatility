import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit

from MC_Engines.MC_RBergomi import RBergomi_Engine
from Tools import Types
from Tools import RNG
from AnalyticEngines.VolatilityTools import NonParametricEstimatorSLV

# simulation info
h = 0.4
nu = 0.5
rho = 0.0
v0 = 0.05
sigma_0 = np.sqrt(v0)

parameters = [nu, rho, h]

f0 = 100
T = np.arange(7, 90, 2) * 1.0 / 365.0

seed = 123456789

no_time_steps = 100
no_paths = 500000

# Compute the conditional expected by kernel estimators
delta = 0.01
x = np.zeros(3)
x[0] = np.log(f0) - delta
x[1] = np.log(f0)
x[2] = np.log(f0) + delta

atm_lv = []
atm_lv_skew = []
atm_lv_skew_derive_estimator = []

rnd_generator = RNG.RndGenerator(seed)

for t_i in T:
    rnd_generator.set_seed(seed)
    # Julien Guyon paper https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032&download=yes
    h = 1.5 * np.sqrt(t_i) * np.power(no_paths, - 0.2)

    # Rbergomi paths simulation
    map_bergomi_output = RBergomi_Engine.get_path_multi_step(0.0, t_i, parameters, f0, v0, no_paths,
                                                             no_time_steps, Types.TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC,
                                                             rnd_generator)


    gaussian_estimator_t = NonParametricEstimatorSLV.gaussian_kernel_estimator_slv(map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
                                                                                   np.log(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1]),
                                                                                   x,
                                                                                   h)
    skew_gaussian_estimator_t = NonParametricEstimatorSLV.gaussian_kernel_estimator_skew_slv(
        map_bergomi_output[Types.RBERGOMI_OUTPUT.VARIANCE_SPOT_PATHS][:, -1],
        np.log(map_bergomi_output[Types.RBERGOMI_OUTPUT.PATHS][:, -1]),
        np.asfortranarray(np.log(f0)),
        h)

    atm_lv.append(np.sqrt(gaussian_estimator_t[1]))
    skew = 0.5 * (gaussian_estimator_t[2] - gaussian_estimator_t[0]) / delta
    # atm_lv_skew.append(skew)
    atm_lv_skew.append(0.5 * skew / np.sqrt(gaussian_estimator_t[1]))
    # atm_lv_skew_derive_estimator.append(skew_gaussian_estimator_t[0])


def f_law(x, b, c):
    return b * np.power(x, c)


popt_atm_lv_skew, pcov_diff_vols_swap = curve_fit(f_law, T[1:], atm_lv_skew[1:])
y_fit_atm_lv_skew = f_law(T, *popt_atm_lv_skew)


plt.plot(T[1:], atm_lv_skew[1:], label="atm skew LV", color="blue", linestyle="dotted")
plt.plot(T[1:], y_fit_atm_lv_skew[1:], label=" %s * T^(%s)" % (round(popt_atm_lv_skew[0], 5),
         round(popt_atm_lv_skew[1], 5)), color="green", linestyle="dotted")
# plt.plot(x, atm_lv_skew_derive_estimator, label="derivative estimator", color="black", linestyle="dotted")
plt.xlabel("T")
plt.ylabel("d E(V_t|S_t=x) / dx")


plt.legend()
plt.show()
