import numpy as np
import time
import matplotlib.pylab as plt

from MC_Engines.MC_SABR import VarianceSamplingMatchingMoment
from MC_Engines.MC_SABR import SABR_Engine
from Tools.Types import TYPE_STANDARD_NORMAL_SAMPLING, SABR_OUTPUT
from Tools import RNG
from AnalyticEngines.MalliavinMethod.ExpansionTools import  get_vol_swap_new_approximation
from Instruments.EuropeanInstruments import EuropeanOption, TypeSellBuy, TypeEuropeanOption

# model parameters
alpha = 0.5
nu = 0.8
rho = -0.6
parameters = [alpha, nu, rho]
f0 = 100.0
t = 2.0

# simulation info
no_paths = 500000
no_time_steps = 300
seed = 12345
rnd_generator = RNG.RndGenerator(seed)

start_time = time.time()
output = SABR_Engine.get_path_multi_step(0.0, t, parameters, f0, no_paths, no_time_steps,
                                         TYPE_STANDARD_NORMAL_SAMPLING.REGULAR_WAY, rnd_generator)
end_time = time.time()
diff = (end_time - start_time)
print(diff)

sampling_I_t = np.sum(output[SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS], axis=1)
vol_swap = np.sqrt(np.mean(sampling_I_t) / t)
vol_swap_approximation = get_vol_swap_new_approximation(parameters, 0.0, t)

# one step
rnd_generator.set_seed(seed)
z_int_t = rnd_generator.normal(0.0, 1.0, no_paths)
z_t = rnd_generator.normal(0.0, 1.0, no_paths)
alpha_t0 = np.full(no_paths, alpha, dtype=np.float)
alpha_t = SABR_Engine.get_vol_sampling(0.0, t, alpha_t0, nu, z_t)

approx_I_t = VarianceSamplingMatchingMoment.get_variance(alpha_t0, nu, alpha_t, t, z_int_t)
vol_swap_match = np.sqrt(np.mean(approx_I_t) / t)

# paths one step
rnd_generator.set_seed(seed)
start_time = time.time()
paths_one_step = SABR_Engine.get_path_one_step(0.0, t, parameters, f0, no_paths, rnd_generator)
end_time = time.time()
diff = (end_time - start_time)
print(diff)

# comparison of distributions
a = np.minimum(np.min(sampling_I_t), np.min(approx_I_t))
b = np.maximum(np.max(sampling_I_t), np.max(approx_I_t))
b=15

bins = np.linspace(a, b, 200)

plt.hist(approx_I_t, bins, label='approximation')
plt.hist(sampling_I_t, bins, label='empirical')

plt.legend()
plt.show()

# option price comparison
# no_strikes = 40
# strikes = np.linspace(50.0, 160.0, no_strikes)
# prices_multistep = []
# prices_onestep = []
#
# for i in range(0, no_strikes):
#     option = EuropeanOption(strikes[i], 1.0, TypeSellBuy.BUY, TypeEuropeanOption.CALL, f0, t)
#     prices_multistep.append(option.get_price_control_variate(output[SABR_OUTPUT.PATHS][:, -1],
#                                                              output[SABR_OUTPUT.INTEGRAL_VARIANCE_PATHS])[0])
#     prices_onestep.append(option.get_price(paths_one_step[:])[0])
#
#
# plt.plot(strikes, prices_onestep, label='sabr one step')
# plt.plot(strikes, prices_multistep, label='sabr multi step')
# plt.title("T= %s" % t)
# plt.xlabel("K")
#
# plt.legend()
# plt.show()






