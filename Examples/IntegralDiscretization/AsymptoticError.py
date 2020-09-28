from Tools.RNG import RndGenerator
from MC_Engines.MC_SABR import SABR_Engine
from Tools import Types
from Examples.IntegralDiscretization import AsymptoticTools

import numpy as np

# SABR parameters
alpha = 0.4
nu = 0.00001
rho = 0.0
parameters = [alpha, nu, rho]

# Simulation parameters
f0 = 100.0
T0 = 0.0
T1 = 1.0

# MC parameters
no_paths = 100000
min_level = 5
no_level = 11
no_time_steps_benchmark = int(2**no_level)
seed = 123456
rng = RndGenerator(seed)

# Compute mean and variance benchmark case
map_out = SABR_Engine.get_path_multi_step(T0, T1, parameters, f0, no_paths, no_time_steps_benchmark,
                                          Types.TYPE_STANDARD_NORMAL_SAMPLING, rng)


# Mean and variance asymptotics
asymptotic_output = AsymptoticTools.get_sabr_asymptotic(map_out[Types.SABR_OUTPUT.SIGMA_PATHS],
                                                        map_out[Types.SABR_OUTPUT.TIMES],
                                                        np.array(parameters),
                                                        no_paths,
                                                        no_time_steps_benchmark)

error_mean_discretization = {}
error_variance_discretization = {}

for i in range(min_level, no_level + 1):
    no_time_steps = int(2**i)
    index_i = [int(2**(no_level - i)) * k - 1 for k in range(0, 2**i + 1)]
    index_i[0] = 0

    output = AsymptoticTools.get_discretization(map_out[Types.SABR_OUTPUT.PATHS][:, index_i],
                                                map_out[Types.SABR_OUTPUT.TIMES][index_i],
                                                np.array(parameters),
                                                no_paths,
                                                no_time_steps)

    error_mean_discretization[i] = output[0] * no_time_steps
    error_variance_discretization[i] = output[1] * no_time_steps * 0.25

print("hola")
