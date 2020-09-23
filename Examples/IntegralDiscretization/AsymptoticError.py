from Tools.RNG import RndGenerator
from MC_Engines.MC_SABR import SABR_Engine
from Tools import Types
from Examples.IntegralDiscretization import AsymptoticTools

import numpy as np

# SABR parameters
alpha = 0.4
nu = 0.1
rho = 0.0
parameters = [alpha, nu, rho]

# Simulation parameters
f0 = 100.0
T0 = 0.0
T1 = 1.0

# MC parameters
no_paths = 100000
no_nodes = 20
k_s = np.arange(1, no_nodes + 1, 1)
no_time_steps = 50 * k_s
no_time_steps_benchmark = 4*365
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

error_mean_discretization = []
error_variance_discretization = []

for i in range(0, no_nodes):
    rng.set_seed(seed)
    map_out = SABR_Engine.get_path_multi_step(T0, T1, parameters, f0, no_paths, no_time_steps[i],
                                              Types.TYPE_STANDARD_NORMAL_SAMPLING, rng)

    output = AsymptoticTools.get_discretization(map_out[Types.SABR_OUTPUT.PATHS],
                                                map_out[Types.SABR_OUTPUT.TIMES],
                                                np.array(parameters),
                                                no_paths,
                                                no_time_steps[i])

    error_mean_discretization.append(output[0])
    error_variance_discretization.append(output[1])

print("hola")
