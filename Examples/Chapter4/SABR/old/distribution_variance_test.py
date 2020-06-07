import numpy as np

from MC_Engines.MC_SABR import VarianceMC, VarianceSamplingMatchingMoment
from MC_Engines.MC_SABR.SABR_Engine import get_vol_sampling
from Tools import RNG

alpha = 0.4
nu = 0.4
rho = -0.4
no_paths = 1000
seed = 123456789
no_deep = 10
T = 1.0

rnd_generator = RNG.RndGenerator(seed)
z = rnd_generator.normal(0.0, 1.0, no_paths)

alpha_T = get_vol_sampling(0.0, T, alpha, nu, z)
v_t = VarianceMC.get_variance(alpha, nu, alpha_T, T, no_deep, rnd_generator)

v_t_matching = VarianceSamplingMatchingMoment.get_variance(alpha,
                                                           nu,
                                                           alpha_T,
                                                           T,
                                                           rnd_generator.normal_sobol(0.0, 1.0, no_paths))
mean_t = np.mean(v_t)
var_t = np.mean(np.power(v_t, 2.0)) - mean_t * mean_t

mean_t_matching = np.mean(v_t_matching)
var_t_matching = np.mean(np.power(v_t_matching, 2.0)) - mean_t_matching * mean_t_matching






