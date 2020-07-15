from MC_Engines.MC_RBergomi import ToolsVariance
from Tools.RNG import RndGenerator

hurst_parameter = 0.3

s = 0.1
t = 0.3

covariance = ToolsVariance.get_volterra_covariance(s, t, hurst_parameter)

t0 = 0.0
t1 = 2.0
n = 5
no_paths = 10
h = 0.3
rho = 0.5

rnd = RndGenerator(1234567)

bridge_sampling = ToolsVariance.get_path_gaussian_bridge(t0,
                                                         t1,
                                                         n,
                                                         no_paths,
                                                         h,
                                                         rho,
                                                         rnd)
