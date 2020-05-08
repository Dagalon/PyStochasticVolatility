from Tools import fBM, RNG

no_paths = 10
no_time_steps = 20
t0 = 0.0
t1 = 1.0
z0 = 0.0
hurst_parameter = 0.7
seed = 123456789

rng = RNG.RndGenerator(seed)
paths = fBM.cholesky_method(t0, t1, z0, rng, hurst_parameter, no_paths, no_time_steps)

