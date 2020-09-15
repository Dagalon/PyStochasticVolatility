import matplotlib.pylab as plt
from Tools import RNG
from FractionalBrownian import fBM, ToolsFBM

no_paths = 1
no_time_steps = 2 ** 11
t0 = 0.0
t1 = 1.0
z0 = 0.0
seed = 123456789
rng = RNG.RndGenerator(seed)

fig, axs = plt.subplots(1, 3, figsize=(7, 3))

# Low Hurst parameter
low_hurst_parameter = 0.3
low_hurst_parameter_paths = fBM.cholesky_method(t0, t1, z0, rng, low_hurst_parameter, no_paths, int(no_time_steps * t1))
low_output_estimation = ToolsFBM.get_estimator_rs(low_hurst_parameter_paths[0, :], 5, 10)

axs[0].plot(low_output_estimation[2], low_output_estimation[3], color='black')
axs[0].plot(low_output_estimation[2], low_output_estimation[4], linestyle='dashed', color='black')
axs[0].set_title('y='+str(round(low_output_estimation[0], 4)) + '+' + str(round(low_output_estimation[1], 4)) + '*x')

# Medium Hurst parameter
rng.set_seed(seed)
medium_hurst_parameter = 0.5
medium_hurst_parameter_paths = fBM.cholesky_method(t0, t1, z0, rng, medium_hurst_parameter, no_paths, int(no_time_steps * t1))
medium_output_estimation = ToolsFBM.get_estimator_rs(medium_hurst_parameter_paths[0, :], 5, 10)

axs[1].plot(medium_output_estimation[2], medium_output_estimation[3], color='black')
axs[1].plot(medium_output_estimation[2], medium_output_estimation[4], linestyle='dashed', color='black')
axs[1].set_title('y='+str(round(medium_output_estimation[0], 4)) + '+' + str(round(medium_output_estimation[1], 4)) + '*x')

# Large Hurst parameter
rng.set_seed(seed)
large_hurst_parameter = 0.7
large_hurst_parameter_paths = fBM.cholesky_method(t0, t1, z0, rng, large_hurst_parameter, no_paths, int(no_time_steps * t1))
large_output_estimation = ToolsFBM.get_estimator_rs(large_hurst_parameter_paths[0, :], 5, 10)

axs[2].plot(large_output_estimation[2], large_output_estimation[3], color='black')
axs[2].plot(large_output_estimation[2], large_output_estimation[4], linestyle='dashed', color='black')

axs[2].set_title('y='+str(round(large_output_estimation[0], 4)) + '+' + str(round(large_output_estimation[1], 4)) + '*x')


plt.show()
