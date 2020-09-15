import numpy as np
import numba as nb
import matplotlib.pylab as plt

from ncephes import ndtri, ndtr
from Tools import VolatilityEstimators
from Tools import Types


# numba function to get market paths
@nb.jit("(f8,f8,f8,f8,f8,f8,i8,i8,i8)", nopython=True, nogil=True)
def get_paths(p0: float, sigma0: float, t: float, theta: float, w: float, k: float,
              no_paths: int, no_time_steps: int, seed: int):

    paths = np.empty(shape=(no_paths, no_time_steps))
    v_t = np.empty(shape=(no_paths, no_time_steps))

    paths[:, 0] = p0
    v_t[:, 0] = sigma0 * sigma0

    t_i_s = np.linspace(0.0, t, no_time_steps)
    np.random.rand(seed)
    nu = np.sqrt(2.0 * k * theta)

    for i in range(0, no_paths):
        u_s = np.random.rand(no_time_steps)
        u_sigma = np.random.rand(no_time_steps)

        for j in range(1, no_time_steps):
            delta_time = (t_i_s[j] - t_i_s[j - 1])
            z_sigma_i = ndtri(u_sigma[j])
            z_s_i = ndtri(u_s[j])
            exp_t = np.exp(- theta * delta_time)
            v_t[i, j] = v_t[i, j - 1] * exp_t + w * (1.0 - exp_t) + \
                        nu * np.sqrt(0.5 * ((1.0 - exp_t * exp_t) / theta)) * v_t[i, j - 1] * z_sigma_i
            paths[i, j] = paths[i, j - 1] + np.sqrt(v_t[i, j - 1]) * np.sqrt(delta_time) * z_s_i

    return paths, v_t, t_i_s


# market simulation
theta = 0.035
w = 0.6365
k = 0.2962

p0 = np.log(100)
sigma0 = np.sqrt(0.6365)

t = 1
seed = 123456

no_time_steps = 365*4
no_paths = 5000
freq_sampling = 1
# Simulated integrated variance
paths, v_t, t_i_s = get_paths(p0, sigma0, t, theta, w, k, no_paths, no_time_steps, seed)


# Integrated variance estimator
model_integrated_variance = VolatilityEstimators.get_integrated_variance_from_sim(v_t,
                                                                                  t_i_s,
                                                                                  no_paths)

fourier_estimator = VolatilityEstimators.get_integrated_variance_estimator(paths,
                                                                           no_paths,
                                                                           freq_sampling,
                                                                           t_i_s,
                                                                           Types.ESTIMATOR_TYPE.INTEGRATED_VARIANCE_FOURIER)

empirical_estimator = VolatilityEstimators.get_integrated_variance_estimator(paths,
                                                                             no_paths,
                                                                             freq_sampling,
                                                                             t_i_s,
                                                                             Types.ESTIMATOR_TYPE.INTEGRATED_VARIANCE_EMPIRICAL)
# Estimator's Statistics
# From simulation
mean_simulation = np.mean(model_integrated_variance)
std_simulation = np.std(model_integrated_variance)

# Fourier
mean_fourier = np.mean(fourier_estimator)
std_fourier = np.std(fourier_estimator)

# RV
mean_rv = np.mean(empirical_estimator)
std_rv = np.std(empirical_estimator)

# bias
bias_empirical = (empirical_estimator - model_integrated_variance) / model_integrated_variance
rbias_empirical = np.mean(bias_empirical)
rrmse_empirical = np.sqrt(np.mean(bias_empirical * bias_empirical))

bias_fourier = (fourier_estimator - model_integrated_variance) / model_integrated_variance
rbias_fourier = np.mean(bias_fourier)
rrmse_fourier = np.sqrt(np.mean(bias_fourier * bias_fourier))

# Histogram of the integrated variance estimator

a_fourier = bias_fourier.min()
b_fourier = bias_fourier.max()

a_empirical = bias_empirical.min()
b_empirical = bias_empirical.max()

a = np.min([a_fourier, a_empirical])
b = np.max([b_fourier, b_empirical])

bins = np.linspace(a, b, 30)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].hist(bias_fourier, bins, density=True, color="white", ec="black")
axs[0].set_title("RV estimator")
axs[0].set_ylabel("Observation frequency four times per day")
axs[1].hist(bias_empirical, bins, density=True, color="white", ec="black", label="Bias RV estimator")
axs[1].set_title("Fourier estimator")
axs[1].set_ylabel("Observation frequency four times per day")
# plt.hist(bias_empirical, bins, alpha=0.5, label='bias RV integrated variance')


plt.text(0.8, 0.8, 'RBIAS='+str("{:e}".format(rbias_empirical)), fontsize=10, horizontalalignment='center',
         verticalalignment='center', transform=axs[0].transAxes)

plt.text(0.8, 0.9, 'RRMSE='+str("{:e}".format(rrmse_empirical)), fontsize=10, horizontalalignment='center',
         verticalalignment='center', transform=axs[0].transAxes)

plt.text(0.8, 0.8, 'RBIAS='+str("{:e}".format(rbias_fourier)), fontsize=10, horizontalalignment='center',
         verticalalignment='center', transform=axs[1].transAxes)

plt.text(0.8, 0.9, 'RRMSE='+str("{:e}".format(rrmse_fourier)), fontsize=10, horizontalalignment='center',
         verticalalignment='center', transform=axs[1].transAxes)

plt.show()


