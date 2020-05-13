import matplotlib.pylab as plt
import numpy as np

from MC_Engines.GenericSDE.SDE import cev_sigma, cev_drift, z_sigma, z_drift
from MC_Engines.GenericSDE.SDESimulation import sde_euler_simulation
from Tools.CEVMalliavinTools import get_error
from Tools.Types import EULER_SCHEME_TYPE

mu = 0.0
sigma = 0.2
rho = 0.6
y0 = 100.0
z_0 = np.power(y0, 1.0 - rho)

t0 = 0.0
t1 = 1.0

no_paths = 50000
n_max = 13

f_drift = lambda t, x: cev_drift(mu, t, x)
f_sigma = lambda t, x: cev_sigma(sigma, rho, t, x)

z_t_drift = lambda t, x: z_drift(mu, rho, sigma, t, x)
z_t_sigma = lambda t, x: z_sigma(sigma, rho, t, x)

n_step_max = 2**n_max
z = np.random.standard_normal(size=(no_paths, n_step_max - 1))

exact_paths_cev = sde_euler_simulation(t0, t1, y0, n_step_max, no_paths, z, f_drift,
                                       f_sigma, EULER_SCHEME_TYPE.STANDARD)

mean_by_no_steps_transform = []
variance_by_no_steps_transform = []

mean_by_no_steps = []
variance_by_no_steps = []

for i in range(4, n_max-1):
    no_steps = 2 ** i
    scale_factor = 2 ** (n_max - i)
    nodes_index = [scale_factor * k for k in range(0, int(2 ** i) + 1)]
    nodes_index[-1] -= 2
    z_nodes = z[:, nodes_index]

    y_t_n_paths_cev = sde_euler_simulation(t0, t1, y0, no_steps, no_paths, z_nodes, f_drift,
                                           f_sigma, EULER_SCHEME_TYPE.STANDARD)
    z_t_paths_cev = sde_euler_simulation(t0, t1, z_0, no_steps, no_paths, z_nodes, z_t_drift,
                                         z_t_sigma, EULER_SCHEME_TYPE.STANDARD)

    paths_transform_cev = np.power(z_t_paths_cev, 1.0 / (1.0 - rho))
    errors_transform = get_error(paths_transform_cev[:, no_steps-1], exact_paths_cev[:, n_step_max-1])
    errors = get_error(y_t_n_paths_cev[:, no_steps-1], exact_paths_cev[:, n_step_max-1])
    
    mean_transform_error = np.mean(errors_transform)
    var_transform_error = np.mean(np.power(errors_transform, 2.0)) - mean_transform_error * mean_transform_error

    mean_error = np.mean(errors)
    var_error = np.mean(np.power(errors, 2.0)) - mean_transform_error * mean_transform_error

    mean_by_no_steps.append(mean_error)
    variance_by_no_steps.append(var_error)

    mean_by_no_steps_transform.append(mean_transform_error)
    variance_by_no_steps_transform.append(var_transform_error)


n_i = list(range(4, n_max-1))
plt.plot(n_i, mean_by_no_steps, label='variance error')
plt.plot(n_i, mean_by_no_steps_transform, label='variance error with transform')

plt.legend()
plt.title('mean (y_t - y_n_t) evolution with each number of time steps')
plt.show()
