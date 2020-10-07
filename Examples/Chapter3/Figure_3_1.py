import matplotlib.pylab as plt
import numpy as np

from MC_Engines.GenericSDE.SDE import derive_cev_drift, derive_cev_sigma
from MC_Engines.GenericSDE.SDESimulation import sde_euler_simulation
from Solvers.ODESolver import ode_euler_solver_malliavin
from Solvers.ODE import a_cev
from Tools.VolatilityTools.CEVMalliavinTools import get_error, transform_cev_malliavin
from MC_Engines.GenericSDE.SDE import cev_sigma, cev_drift, z_drift, z_sigma
from Tools.Types import EULER_SCHEME_TYPE

mu = 0.0
sigma = 0.2
rho = 0.6
y0 = 10
derive_y0 = sigma * np.power(y0, rho)

d_z_0 = sigma * (1.0 - rho)
z_0 = np.power(y0, 1.0 - rho)

t0 = 0.0
t1 = 1.0

no_paths = 10000
n_max = 14

f_derive_drift = lambda t, x: derive_cev_drift(mu, t, x)
f_derive_sigma = lambda t, x: derive_cev_sigma(sigma, rho, t, x)

f_drift = lambda t, x: cev_drift(mu, t, x)
f_sigma = lambda t, x: cev_sigma(sigma, rho, t, x)

z_t_drift = lambda t, x: z_drift(mu, rho, sigma, t, x)
z_t_sigma = lambda t, x: z_sigma(sigma, rho, t, x)
a_f = lambda t, x: a_cev(rho, mu, sigma, d_z_0, t, x)

# Aplico directamente la derivada directamente sobre el proceso de difusion CEV y simulo con frecuencia alta para
# obtener el proceso derivada "exacto"

n_step_max = 2**n_max
z = np.random.standard_normal(size=(no_paths, n_step_max - 1))
derive_paths_cev = sde_euler_simulation(t0, t1, derive_y0, n_step_max, no_paths, z, f_derive_drift,
                                        f_derive_sigma, EULER_SCHEME_TYPE.STANDARD)

mean_by_no_steps = []
variance_by_no_steps = []

mean_transform_by_no_steps = []
variance_transform_by_no_steps = []

for i in range(4, n_max-1):
    no_steps = 2 ** i
    scale_factor = 2 ** (n_max - i)
    nodes_index = [scale_factor * k for k in range(0, int(2 ** i) + 1)]
    nodes_index[-1] -= 2
    z_nodes = z[:, nodes_index]

    # We solve the SDE associated with transformed process
    z_t = sde_euler_simulation(t0, t1, z_0, no_steps, no_paths, z_nodes, z_t_drift,
                               z_t_sigma, EULER_SCHEME_TYPE.STANDARD)

    # We solve the ODE associated with transformed process
    ode_derive_z_t = ode_euler_solver_malliavin(t0, t1, d_z_0, no_steps, z_t, a_f)

    # Using D_0Z_t, Z_t and inverse transform we build D_0_Y_t
    derive_transform_path_cev = transform_cev_malliavin(rho, z_t, ode_derive_z_t)
    d_y_t_n = sde_euler_simulation(t0, t1, derive_y0, no_steps, no_paths, z_nodes, f_derive_drift,
                                   f_derive_sigma, EULER_SCHEME_TYPE.STANDARD)

    errors_transform = get_error(derive_transform_path_cev[:, no_steps - 1], derive_paths_cev[:, n_step_max - 1])
    errors = get_error(d_y_t_n[:, no_steps - 1], derive_paths_cev[:, n_step_max - 1])

    mean_error = np.mean(errors)
    var_error = np.mean(np.power(errors, 2.0)) - mean_error * mean_error

    mean_transform_error = np.mean(errors_transform)
    var_transform_error = np.mean(np.power(errors_transform, 2.0)) - mean_transform_error * mean_transform_error

    mean_by_no_steps.append(mean_error)
    variance_by_no_steps.append(var_error)

    mean_transform_by_no_steps.append(mean_transform_error)
    variance_transform_by_no_steps.append(var_transform_error)


n_i = list(range(4, n_max-1))
plt.plot(n_i, mean_by_no_steps, label='mean error', color='black')
plt.plot(n_i, mean_transform_by_no_steps, label='mean error with transform', linestyle='dashed', color='black')

plt.legend()
# plt.title('mean (d_y_t - d_y_n_t) evolution with each number of time steps')
plt.show()
