import numpy as np
import numba as nb

from MC_Engines.MC_RBergomi import ToolsVariance
from Tools import Functionals
from Tools.Types import Vector, ndarray, SABR_OUTPUT, TYPE_STANDARD_NORMAL_SAMPLING


def get_v_t_sampling(t: float,
                     h: float,
                     nu: float,
                     z: ndarray):

    std_t = np.sqrt(ToolsVariance.get_volterra_covariance(t, h))
    return std_t * z


def get_path_multi_step(t0: float,
                        t1: float,
                        parameters: Vector,
                        f0: float,
                        no_paths: int,
                        no_time_steps: int,
                        type_random_number: TYPE_STANDARD_NORMAL_SAMPLING,
                        rnd_generator) -> map:

    nu = parameters[0]
    rho = parameters[1]
    h = parameters[2]

    no_paths = 2 * no_paths if type_random_number == TYPE_STANDARD_NORMAL_SAMPLING.ANTITHETIC else no_paths

    t_i_s = np.linspace(t0, t1, no_time_steps)

    s_t = np.empty((no_paths, no_time_steps))
    s_t[:, 0] = f0

    cov = ToolsVariance.get_covariance_matrix(t_i_s, h, rho)
    z_i_s = rnd_generator.normal(mu=0.0, sigma=1.0, size=(2 * no_time_steps - 1, no_paths))












